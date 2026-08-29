#include "trt_common.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unordered_map>

#if defined(_WIN32)
#include <direct.h>
#define CMPEXT3_MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define CMPEXT3_MKDIR(path) mkdir(path, 0755)
#endif

using namespace nvinfer1;

namespace cmpext3_trt {

namespace {

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[cmpext3][TensorRT] %s\n", msg);
        }
    }
};

std::mutex g_cache_mutex;
std::unordered_map<std::string, std::shared_ptr<CachedEngine>> g_engine_cache;

}  // namespace

ILogger& logger() {
    static Logger instance;
    return instance;
}

uint64_t fnv1a_update(const void* data, size_t len, uint64_t hash) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hash_to_hex(uint64_t h) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string(buf);
}

std::string cache_dir() {
    std::string base;
#if defined(_WIN32)
    const char* local_appdata = std::getenv("LOCALAPPDATA");
    base = local_appdata ? std::string(local_appdata) + "\\cmpext3\\trt_engines"
                          : std::string(".cmpext3_trt_engines");
#else
    const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache && xdg_cache[0] != '\0') {
        base = std::string(xdg_cache) + "/cmpext3/trt_engines";
    } else {
        const char* home = std::getenv("HOME");
        base = home ? std::string(home) + "/.cache/cmpext3/trt_engines"
                     : std::string(".cache/cmpext3/trt_engines");
    }
#endif
    return base;
}

void mkdir_p(const std::string& path) {
#if defined(_WIN32)
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    std::string accum;
    size_t pos = 0;
    while (pos < path.size()) {
        size_t next = path.find(sep, pos);
        if (next == std::string::npos) next = path.size();
        accum = path.substr(0, next);
        if (!accum.empty()) {
            CMPEXT3_MKDIR(accum.c_str());  // ignore EEXIST/any error here; the
        }                                   // subsequent file open surfaces a
        pos = next + 1;                      // real problem if it truly isn't usable
    }
}

bool read_file(const std::string& path, std::vector<char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize size = f.tellg();
    if (size <= 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    return static_cast<bool>(f.read(out.data(), size));
}

bool write_file(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    return static_cast<bool>(f);
}

bool copy_to_host(const void* device_ptr, void* host_ptr, size_t bytes, cudaStream_t stream) {
    if (bytes == 0) return true;
    if (cudaMemcpyAsync(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return false;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

namespace {

std::shared_ptr<CachedEngine> deserialize_engine(const std::vector<char>& bytes) {
    auto entry = std::make_shared<CachedEngine>();
    entry->runtime.reset(createInferRuntime(logger()));
    if (!entry->runtime) return nullptr;
    entry->engine.reset(entry->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!entry->engine) return nullptr;
    entry->context.reset(entry->engine->createExecutionContext());
    if (!entry->context) return nullptr;
    return entry;
}

}  // namespace

std::shared_ptr<CachedEngine> get_or_build_engine(
    const std::string& key,
    const std::function<std::vector<char>()>& build_fn)
{
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_engine_cache.find(key);
        if (it != g_engine_cache.end()) {
            return it->second;
        }
    }

    std::string dir = cache_dir();
    std::string path = dir + "/" + key + ".trt";

    std::shared_ptr<CachedEngine> entry;
    std::vector<char> bytes;
    if (read_file(path, bytes)) {
        entry = deserialize_engine(bytes);
    }

    if (!entry) {
        bytes = build_fn();
        if (bytes.empty()) return nullptr;

        mkdir_p(dir);
        write_file(path, bytes.data(), bytes.size());  // best-effort; a failed
        // disk cache write still lets this call succeed via the in-process
        // cache, it just won't survive a process restart.

        entry = deserialize_engine(bytes);
        if (!entry) return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_engine_cache[key] = entry;
    }
    return entry;
}

namespace {

std::mutex g_bench_mutex;
std::unordered_map<std::string, bool> g_bench_decisions;

// Average per-call elapsed time (ms) of `fn` on `stream`, timing `iters`
// calls and reporting their average -- but AFTER `warmup` untimed calls,
// and after silently dropping the first TIMED call's own sample too. Two
// separate layers on purpose: `warmup` amortizes generic one-time costs
// (CUDA context/module setup, caching-allocator warmup) that are the same
// for every caller; the dropped first timed sample specifically guards
// against TensorRT's own first-execution cost surviving that -- an
// IExecutionContext can still lazily JIT/load the CUDA module for
// whichever tactic the engine build picked on its OWN first enqueue,
// separate from (and after) the build step itself, so a fixed `warmup`
// count picked without knowing that isn't guaranteed to fully amortize
// it. Each call gets its own start/stop event pair (not one batched
// region around the whole loop) specifically so that first sample can be
// identified and excluded rather than silently averaged in.
float time_calls(const std::function<void()>& fn, cudaStream_t stream, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;
    for (int i = 0; i < iters + 1; ++i) {
        cudaEventRecord(start, stream);
        fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float call_ms = 0.0f;
        cudaEventElapsedTime(&call_ms, start, stop);
        if (i > 0) total_ms += call_ms;  // i == 0: the excluded first timed run
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / static_cast<float>(iters);
}

}  // namespace

bool trt_faster_than_fallback(
    const std::string& key,
    cudaStream_t stream,
    const std::function<bool()>& trt_fn,
    const std::function<void()>& fallback_fn)
{
    {
        std::lock_guard<std::mutex> lock(g_bench_mutex);
        auto it = g_bench_decisions.find(key);
        if (it != g_bench_decisions.end()) {
            if (it->second && trt_fn()) return true;
            // Either the cached decision was "use the kernel", or TensorRT
            // was cached as the winner but just failed THIS call (e.g.
            // transient OOM) -- fall back for this call only, decision
            // cache untouched so the next call tries TensorRT again.
            fallback_fn();
            return false;
        }
    }

    // First call for this shape/dtype: an untimed probe first, so the
    // real one-time engine-build cost (up to tens of seconds, see
    // get_or_build_engine above) doesn't pollute the timed comparison
    // below. Nothing to benchmark if TensorRT can't even run once here.
    if (!trt_fn()) {
        std::lock_guard<std::mutex> lock(g_bench_mutex);
        g_bench_decisions[key] = false;
        fallback_fn();
        return false;
    }

    constexpr int kWarmup = 2;
    constexpr int kIters = 5;
    float trt_ms = time_calls([&]() { trt_fn(); }, stream, kWarmup, kIters);
    float fallback_ms = time_calls(fallback_fn, stream, kWarmup, kIters);
    bool use_trt = trt_ms < fallback_ms;

    fprintf(stderr, "[cmpext3][TensorRT] %s: TensorRT %.3f ms vs. hand-tuned kernel %.3f ms -- using %s\n",
            key.c_str(), trt_ms, fallback_ms, use_trt ? "TensorRT" : "the hand-tuned kernel");

    // Leave a fresh, correct result behind matching the decision. The
    // fallback is timed AFTER TensorRT above, so `output` already holds a
    // correct fallback result no matter which one won -- only need one
    // more call here, and only when TensorRT actually won.
    if (use_trt) {
        trt_fn();
    }

    {
        std::lock_guard<std::mutex> lock(g_bench_mutex);
        g_bench_decisions[key] = use_trt;
    }
    return use_trt;
}

}  // namespace cmpext3_trt
