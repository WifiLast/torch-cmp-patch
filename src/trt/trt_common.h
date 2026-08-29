// =================================================================================
// trt_common.h
//
// Shared infrastructure for every op's runtime TensorRT path (conv3d,
// conv2d, matmul, attention -- see each op's own <op>_trt_runtime.cpp).
// Extracted to one place specifically so the trickiest, correctness-
// critical parts (content-addressed engine caching, the device->host copy
// needed to compute that hash, TensorRT object lifetime) exist ONCE
// instead of being re-copied (and re-risked) into four separate files.
//
// UNVERIFIED: no TensorRT SDK anywhere in the environment this was written
// in, targeting Linux (the stated deployment) from a Windows sandbox that
// can't compile-check the Linux build either. Read this file's design
// rationale before trusting it, and validate for real -- correctness
// (matching the hand-tuned kernel's output, not just "did it run") before
// performance.
//
// DESIGN: build-on-first-use, cache-forever, keyed by the FULL content of
// every weight tensor involved (not just shape) -- reusing an engine built
// for different weights would silently produce wrong results, which is
// worse than any performance cost. This means every op's first call for a
// new (shape, weights) combination pays a real TensorRT build/tactic-
// search cost (can be tens of seconds); every later call with the SAME
// shape+weights is just an enqueue. Chosen because this was built for a
// stated deployment with fixed hardware and (implied by that) a stable,
// long-running process -- amortizing a one-time build cost is the right
// tradeoff there. It would be the WRONG tradeoff for a workload that
// constantly sees new weight tensors (e.g. training, or serving many
// distinct fine-tunes) -- know your workload before enabling this
// (CMPEXT3_USE_TENSORRT=1) for anything beyond a fixed, already-trained
// model.
// =================================================================================
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cmpext3_trt {

nvinfer1::ILogger& logger();

// TensorRT 8.x/9.x interfaces are plain `delete`-able.
template <typename T>
struct Deleter {
    void operator()(T* obj) const { delete obj; }
};
template <typename T>
using Ptr = std::unique_ptr<T, Deleter<T>>;

struct CachedEngine {
    Ptr<nvinfer1::IRuntime> runtime;
    Ptr<nvinfer1::ICudaEngine> engine;
    Ptr<nvinfer1::IExecutionContext> context;
};

// FNV-1a 64-bit -- simple, dependency-free, fast enough for a
// once-per-call cache-key hash over tensors that can run into the tens of
// megabytes. Not cryptographic; a collision (astronomically unlikely for
// FNV-1a over that much essentially-random float data) would mean reusing
// the wrong cached engine -- upgrade to a stronger hash if that's ever a
// real concern for your workload, deliberately not done here to keep this
// dependency-free.
constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
uint64_t fnv1a_update(const void* data, size_t len, uint64_t hash);
std::string hash_to_hex(uint64_t h);

// $XDG_CACHE_HOME/cmpext3/trt_engines or $HOME/.cache/cmpext3/trt_engines
// on Linux (the stated deployment target); %LOCALAPPDATA%\cmpext3\
// trt_engines on Windows, kept for parity with the rest of this codebase's
// dual-platform support elsewhere.
std::string cache_dir();
void mkdir_p(const std::string& path);
bool read_file(const std::string& path, std::vector<char>& out);
bool write_file(const std::string& path, const void* data, size_t size);

// Blocking device->host copy on `stream`, synchronizing before returning
// -- every op needs this to (a) compute the content-hash cache key and
// (b) have host-side Weights data available if a fresh build turns out to
// be necessary. Returns false on any CUDA error.
bool copy_to_host(const void* device_ptr, void* host_ptr, size_t bytes, cudaStream_t stream);

// Get-or-build-or-load a cached engine for `key` (already expected to be a
// full content hash, unique per op -- callers prefix it per-op, e.g.
// "conv3d_<hex>", so different ops' caches never collide even if they
// happened to hash to the same value). Checks the in-process map first,
// then the on-disk file, and only calls `build_fn` (which must return
// serialized engine bytes, or an empty vector to signal failure) on a full
// miss -- persisting the result to disk for reuse across process restarts.
// Returns nullptr if every path failed (including build_fn failing).
std::shared_ptr<CachedEngine> get_or_build_engine(
    const std::string& key,
    const std::function<std::vector<char>()>& build_fn);

// Benchmarks `trt_fn` (must return false on ANY failure -- unusable
// dtype/shape, engine build failure, CUDA error, matching every other TRT
// entry point's "false means try the fallback instead" contract) against
// `fallback_fn` (must always succeed -- the existing hand-tuned kernel)
// for a few iterations each on `stream`, and returns whether TensorRT won.
//
// `key` should be built from shape+dtype+op-parameters ONLY, deliberately
// NOT weight content -- unlike get_or_build_engine's cache key, wall-clock
// speed for a fixed shape doesn't depend on the actual tensor values, so
// one benchmark per shape/dtype covers every weight tensor that shape ever
// sees. This cache is in-process only (not persisted to disk like the
// engine cache above): re-benchmarking after a process restart is cheap
// -- a handful of kernel launches -- since get_or_build_engine's own disk
// cache already makes the TensorRT side's first call fast too.
//
// EVERY call (cache hit or miss) actually runs the winning path and
// leaves a complete, correct result behind -- callers don't need to call
// either function again afterward. If TensorRT was cached as the winner
// but `trt_fn` fails on some later call (e.g. transient OOM), that one
// call falls back silently without disturbing the cached decision, so the
// next call tries TensorRT again rather than getting stuck on one bad run.
bool trt_faster_than_fallback(
    const std::string& key,
    cudaStream_t stream,
    const std::function<bool()>& trt_fn,
    const std::function<void()>& fallback_fn);

}  // namespace cmpext3_trt
