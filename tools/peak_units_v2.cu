// Peak throughput per instruction type, with every accumulator feeding the
// output so the compiler cannot delete any of the measured work (v1's wmma
// numbers were inflated that way). Verify with cuobjdump -sass.
//   nvcc -O3 -arch=sm_75 tools/peak_units_v2.cu -o bench_logs/peak_units_v2
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda::wmma;
constexpr int IT = 1024;

template <typename AB, typename ACC, int CH>
__global__ void k_mma(float* out, int seed) {
    fragment<matrix_a, 16, 16, 16, AB, row_major> fa[CH];
    fragment<matrix_b, 16, 16, 16, AB, col_major> fb;
    fragment<accumulator, 16, 16, 16, ACC> acc[CH];
    for (int c = 0; c < CH; ++c) { fill_fragment(fa[c], (AB)(seed + c)); fill_fragment(acc[c], (ACC)0); }
    fill_fragment(fb, (AB)seed);
    for (int it = 0; it < IT; ++it)
#pragma unroll
        for (int c = 0; c < CH; ++c) mma_sync(acc[c], fa[c], fb, acc[c]);
    float s = 0;
    for (int c = 0; c < CH; ++c) for (int e = 0; e < acc[c].num_elements; ++e) s += (float)acc[c].x[e];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int CH>
__global__ void k_hfma2(float* out, float a, float b) {
    __half2 acc[CH], ha = __float2half2_rn(a), hb = __float2half2_rn(b);
    for (int c = 0; c < CH; ++c) acc[c] = __float2half2_rn(threadIdx.x * 1e-3f + c);
    for (int i = 0; i < IT * 4; ++i)
#pragma unroll
        for (int c = 0; c < CH; ++c) acc[c] = __hfma2(acc[c], ha, hb);
    float s = 0; for (int c = 0; c < CH; ++c) s += __low2float(acc[c]) + __high2float(acc[c]);
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int CH>
__global__ void k_ffma(float* out, float a, float b) {
    float acc[CH];
    for (int c = 0; c < CH; ++c) acc[c] = threadIdx.x + c;
    for (int i = 0; i < IT * 4; ++i)
#pragma unroll
        for (int c = 0; c < CH; ++c) acc[c] = fmaf(acc[c], a, b);
    float s = 0; for (int c = 0; c < CH; ++c) s += acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <typename F>
void run(const char* name, F launch, double flops) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    launch(); cudaDeviceSynchronize();
    cudaEventRecord(s); for (int r = 0; r < 3; ++r) launch(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    printf("%-34s %8.2f T(FL)OPS\n", name, flops * 3 / (ms * 1e-3) / 1e12);
}
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int blocks = p.multiProcessorCount * 8, threads = 256, warps = 8;
    float* out; cudaMalloc(&out, (size_t)blocks * threads * 4);
    double mma1 = (double)blocks * warps * IT * 16 * 16 * 16 * 2;
    run("HMMA fp16->fp32, 4 chains", [&] { k_mma<half, float, 4><<<blocks, threads>>>(out, 0); }, mma1 * 4);
    run("HMMA fp16->fp32, 8 chains", [&] { k_mma<half, float, 8><<<blocks, threads>>>(out, 0); }, mma1 * 8);
    run("HMMA fp16->fp16, 8 chains", [&] { k_mma<half, half, 8><<<blocks, threads>>>(out, 0); }, mma1 * 8);
    run("IMMA int8->int32, 8 chains", [&] { k_mma<signed char, int, 8><<<blocks, threads>>>(out, 0); }, mma1 * 8);
    double sc = (double)blocks * threads * IT * 4;
    run("HFMA2 fp16, 8 chains", [&] { k_hfma2<8><<<blocks, threads>>>(out, 1.0001f, 1e-4f); }, sc * 8 * 4);
    run("FFMA fp32, 8 chains", [&] { k_ffma<8><<<blocks, threads>>>(out, 1.0001f, 1e-4f); }, sc * 8 * 2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
}
