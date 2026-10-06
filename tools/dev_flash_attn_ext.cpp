// Standalone binding for developing src/cuda/fp16_flash_attention_hfma2.cu
// without rebuilding all of cmpext3 (tools/dev_flash_attn.py loads it).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

bool launch_flash_attention_hfma2(const void* q, const void* k, const void* v, void* o,
                                  int B, int H, int Sq, int Sk, int D,
                                  const int64_t* qs, const int64_t* ks, const int64_t* vs,
                                  const int64_t* os, float scale, cudaStream_t stream);

torch::Tensor attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale) {
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1, "last dim must be contiguous");
    // output as [B, Sq, H, D] memory, returned as a [B, H, Sq, D] view
    auto o = torch::empty({q.size(0), q.size(2), q.size(1), q.size(3)}, q.options()).transpose(1, 2);
    int64_t qs[3] = {q.stride(0), q.stride(1), q.stride(2)}, ks[3] = {k.stride(0), k.stride(1), k.stride(2)};
    int64_t vs[3] = {v.stride(0), v.stride(1), v.stride(2)}, os[3] = {o.stride(0), o.stride(1), o.stride(2)};
    bool ok = launch_flash_attention_hfma2(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), q.size(0), q.size(1), q.size(2), k.size(2), q.size(3),
        qs, ks, vs, os, (float)scale, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(ok, "unsupported head_dim or launch failure");
    return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn", &attn); }
