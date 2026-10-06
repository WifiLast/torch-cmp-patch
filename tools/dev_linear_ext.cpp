// Standalone binding for developing src/cuda/fp16_linear_hfma2.cu (tools/dev_linear.py).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
bool launch_linear_hfma2(const void* x, int64_t ldx, const void* w, const void* bias,
                         void* y, int64_t ldy, int N, int M, int K, cudaStream_t stream);
torch::Tensor lin(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b) {
    auto x2 = x.reshape({-1, x.size(-1)}).contiguous();
    auto wc = w.contiguous();
    int N = x2.size(0), K = x2.size(1), M = w.size(0);
    auto y = torch::empty({N, M}, x.options());
    TORCH_CHECK(launch_linear_hfma2(x2.data_ptr(), K, wc.data_ptr(), b.has_value() ? b->data_ptr() : nullptr,
                                    y.data_ptr(), M, N, M, K, c10::cuda::getCurrentCUDAStream().stream()));
    auto shape = x.sizes().vec(); shape.back() = M;
    return y.view(shape);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("lin", &lin); }
