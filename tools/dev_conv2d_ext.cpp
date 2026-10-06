// Standalone binding for developing src/cuda/fp16_conv2d_hfma2.cu (tools/dev_conv2d.py).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
bool launch_conv2d_hfma2(const void* x, const void* w_khwc, const void* bias, void* y,
                         int N, int Ci, int H, int W, int Co, int KH, int KW,
                         int sh, int sw, int ph, int pw, int dh, int dw, int Ho, int Wo,
                         cudaStream_t stream);
torch::Tensor conv(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b, int s, int p) {
    x = x.contiguous();
    auto wk = w.permute({0, 2, 3, 1}).contiguous();
    int N = x.size(0), Ci = x.size(1), H = x.size(2), W = x.size(3), Co = w.size(0), KH = w.size(2), KW = w.size(3);
    int Ho = (H + 2 * p - KH) / s + 1, Wo = (W + 2 * p - KW) / s + 1;
    auto y = torch::empty({N, Co, Ho, Wo}, x.options());
    bool ok = launch_conv2d_hfma2(x.data_ptr(), wk.data_ptr(), b.has_value() ? b->data_ptr() : nullptr, y.data_ptr(),
                                  N, Ci, H, W, Co, KH, KW, s, s, p, p, 1, 1, Ho, Wo,
                                  c10::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(ok, "launch failed");
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("conv", &conv); }
