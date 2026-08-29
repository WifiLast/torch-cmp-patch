#include <torch/extension.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Optional.h>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>

#ifdef CMPEXT3_WITH_TENSORRT
#include <c10/cuda/CUDAStream.h>
#include "trt/trt_common.h"
#include "trt/conv3d_trt_runtime.h"
#include "trt/conv2d_trt_runtime.h"
#include "trt/matmul_trt_runtime.h"
#include "trt/attention_trt_runtime.h"

// CMPEXT3_USE_TENSORRT=1 opt-in, checked once (not per-call) -- see
// src/trt/trt_common.h for the full design rationale. Every call site
// below still requires the compile-time CMPEXT3_WITH_TENSORRT guard (set
// by setup.py only when a TensorRT SDK was found at build time) on top of
// this runtime check, and every one of them falls straight through to the
// existing hand-tuned-kernel dispatch on any failure -- this must never be
// the only path to a result.
static bool cmpext3_trt_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("CMPEXT3_USE_TENSORRT");
        if (v == nullptr) return false;
        std::string s(v);
        return !s.empty() && s != "0";
    }();
    return enabled;
}
#endif

// ------------------------------------------------------------------
// 外部函数声明
// ------------------------------------------------------------------

// Matmul Launchers
void launch_matmul_fp16(const void* input, const void* weight, void* output, int M, int N, int K);
void launch_matmul_fp32(const float* input, const float* weight, float* output, int M, int N, int K);

// Bias Add Launchers
void launch_add_bias_fp16(void* output, const void* bias, int rows, int cols);
void launch_add_bias_fp32(float* output, const float* bias, int rows, int cols);

// Conv2d Launchers
// 参数说明: input, weight, bias, output, 
// batch, in_channels, in_h, in_w, out_channels, k_h, k_w, 
// out_h, out_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
void launch_conv2d_fp32(const float* input, const float* weight, const float* bias, float* output,int B, int C_in, int H_in, int W_in, int C_out, int K_H, int K_W,int H_out, int W_out,int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w);
void launch_conv2d_fp16(const void* input, const void* weight, const void* bias, void* output,int B, int C_in, int H_in, int W_in, int C_out, int K_H, int K_W,int H_out, int W_out,int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w);

// Conv3d Launchers (see src/cuda/fp32_conv3d.cu / fp16_conv3d.cu)
// 参数说明: input, weight, bias, output,
// batch, in_channels, in_d, in_h, in_w, out_channels, k_d, k_h, k_w,
// out_d, out_h, out_w, stride_d/h/w, pad_d/h/w, dilation_d/h/w
void launch_conv3d_fp32(const float* input, const float* weight, const float* bias, float* output,int B, int C_in, int D_in, int H_in, int W_in, int C_out, int K_D, int K_H, int K_W,int D_out, int H_out, int W_out,int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w, int dil_d, int dil_h, int dil_w);
void launch_conv3d_fp16(const void* input, const void* weight, const void* bias, void* output,int B, int C_in, int D_in, int H_in, int W_in, int C_out, int K_D, int K_H, int K_W,int D_out, int H_out, int W_out,int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w, int dil_d, int dil_h, int dil_w);

// Conv3d Winograd launcher (see src/cuda/fp32_conv3d_winograd.cu).
// Same argument list as launch_conv3d_fp32 on purpose, so the two are
// interchangeable at the call site -- but this one returns a bool:
// false means "this shape is outside my scope, nothing was written,
// use the direct kernel instead" (its own guard block is the single
// source of truth for that scope; main.cpp deliberately doesn't
// re-check it, so the two can never drift apart).
extern "C" bool launch_conv3d_fp32_winograd(const float* input, const float* weight, const float* bias, float* output,int B, int C_in, int D_in, int H_in, int W_in, int C_out, int K_D, int K_H, int K_W,int D_out, int H_out, int W_out,int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w, int dil_d, int dil_h, int dil_w);

// ConvTranspose2d Launchers
// 参数说明: input, weight, bias, output
// B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
// stride, pad, output_padding, dilation
void launch_conv_transpose2d_fp32(const float* input, const float* weight, const float* bias, float* output,int B, int C_in, int H_in, int W_in, int C_out, int K_H, int K_W, int H_out, int W_out,int stride_h, int stride_w, int pad_h, int pad_w, int out_pad_h, int out_pad_w, int dil_h, int dil_w);
void launch_conv_transpose2d_fp16(const void* input, const void* weight, const void* bias, void* output,int B, int C_in, int H_in, int W_in, int C_out, int K_H, int K_W, int H_out, int W_out,int stride_h, int stride_w, int pad_h, int pad_w, int out_pad_h, int out_pad_w, int dil_h, int dil_w);
// UpSample
// 参数: input, output, B, C, H_in, W_in, H_out, W_out
void launch_upsample_nearest_fp32(const float* input, float* output, int B, int C, int H_in, int W_in, int H_out, int W_out);
void launch_upsample_nearest_fp16(const void* input, void* output, int B, int C, int H_in, int W_in, int H_out, int W_out);

// Attention Launchers
// 输入: Q, K, V (B, H, S, D), Output (B, H, S, D)
// 参数: B, H, S, D, scale
void launch_attention_fp32(const float* q, const float* k, const float* v, float* output, int B, int H, int S, int D, float scale);
void launch_attention_fp16(const void* q, const void* k, const void* v, void* output, int B, int H, int S, int D, float scale);

//GroupNorm
// 参数: output, input, weight(gamma), bias(beta), N, C, HxW, groups, eps
void launch_groupnorm_fp32(float* output, const float* input, const float* weight, const float* bias, int N, int C, int HxW, int groups, float eps, bool apply_silu);
void launch_groupnorm_fp16(void* output, const void* input, const void* weight, const void* bias, int N, int C, int HxW, int groups, float eps, bool apply_silu);

// LayerNorm 
// 参数: output, input, weight(gamma), bias(beta), rows, cols, eps
// 注意：为了简单，这里不输出 mean 和 rstd (仅做推理用)
void launch_layernorm_fp32(float* output, const float* input, const float* gamma, const float* beta, int rows, int cols, float eps);
void launch_layernorm_fp16(void* output, const void* input, const void* gamma, const void* beta, int rows, int cols, float eps);

// RMSNorm Launchers
// 参数: output, input, weight(gamma), rows, cols, eps
void launch_rmsnorm_fp32(float* output, const float* input, const float* weight, int rows, int cols, float eps);
void launch_rmsnorm_fp16(void* output, const void* input, const void* weight, int rows, int cols, float eps);

// GELU Launchers
void launch_gelu_fp32(const float* input, float* output, int total_elements);
void launch_gelu_fp16(const void* input, void* output, int total_elements);

// Silu Launchers
void launch_silu_fp32(const float* input, float* output, int total_elements);
void launch_silu_fp16(const void* input, void* output, int total_elements);

// Swish Launchers (With Beta parameter)
// beta 是指向 GPU 显存中单标量参数的指针
void launch_swish_fp32(const float* input, const float* beta, float* output, int total_elements);
void launch_swish_fp16(const void* input, const void* beta, void* output, int total_elements);

// Mish Launchers
void launch_mish_fp32(const float* input, float* output, int total_elements);
void launch_mish_fp16(const void* input, void* output, int total_elements);

// Softmax Launchers
// 参数: input, output, rows, cols
void launch_softmax_fp32(const float* input, float* output, int rows, int cols);
void launch_softmax_fp16(const void* input, void* output, int rows, int cols);

// Softplus Launchers
// Formula: 1/beta * log(1 + exp(beta * x))
// 这里的 beta 和 threshold 均为标量
void launch_softplus_fp32(const float* input, float* output, int total_elements, float beta, float threshold);
void launch_softplus_fp16(const void* input, void* output, int total_elements, float beta, float threshold);

// Softsign Launchers
// Formula: x / (1 + |x|)
void launch_softsign_fp32(const float* input, float* output, int total_elements);
void launch_softsign_fp16(const void* input, void* output, int total_elements);

// Softshrink Launchers
// Formula: x - lambda if x > lambda; x + lambda if x < -lambda; 0 otherwise
void launch_softshrink_fp32(const float* input, float* output, int total_elements, float lambd);
void launch_softshrink_fp16(const void* input, void* output, int total_elements, float lambd);

// Embedding Launchers
// 参数: indices(long*), weight, output, num_indices, embedding_dim, padding_idx, num_weights(rows)
void launch_embedding_fp32(const int64_t* indices, const float* weight, float* output, int num_indices, int embedding_dim, int padding_idx, int num_embeddings);
void launch_embedding_fp16(const int64_t* indices, const void* weight, void* output, int num_indices, int embedding_dim, int padding_idx, int num_embeddings);

// Tanh Launchers
void launch_tanh_fp32(const float* input, float* output, int total_elements);
void launch_tanh_fp16(const void* input, void* output, int total_elements);

// ERF Launchers
void launch_erf_fp32(const float* input, float* output, int total_elements);
void launch_erf_fp16(const void* input, void* output, int total_elements);

// ------------------------------------------------------------------
// 辅助函数：将 int 或 list 统一转为 vector
// ------------------------------------------------------------------
std::vector<int64_t> expand_param_if_needed(py::object param) {
    // 如果是整数，扩展为 {val, val} (假设是2D)
    if (py::isinstance<py::int_>(param)) {
        int64_t val = param.cast<int64_t>();
        return {val, val};
    }
    // 否则尝试转为 vector (如果是 tuple 或 list)
    return param.cast<std::vector<int64_t>>();
}


// ------------------------------------------------------------------
// Linear 实现
// ------------------------------------------------------------------

torch::Tensor custom_linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias) {

    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        torch::Tensor input_f32 = input.to(torch::kFloat32);
        torch::Tensor weight_f32 = weight.to(torch::kFloat32);
        c10::optional<torch::Tensor> bias_f32 = c10::nullopt;
        if (bias.has_value() && bias->defined()) {
            bias_f32 = bias->to(torch::kFloat32);
        }
        torch::Tensor out = custom_linear_forward(input_f32, weight_f32, bias_f32);
        return out.to(torch::kBFloat16);
    }

    // 1. 基础检查
    TORCH_CHECK(input.dim() >= 2, "Input dim >= 2");
    TORCH_CHECK(weight.dim() == 2, "Weight dim == 2");
    TORCH_CHECK(input.device() == weight.device(), "Device mismatch");
    
    // 2. 维度获取 (PyTorch 布局: Weight is [N, K])
    int N = weight.size(0); // Out Features
    int K = weight.size(1); // In Features
    
    // 检查 Input 的 K 是否匹配
    TORCH_CHECK(input.size(-1) == K, "Input shape mismatch with Weight");

    long long total_elements = input.numel();
    int M = total_elements / K;

    // -------------------------------------------------------------------------
    // 关键修正 1: 权重转置
    // Kernel 期望 B 是 [K, N] 且行主序 (Row Major)。
    // PyTorch 的 weight 是 [N, K]。
    // 我们执行 weight.t() 得到 [K, N]，并调用 contiguous() 重新排列物理内存。
    // 这样 Kernel 就能线性读取到正确的列向量了。
    // -------------------------------------------------------------------------
    torch::Tensor weight_t = weight.t().contiguous(); 

    // 确保 Input 也是连续的
    input = input.contiguous();

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    // 这里的 Output 形状是 [M, N]
    torch::Tensor output = torch::empty({M, N}, options);

    // -------------------------------------------------------------------------
    // 关键修正 2: 鲁棒性检查
    // 如果 N 是奇数，目前的 Kernel 会崩溃或出错。
    // 这里做一个回退（Fallback）或者报错。为了简单，我们先报错。
    // -------------------------------------------------------------------------
    if (N % 2 != 0 && input.dtype() == torch::kFloat16) {
        // 如果你需要支持奇数 N，需要修改 CUDA Kernel 不使用 half2，或者在这里回退到 PyTorch 原生实现
        // 这里暂时抛出异常提示
         TORCH_CHECK(false, "Custom FP16 Kernel requires even Output Channels (N)");
    }

    // 3. 执行矩阵乘法
#ifdef CMPEXT3_WITH_TENSORRT
    bool matmul_done_by_trt = false;
    if (cmpext3_trt_enabled() &&
        (input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32)) {
        bool is_fp16 = input.dtype() == torch::kFloat16;
        const void* input_ptr = is_fp16 ? (const void*)input.data_ptr<at::Half>() : (const void*)input.data_ptr<float>();
        const void* weight_ptr = is_fp16 ? (const void*)weight_t.data_ptr<at::Half>() : (const void*)weight_t.data_ptr<float>();
        void* output_ptr = is_fp16 ? (void*)output.data_ptr<at::Half>() : (void*)output.data_ptr<float>();
        matmul_done_by_trt = cmpext3_trt_matmul_forward(
            input_ptr, weight_ptr, output_ptr, M, N, K, is_fp16,
            at::cuda::getCurrentCUDAStream(input.device().index()).stream());
    }
    if (!matmul_done_by_trt) {
#endif
    if (input.dtype() == torch::kFloat16) {
        launch_matmul_fp16(
            input.data_ptr<at::Half>(),
            weight_t.data_ptr<at::Half>(), // 传入转置后的权重指针
            output.data_ptr<at::Half>(),
            M, N, K
        );
    } else if (input.dtype() == torch::kFloat32) {
        // FP32 通常不需要向量化对齐，但也需要转置权重
        launch_matmul_fp32(
            input.data_ptr<float>(),
            weight_t.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
#ifdef CMPEXT3_WITH_TENSORRT
    }
#endif

    // 4. 执行 Bias 加法
    if (bias.has_value() && bias->defined()) {
        torch::Tensor bias_t = *bias;
        TORCH_CHECK(bias_t.device() == input.device(), "Bias device mismatch");
        TORCH_CHECK(bias_t.dtype() == input.dtype(), "Bias dtype mismatch");
        TORCH_CHECK(bias_t.dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias_t.size(0) == N, "Bias size mismatch");
        
        bias_t = bias_t.contiguous();

        if (input.dtype() == torch::kFloat16) {
            launch_add_bias_fp16(output.data_ptr<at::Half>(), bias_t.data_ptr<at::Half>(), M, N);
        } else if (input.dtype() == torch::kFloat32) {
            launch_add_bias_fp32(output.data_ptr<float>(), bias_t.data_ptr<float>(), M, N);
        }
    }

    // 5. 恢复形状 (Output 应该是 Input 的形状，只是最后一个维度变成 N)
    std::vector<int64_t> output_shape;
    for (int i = 0; i < input.dim() - 1; ++i) {
        output_shape.push_back(input.size(i));
    }
    output_shape.push_back(N);
    
    return output.view(output_shape);
}

// Wrapper for PyBind
torch::Tensor custom_linear_wrapper(
    torch::Tensor input, 
    torch::Tensor weight, 
    c10::optional<torch::Tensor> bias) {
    return custom_linear_forward(input, weight, bias);
}


// ------------------------------------------------------------------
// BMM 实现 (Batch Matrix Multiplication)
// 基于现有 matmul 接口，在 C++ 端循环 Batch
// Input: (B, M, K), Mat2: (B, K, N) -> Output: (B, M, N)
// ------------------------------------------------------------------

torch::Tensor custom_bmm_forward(torch::Tensor input, torch::Tensor mat2) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        torch::Tensor out = custom_bmm_forward(input.to(torch::kFloat32), mat2.to(torch::kFloat32));
        return out.to(torch::kBFloat16);
    }

    // 1. 基础检查
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (B, M, K)");
    TORCH_CHECK(mat2.dim() == 3, "Mat2 must be 3D (B, K, N)");
    TORCH_CHECK(input.device() == mat2.device(), "Device mismatch");
    TORCH_CHECK(input.dtype() == mat2.dtype(), "Dtype mismatch");

    int B = input.size(0);
    int M = input.size(1);
    int K = input.size(2);
    
    int B_mat2 = mat2.size(0);
    int K_mat2 = mat2.size(1);
    int N = mat2.size(2);

    TORCH_CHECK(B == B_mat2, "Batch size mismatch");
    TORCH_CHECK(K == K_mat2, "Contracting dimension (K) mismatch");

    // 2. 内存连续化
    // 必须确保内存连续，以便通过指针偏移访问每个 Batch 的数据
    input = input.contiguous();
    mat2 = mat2.contiguous();

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({B, M, N}, options);

    // 3. 维度限制检查 (沿用 Linear 的逻辑)
    if (N % 2 != 0 && input.dtype() == torch::kFloat16) {
         TORCH_CHECK(false, "Custom FP16 Kernel requires even Output dim (N)");
    }

    // 4. 计算每个 Batch 的元素偏移量 (Stride)
    long long input_step = M * K;
    long long mat2_step = K * N;
    long long output_step = M * N;

    // 5. 循环执行 Matmul
    // 注意：Linear 中对 weight 做了转置，是因为 Linear 权重通常是 (N, K)。
    // 但 BMM 中 mat2 输入通常就是 (B, K, N)，符合 (M,K)x(K,N) 的顺序，因此不需要转置。
    
    if (input.dtype() == torch::kFloat32) {
        float* input_ptr = input.data_ptr<float>();
        float* mat2_ptr = mat2.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        for (int b = 0; b < B; ++b) {
            launch_matmul_fp32(
                input_ptr + b * input_step,
                mat2_ptr + b * mat2_step,
                output_ptr + b * output_step,
                M, N, K
            );
        }
    } else if (input.dtype() == torch::kFloat16) {
        at::Half* input_ptr = input.data_ptr<at::Half>();
        at::Half* mat2_ptr = mat2.data_ptr<at::Half>();
        at::Half* output_ptr = output.data_ptr<at::Half>();

        for (int b = 0; b < B; ++b) {
            launch_matmul_fp16(
                input_ptr + b * input_step,
                mat2_ptr + b * mat2_step,
                output_ptr + b * output_step,
                M, N, K
            );
        }
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom bmm");
    }

    return output;
}

// ------------------------------------------------------------------
// Conv2d 实现 
// ------------------------------------------------------------------

torch::Tensor custom_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        torch::Tensor input_f32 = input.to(torch::kFloat32);
        torch::Tensor weight_f32 = weight.to(torch::kFloat32);
        c10::optional<torch::Tensor> bias_f32 = c10::nullopt;
        if (bias.has_value() && bias->defined()) {
            bias_f32 = bias->to(torch::kFloat32);
        }
        torch::Tensor out = custom_conv2d_forward(input_f32, weight_f32, bias_f32, stride, padding, dilation, groups);
        return out.to(torch::kBFloat16);
    }

    // 1. 基础检查
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (O I/G kH kW)");
    TORCH_CHECK(input.device() == weight.device(), "Device mismatch");
    TORCH_CHECK(groups == 1, "Currently only groups=1 is supported in this custom op");

    // 2. 获取维度
    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int C_out = weight.size(0);
    int K_H = weight.size(2);
    int K_W = weight.size(3);

    TORCH_CHECK(input.size(1) == weight.size(1), "Input channels mismatch weight channels");

    // 处理参数
    int s_h = stride[0];
    int s_w = stride.size() > 1 ? stride[1] : stride[0];
    int p_h = padding[0];
    int p_w = padding.size() > 1 ? padding[1] : padding[0];
    int d_h = dilation[0];
    int d_w = dilation.size() > 1 ? dilation[1] : dilation[0];

    // =========================================================================
    // 优化分支: 1x1 Convolution 优化
    // 条件: Kernel=1x1, Stride=1, Padding=0, Dilation=1
    // =========================================================================
    if (K_H == 1 && K_W == 1 && s_h == 1 && s_w == 1 && 
        p_h == 0 && p_w == 0 && d_h == 1 && d_w == 1) {
        
        // 1. 变换 Input: NCHW -> NHWC
        // memory layout: [B, C, H, W] -> [B, H, W, C]
        // 注意：permute 不会拷贝内存，但 linear 内部的 .contiguous() 会触发拷贝
        torch::Tensor input_nhwc = input.permute({0, 2, 3, 1}); 

        // 2. 变换 Weight: [C_out, C_in, 1, 1] -> [C_out, C_in]
        torch::Tensor weight_flat = weight.view({C_out, C_in});

        // 3. 调用 Linear
        // Input: [B, H, W, C_in]
        // Weight: [C_out, C_in]
        // Output: [B, H, W, C_out]
        // 注意：linear 内部会自动处理 input.contiguous() 和 weight转置
        torch::Tensor output_nhwc = custom_linear_forward(input_nhwc, weight_flat, bias);

        // 4. 变换 Output: NHWC -> NCHW
        // [B, H, W, C_out] -> [B, C_out, H, W]
        // 最后调用 contiguous 确保内存布局符合后续 Conv 操作的标准
        return output_nhwc.permute({0, 3, 1, 2}).contiguous();
    }

    // =========================================================================
    // 常规 Conv2d 路径
    // =========================================================================

    // 3. 计算输出尺寸
    int H_out = (H_in + 2 * p_h - d_h * (K_H - 1) - 1) / s_h + 1;
    int W_out = (W_in + 2 * p_w - d_w * (K_W - 1) - 1) / s_w + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "Calculated output size is too small");

    // 4. 准备内存
    input = input.contiguous();
    weight = weight.contiguous();
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({B, C_out, H_out, W_out}, options);

    // 处理 Bias
    const void* bias_ptr = nullptr;
    const float* bias_ptr_fp32 = nullptr;

    if (bias.has_value() && bias->defined()) {
        torch::Tensor bias_t = *bias;
        TORCH_CHECK(bias_t.dim() == 1 && bias_t.size(0) == C_out, "Bias shape mismatch");
        bias_t = bias_t.contiguous();
        if (input.dtype() == torch::kFloat32) bias_ptr_fp32 = bias_t.data_ptr<float>();
        else bias_ptr = bias_t.data_ptr();
    }

    // 5. 调度 Kernel
#ifdef CMPEXT3_WITH_TENSORRT
    bool conv2d_done_by_trt = false;
    if (cmpext3_trt_enabled() &&
        (input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32)) {
        bool is_fp16 = input.dtype() == torch::kFloat16;
        const void* input_ptr = is_fp16 ? (const void*)input.data_ptr<at::Half>() : (const void*)input.data_ptr<float>();
        const void* weight_ptr = is_fp16 ? (const void*)weight.data_ptr<at::Half>() : (const void*)weight.data_ptr<float>();
        const void* bias_ptr_trt = is_fp16 ? bias_ptr : (const void*)bias_ptr_fp32;
        conv2d_done_by_trt = cmpext3_trt_conv2d_forward(
            input_ptr, weight_ptr, bias_ptr_trt, output.data_ptr(),
            B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
            s_h, s_w, p_h, p_w, d_h, d_w,
            is_fp16, bias_ptr_trt != nullptr,
            at::cuda::getCurrentCUDAStream(input.device().index()).stream());
    }
    if (!conv2d_done_by_trt) {
#endif
    if (input.dtype() == torch::kFloat32) {
        launch_conv2d_fp32(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr_fp32,
            output.data_ptr<float>(),
            B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
            s_h, s_w, p_h, p_w, d_h, d_w
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_conv2d_fp16(
            input.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            bias_ptr,
            output.data_ptr<at::Half>(),
            B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
            s_h, s_w, p_h, p_w, d_h, d_w
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom_conv2d");
    }
#ifdef CMPEXT3_WITH_TENSORRT
    }
#endif

    return output;
}

// Wrapper for PyBind
torch::Tensor custom_conv2d_wrapper(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride_arg,   // 改为 py::object
    py::object padding_arg,  // 改为 py::object
    py::object dilation_arg, // 改为 py::object
    int64_t groups) {
    
    // 解析参数
    std::vector<int64_t> stride = expand_param_if_needed(stride_arg);
    std::vector<int64_t> padding = expand_param_if_needed(padding_arg);
    std::vector<int64_t> dilation = expand_param_if_needed(dilation_arg);

    return custom_conv2d_forward(input, weight, bias, stride, padding, dilation, groups);
}


// ------------------------------------------------------------------
// "Two kernels, same result -- measure which one actually wins" helper.
//
// Same contract as trt_faster_than_fallback (src/trt/trt_common.h):
// `alt_fn` returns false when it can't handle this call at all (and must
// then have written nothing), `base_fn` always succeeds, and EVERY call
// -- benchmark run, cache hit, or cache miss -- leaves a complete correct
// result behind, so callers never need to re-run either one afterward.
//
// Deliberately NOT reusing that function: src/trt/*.cpp is only compiled
// when a TensorRT SDK was found at build time (see setup.py), which is
// the uncommon case, and this decision has nothing to do with TensorRT.
// Hoisting trt_common.cpp's version out of the optional build would drag
// NvInfer.h into the mandatory one; ~40 lines duplicated is the cheaper
// trade. The timing loop below mirrors trt_common.cpp's time_calls()
// exactly, including dropping the first timed sample, so the two paths'
// numbers stay directly comparable when both are active.
//
// Timed on the legacy default stream, which is where every launch_*()
// in this file actually runs its kernels (none of them take a stream).
// ------------------------------------------------------------------
// CMPEXT3_VERBOSE=1 prints each kernel choice as it's decided. Off by
// default because the Python layer (cmpext3/autoselect.py) makes the same
// kind of choice for every patched op, not just conv3d, and reads the same
// variable -- together they'd be dozens of lines of startup noise. Both
// layers stay queryable when quiet: cmpext3.autoselect_report() for the
// Python side, and this line for the native one by setting the variable.
static bool cmpext3_verbose() {
    static const bool verbose = []() {
        const char* v = std::getenv("CMPEXT3_VERBOSE");
        if (v == nullptr) return false;
        std::string s(v);
        return !s.empty() && s != "0";
    }();
    return verbose;
}

static std::mutex g_alt_kernel_bench_mutex;
static std::unordered_map<std::string, bool> g_alt_kernel_decisions;

static float cmpext3_time_kernel(const std::function<void()>& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;
    for (int i = 0; i < iters + 1; ++i) {
        cudaEventRecord(start);
        fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float call_ms = 0.0f;
        cudaEventElapsedTime(&call_ms, start, stop);
        if (i > 0) total_ms += call_ms;  // i == 0: the excluded first timed run
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / static_cast<float>(iters);
}

static bool cmpext3_alt_kernel_faster(
    const std::string& key,
    const std::function<bool()>& alt_fn,
    const std::function<void()>& base_fn)
{
    {
        std::lock_guard<std::mutex> lock(g_alt_kernel_bench_mutex);
        auto it = g_alt_kernel_decisions.find(key);
        if (it != g_alt_kernel_decisions.end()) {
            if (it->second && alt_fn()) return true;
            // Either the cached decision was "use the base kernel", or the
            // alternative won earlier but failed THIS call (e.g. a transient
            // workspace cudaMalloc failure) -- fall back for this call only,
            // decision cache untouched so the next call tries it again.
            base_fn();
            return false;
        }
    }

    // First call for this shape: one untimed probe. This is also how an
    // out-of-scope shape gets resolved -- alt_fn()'s guard returns false
    // before allocating anything, and the "use the base kernel" answer is
    // cached, so every later call with that shape costs one map lookup.
    if (!alt_fn()) {
        {
            std::lock_guard<std::mutex> lock(g_alt_kernel_bench_mutex);
            g_alt_kernel_decisions[key] = false;
        }
        base_fn();
        return false;
    }

    constexpr int kWarmup = 2;
    constexpr int kIters = 5;
    float alt_ms = cmpext3_time_kernel([&]() { alt_fn(); }, kWarmup, kIters);
    float base_ms = cmpext3_time_kernel(base_fn, kWarmup, kIters);
    bool use_alt = alt_ms < base_ms;

    if (cmpext3_verbose()) {
        fprintf(stderr, "[cmpext3] %s: %.3f ms vs. %.3f ms for the default kernel -- using %s\n",
                key.c_str(), alt_ms, base_ms, use_alt ? "it" : "the default kernel");
    }

    // Leave a fresh, correct result behind matching the decision. base_fn
    // was timed AFTER alt_fn, so `output` already holds a correct base
    // result either way -- one more call is needed only if alt won.
    if (use_alt) {
        alt_fn();
    }

    {
        std::lock_guard<std::mutex> lock(g_alt_kernel_bench_mutex);
        g_alt_kernel_decisions[key] = use_alt;
    }
    return use_alt;
}

// CMPEXT3_DISABLE_WINOGRAD=1 pins conv3d fp32 to the direct kernel --
// an escape hatch for bisecting a suspected Winograd bug, or for forcing
// bit-identical output against a build that predates it.
//
// Explicitly NOT an accuracy fallback: measured against an fp64 reference
// on this GPU (32/64/128/256 channels at 8x16x16 and 8x32x32), Winograd's
// max abs error is 2-4x LOWER than the direct kernel's at every one of
// those shapes -- 1.5e-4 vs. 7.1e-4 at 128 channels, and better than stock
// cuDNN's 3.1e-4 there too. The intuition that Winograd must be the noisier
// of the two (its transforms do introduce cancellation direct correlation
// doesn't have) is the wrong way round here, and it's worth knowing why
// before "improving" either kernel: the direct kernel drives 27*C_in
// products through ONE fp32 accumulator, while Winograd keeps 64 separate
// accumulators of C_in terms each, so it does ~27x fewer sequential
// additions per accumulator and rounds off correspondingly less.
//
// There is no matching "force Winograd on" knob: the benchmark below
// already refuses to use it where it doesn't win.
static bool cmpext3_winograd_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("CMPEXT3_DISABLE_WINOGRAD");
        if (v == nullptr) return true;
        std::string s(v);
        return s.empty() || s == "0";
    }();
    return enabled;
}


// ------------------------------------------------------------------
// Conv3d 实现
// (video patch-embed / 3D causal VAE coverage -- see custom_conv2d_forward
//  above for the 2D version this mirrors; groups==1 only, same as conv2d)
// ------------------------------------------------------------------

torch::Tensor custom_conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        torch::Tensor input_f32 = input.to(torch::kFloat32);
        torch::Tensor weight_f32 = weight.to(torch::kFloat32);
        c10::optional<torch::Tensor> bias_f32 = c10::nullopt;
        if (bias.has_value() && bias->defined()) {
            bias_f32 = bias->to(torch::kFloat32);
        }
        torch::Tensor out = custom_conv3d_forward(input_f32, weight_f32, bias_f32, stride, padding, dilation, groups);
        return out.to(torch::kBFloat16);
    }

    // 1. 基础检查
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (NCDHW)");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D (O I/G kD kH kW)");
    TORCH_CHECK(input.device() == weight.device(), "Device mismatch");
    TORCH_CHECK(groups == 1, "Currently only groups=1 is supported in this custom op");

    // 2. 获取维度
    int B = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int C_out = weight.size(0);
    int K_D = weight.size(2);
    int K_H = weight.size(3);
    int K_W = weight.size(4);

    TORCH_CHECK(input.size(1) == weight.size(1), "Input channels mismatch weight channels");

    // 处理参数 (3 元素: D, H, W; 单个 int 时对三维同时生效)
    int s_d = stride[0];
    int s_h = stride.size() > 1 ? stride[1] : stride[0];
    int s_w = stride.size() > 2 ? stride[2] : stride[0];
    int p_d = padding[0];
    int p_h = padding.size() > 1 ? padding[1] : padding[0];
    int p_w = padding.size() > 2 ? padding[2] : padding[0];
    int d_d = dilation[0];
    int d_h = dilation.size() > 1 ? dilation[1] : dilation[0];
    int d_w = dilation.size() > 2 ? dilation[2] : dilation[0];

    // =========================================================================
    // 优化分支: 1x1x1 Convolution 优化 (pure channel mixing, no spatial window
    // at all). Same trick as custom_conv2d_forward's 1x1 branch above: reshape
    // to NDHWC and dispatch to the tiled linear/matmul kernel instead of the
    // conv3d windowing machinery, which has nothing to do for a 1x1x1 kernel.
    // 条件: Kernel=1x1x1, Stride=1, Padding=0, Dilation=1
    // =========================================================================
    if (K_D == 1 && K_H == 1 && K_W == 1 && s_d == 1 && s_h == 1 && s_w == 1 &&
        p_d == 0 && p_h == 0 && p_w == 0 && d_d == 1 && d_h == 1 && d_w == 1) {

        // 1. 变换 Input: NCDHW -> NDHWC
        torch::Tensor input_ndhwc = input.permute({0, 2, 3, 4, 1});

        // 2. 变换 Weight: [C_out, C_in, 1, 1, 1] -> [C_out, C_in]
        torch::Tensor weight_flat = weight.view({C_out, C_in});

        // 3. 调用 Linear (处理 input.contiguous() 和 weight 转置)
        torch::Tensor output_ndhwc = custom_linear_forward(input_ndhwc, weight_flat, bias);

        // 4. 变换 Output: NDHWC -> NCDHW
        return output_ndhwc.permute({0, 4, 1, 2, 3}).contiguous();
    }

    // 3. 计算输出尺寸
    int D_out = (D_in + 2 * p_d - d_d * (K_D - 1) - 1) / s_d + 1;
    int H_out = (H_in + 2 * p_h - d_h * (K_H - 1) - 1) / s_h + 1;
    int W_out = (W_in + 2 * p_w - d_w * (K_W - 1) - 1) / s_w + 1;

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Calculated output size is too small");

    // 4. 准备内存
    input = input.contiguous();
    weight = weight.contiguous();

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({B, C_out, D_out, H_out, W_out}, options);

    // 处理 Bias
    const void* bias_ptr = nullptr;
    const float* bias_ptr_fp32 = nullptr;

    if (bias.has_value() && bias->defined()) {
        torch::Tensor bias_t = *bias;
        TORCH_CHECK(bias_t.dim() == 1 && bias_t.size(0) == C_out, "Bias shape mismatch");
        bias_t = bias_t.contiguous();
        if (input.dtype() == torch::kFloat32) bias_ptr_fp32 = bias_t.data_ptr<float>();
        else bias_ptr = bias_t.data_ptr();
    }

    // 5. 调度 Kernel -- pulled into a lambda so it can double as both the
    // plain dispatch path AND the "fallback" arg to the TensorRT
    // benchmark below without duplicating the if/else.
    auto run_kernel = [&]() {
        if (input.dtype() == torch::kFloat32) {
            auto run_direct = [&]() {
                launch_conv3d_fp32(
                    input.data_ptr<float>(),
                    weight.data_ptr<float>(),
                    bias_ptr_fp32,
                    output.data_ptr<float>(),
                    B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
                    s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w
                );
            };
            // F(2x2x2,3x3x3) Winograd (src/cuda/fp32_conv3d_winograd.cu):
            // ~5.6x fewer multiplies than the direct kernel, but it pays for
            // two extra transform passes plus a U/V workspace that can run to
            // hundreds of MB, and it only handles 3x3x3/stride1/pad1/
            // dilation1/B=1 with even spatial dims. Which of the two is
            // actually faster is therefore a per-shape question -- measured
            // once per shape by cmpext3_alt_kernel_faster and cached, never
            // assumed from applicability alone. Shapes it can't handle cost
            // one guard check on the first call and a map lookup after that.
            auto run_winograd = [&]() -> bool {
                if (!cmpext3_winograd_enabled()) return false;
                return launch_conv3d_fp32_winograd(
                    input.data_ptr<float>(),
                    weight.data_ptr<float>(),
                    bias_ptr_fp32,
                    output.data_ptr<float>(),
                    B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
                    s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w
                );
            };
            // Shape-only key (fp32 and groups==1 are already implied here).
            // Spelled out rather than hashed like the TensorRT path's key --
            // this one is short enough to stay readable in the one-line
            // benchmark report cmpext3_alt_kernel_faster prints to stderr.
            char wino_key[224];
            std::snprintf(wino_key, sizeof(wino_key),
                "conv3d fp32 Winograd [B%d C%d %dx%dx%d -> C%d k%dx%dx%d s%d,%d,%d p%d,%d,%d d%d,%d,%d bias%d]",
                B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W,
                s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w,
                bias_ptr_fp32 != nullptr ? 1 : 0);
            cmpext3_alt_kernel_faster(wino_key, run_winograd, run_direct);
        } else if (input.dtype() == torch::kFloat16) {
            launch_conv3d_fp16(
                input.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias_ptr,
                output.data_ptr<at::Half>(),
                B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
                s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype for custom_conv3d");
        }
    };

#ifdef CMPEXT3_WITH_TENSORRT
    bool conv3d_done_by_trt = false;
    if (cmpext3_trt_enabled() &&
        (input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32)) {
        bool is_fp16 = input.dtype() == torch::kFloat16;
        const void* input_ptr = is_fp16 ? (const void*)input.data_ptr<at::Half>() : (const void*)input.data_ptr<float>();
        const void* weight_ptr = is_fp16 ? (const void*)weight.data_ptr<at::Half>() : (const void*)weight.data_ptr<float>();
        const void* bias_ptr_trt = is_fp16 ? bias_ptr : (const void*)bias_ptr_fp32;
        bool has_bias_trt = bias_ptr_trt != nullptr;
        void* output_ptr = output.data_ptr();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index()).stream();

        // Shape+dtype-only cache key -- see trt_faster_than_fallback's doc
        // comment (trt_common.h) for why weight content is deliberately
        // excluded: unlike the engine cache inside conv3d_trt_runtime.cpp,
        // which one is FASTER doesn't depend on the actual weight values.
        struct BenchKey {
            int B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W;
            int s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w;
            bool is_fp16, has_bias;
        } bench_key;
        // Zero the WHOLE struct first, padding included -- aggregate
        // list-initialization only guarantees the named fields, leaving
        // padding bytes as uninitialized stack garbage that differs
        // between calls. That garbage feeds directly into the hash below,
        // which would otherwise make the cache key non-reproducible for
        // the identical shape (confirmed by testing: without this, the
        // same conv3d shape called twice in one process re-benchmarked
        // both times instead of hitting the cache on the second call).
        std::memset(&bench_key, 0, sizeof(bench_key));
        bench_key.B = B; bench_key.C_in = C_in; bench_key.D_in = D_in;
        bench_key.H_in = H_in; bench_key.W_in = W_in; bench_key.C_out = C_out;
        bench_key.K_D = K_D; bench_key.K_H = K_H; bench_key.K_W = K_W;
        bench_key.s_d = s_d; bench_key.s_h = s_h; bench_key.s_w = s_w;
        bench_key.p_d = p_d; bench_key.p_h = p_h; bench_key.p_w = p_w;
        bench_key.d_d = d_d; bench_key.d_h = d_h; bench_key.d_w = d_w;
        bench_key.is_fp16 = is_fp16; bench_key.has_bias = has_bias_trt;
        std::string bench_key_str = "conv3d_" + cmpext3_trt::hash_to_hex(
            cmpext3_trt::fnv1a_update(&bench_key, sizeof(bench_key), cmpext3_trt::FNV_OFFSET_BASIS));

        auto run_trt = [&]() {
            return cmpext3_trt_conv3d_forward(
                input_ptr, weight_ptr, bias_ptr_trt, output_ptr,
                B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
                s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w,
                is_fp16, has_bias_trt, stream);
        };

        // Measured once per unique shape (see trt_faster_than_fallback),
        // then cached: TensorRT is only actually used going forward if it
        // was measured faster than fp32_conv3d.cu/fp16_conv3d.cu for THIS
        // shape -- never assumed just because CMPEXT3_USE_TENSORRT=1.
        conv3d_done_by_trt = cmpext3_trt::trt_faster_than_fallback(
            bench_key_str, stream, run_trt, run_kernel);
    }
    if (!conv3d_done_by_trt) {
        run_kernel();
    }
#else
    run_kernel();
#endif

    return output;
}

// Wrapper for PyBind
torch::Tensor custom_conv3d_wrapper(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride_arg,
    py::object padding_arg,
    py::object dilation_arg,
    int64_t groups) {

    auto expand3 = [](py::object param) -> std::vector<int64_t> {
        if (py::isinstance<py::int_>(param)) {
            int64_t val = param.cast<int64_t>();
            return {val, val, val};
        }
        return param.cast<std::vector<int64_t>>();
    };

    std::vector<int64_t> stride = expand3(stride_arg);
    std::vector<int64_t> padding = expand3(padding_arg);
    std::vector<int64_t> dilation = expand3(dilation_arg);

    return custom_conv3d_forward(input, weight, bias, stride, padding, dilation, groups);
}


// ------------------------------------------------------------------
// ConvTranspose2d 实现
// ------------------------------------------------------------------

torch::Tensor custom_conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        torch::Tensor input_f32 = input.to(torch::kFloat32);
        torch::Tensor weight_f32 = weight.to(torch::kFloat32);
        c10::optional<torch::Tensor> bias_f32 = c10::nullopt;
        if (bias.has_value() && bias->defined()) {
            bias_f32 = bias->to(torch::kFloat32);
        }
        torch::Tensor out = custom_conv_transpose2d_forward(input_f32, weight_f32, bias_f32, stride, padding, output_padding, dilation, groups);
        return out.to(torch::kBFloat16);
    }

    // 1. 基础检查
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (NCHW)");
    // PyTorch ConvTranspose2d weight shape: [C_in, C_out/groups, K_H, K_W]
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (In, Out/G, kH, kW)");
    TORCH_CHECK(input.device() == weight.device(), "Device mismatch");
    TORCH_CHECK(groups == 1, "Currently only groups=1 is supported");

    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Weight: [C_in, C_out, K_H, K_W] (assuming groups=1)
    TORCH_CHECK(weight.size(0) == C_in, "Input channels mismatch weight input channels");
    int C_out = weight.size(1); 
    int K_H = weight.size(2);
    int K_W = weight.size(3);

    // 处理参数
    int s_h = stride[0];
    int s_w = stride.size() > 1 ? stride[1] : stride[0];
    int p_h = padding[0];
    int p_w = padding.size() > 1 ? padding[1] : padding[0];
    int op_h = output_padding[0];
    int op_w = output_padding.size() > 1 ? output_padding[1] : output_padding[0];
    int d_h = dilation[0];
    int d_w = dilation.size() > 1 ? dilation[1] : dilation[0];

    // 2. 计算输出尺寸
    // Formula: (H - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    int H_out = (H_in - 1) * s_h - 2 * p_h + d_h * (K_H - 1) + op_h + 1;
    int W_out = (W_in - 1) * s_w - 2 * p_w + d_w * (K_W - 1) + op_w + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "Calculated output size is invalid");

    // 3. 准备内存
    input = input.contiguous();
    weight = weight.contiguous();

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({B, C_out, H_out, W_out}, options);

    // 处理 Bias
    const void* bias_ptr = nullptr;
    const float* bias_ptr_fp32 = nullptr;

    if (bias.has_value() && bias->defined()) {
        torch::Tensor bias_t = *bias;
        TORCH_CHECK(bias_t.dim() == 1 && bias_t.size(0) == C_out, "Bias shape mismatch");
        bias_t = bias_t.contiguous();
        if (input.dtype() == torch::kFloat32) bias_ptr_fp32 = bias_t.data_ptr<float>();
        else bias_ptr = bias_t.data_ptr();
    }

    // 4. 调度 Kernel
    if (input.dtype() == torch::kFloat32) {
        launch_conv_transpose2d_fp32(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr_fp32, output.data_ptr<float>(),
            B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
            s_h, s_w, p_h, p_w, op_h, op_w, d_h, d_w
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_conv_transpose2d_fp16(
            input.data_ptr<at::Half>(), weight.data_ptr<at::Half>(), bias_ptr, output.data_ptr<at::Half>(),
            B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
            s_h, s_w, p_h, p_w, op_h, op_w, d_h, d_w
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for conv_transpose2d");
    }

    return output;
}

// Wrapper
torch::Tensor custom_conv_transpose2d_wrapper(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride_arg,
    py::object padding_arg,
    py::object output_padding_arg,
    py::object dilation_arg,
    int64_t groups) {
    
    std::vector<int64_t> stride = expand_param_if_needed(stride_arg);
    std::vector<int64_t> padding = expand_param_if_needed(padding_arg);
    std::vector<int64_t> output_padding = expand_param_if_needed(output_padding_arg);
    std::vector<int64_t> dilation = expand_param_if_needed(dilation_arg);

    return custom_conv_transpose2d_forward(input, weight, bias, stride, padding, output_padding, dilation, groups);
}

// ------------------------------------------------------------------
// 基于 ConvTranspose2d 的高效 Upsample 实现
// 原理：将 input (B, C, H, W) -> (B*C, 1, H, W)
//      使用 stride=scale, kernel_size=scale, weight=全1 的转置卷积
//      结果 (B*C, 1, H*scale, W*scale) -> (B, C, H*scale, W*scale)
// 健壮性增强：支持 scale_factor(int) 或 output_size(tuple/list)
// ------------------------------------------------------------------

torch::Tensor custom_upsample_smart(
    torch::Tensor input,
    py::object size_or_scale) { // 接收任意 Python 对象
    
    // 1. 基础检查
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t scale_h = 1;
    int64_t scale_w = 1;

    // 2. 参数解析 (Argument Parsing)
    if (py::isinstance<py::int_>(size_or_scale)) {
        // 情况 A: 传入的是整数 scale_factor
        int64_t s = size_or_scale.cast<int64_t>();
        scale_h = s;
        scale_w = s;
    } 
    else if (py::isinstance<py::tuple>(size_or_scale) || py::isinstance<py::list>(size_or_scale)) {
        // 情况 B: 传入的是元组/列表 (target_h, target_w)
        std::vector<int64_t> size = size_or_scale.cast<std::vector<int64_t>>();
        TORCH_CHECK(size.size() == 2, "Output size must be length 2 (H, W)");
        
        int64_t H_out = size[0];
        int64_t W_out = size[1];

        // 检查是否为整数倍
        TORCH_CHECK(H_out % H_in == 0, 
            "Target height ", H_out, " is not divisible by input height ", H_in, 
            ". This optimized op only supports integer scaling.");
        TORCH_CHECK(W_out % W_in == 0, 
            "Target width ", W_out, " is not divisible by input width ", W_in, 
            ". This optimized op only supports integer scaling.");

        scale_h = H_out / H_in;
        scale_w = W_out / W_in;
    } 
    else {
        TORCH_CHECK(false, "scale argument must be int or (h, w) tuple");
    }

    TORCH_CHECK(scale_h >= 1 && scale_w >= 1, "Scale factor must be >= 1");

    // 3. Reshape Trick (B, C, H, W) -> (B*C, 1, H, W)
    // 绕过 groups=1 的限制
    torch::Tensor input_reshaped = input.view({B * C, 1, H_in, W_in});

    // 4. 构造 Weight
    // Shape: [In, Out, K_H, K_W] -> [1, 1, scale_h, scale_w]
    // Nearest Neighbor = 全 1 权重
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor weight = torch::ones({1, 1, scale_h, scale_w}, options);

    // 5. 准备参数
    std::vector<int64_t> stride = {scale_h, scale_w};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> output_padding = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    int64_t groups = 1;

    // 6. 调用 ConvTranspose2d
    torch::Tensor output_reshaped = custom_conv_transpose2d_forward(
        input_reshaped,
        weight,
        c10::nullopt,
        stride,
        padding,
        output_padding,
        dilation,
        groups
    );

    // 7. 恢复形状
    // (B*C, 1, H_out, W_out) -> (B, C, H_out, W_out)
    return output_reshaped.view({B, C, H_in * scale_h, W_in * scale_w});
}

// ------------------------------------------------------------------
// Embedding 实现
// ------------------------------------------------------------------

torch::Tensor custom_embedding_forward(
    torch::Tensor input,      // indices [*, *]
    torch::Tensor weight,     // [num_embeddings, embedding_dim]
    int64_t padding_idx,
    bool scale_grad_by_freq,  // 暂时忽略，仅做接口兼容
    bool sparse               // 暂时忽略
) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (weight.scalar_type() == torch::kBFloat16) {
        torch::Tensor out = custom_embedding_forward(input, weight.to(torch::kFloat32), padding_idx, scale_grad_by_freq, sparse);
        return out.to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.scalar_type() == torch::kLong, "Indices must be Long (int64)");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(input.device() == weight.device(), "Device mismatch");

    // 2. 获取维度
    int num_embeddings = weight.size(0);
    int embedding_dim = weight.size(1);
    int num_indices = input.numel();

    // 3. 准备输出
    // 输出形状 = input.shape + [embedding_dim]
    std::vector<int64_t> output_shape = input.sizes().vec();
    output_shape.push_back(embedding_dim);

    auto options = torch::TensorOptions().dtype(weight.dtype()).device(weight.device());
    torch::Tensor output = torch::empty(output_shape, options);

    // 确保连续
    input = input.contiguous();
    weight = weight.contiguous();

    // 4. 调度
    if (weight.dtype() == torch::kFloat32) {
        launch_embedding_fp32(
            input.data_ptr<int64_t>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            num_indices, embedding_dim, (int)padding_idx, num_embeddings
        );
    } else if (weight.dtype() == torch::kFloat16) {
        launch_embedding_fp16(
            input.data_ptr<int64_t>(),
            weight.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            num_indices, embedding_dim, (int)padding_idx, num_embeddings
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for embedding");
    }

    return output;
}


// ------------------------------------------------------------------
// GroupNorm 实现
// ------------------------------------------------------------------

torch::Tensor custom_group_norm_forward(
    torch::Tensor input,
    int64_t num_groups,
    c10::optional<torch::Tensor> weight, // gamma
    c10::optional<torch::Tensor> bias,   // beta
    double eps,
    bool apply_silu = false) {

    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        c10::optional<torch::Tensor> weight_f32 = c10::nullopt;
        c10::optional<torch::Tensor> bias_f32 = c10::nullopt;
        if (weight.has_value() && weight->defined()) weight_f32 = weight->to(torch::kFloat32);
        if (bias.has_value() && bias->defined()) bias_f32 = bias->to(torch::kFloat32);
        torch::Tensor out = custom_group_norm_forward(input.to(torch::kFloat32), num_groups, weight_f32, bias_f32, eps, apply_silu);
        return out.to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() >= 3, "Input dim must be >= 3 (N, C, ...)");
    int N = input.size(0);
    int C = input.size(1);
    
    // 计算 HxW (Spatial dimensions flattened)
    int64_t HxW = 1;
    for (int i = 2; i < input.dim(); ++i) {
        HxW *= input.size(i);
    }

    TORCH_CHECK(C % num_groups == 0, "Channels must be divisible by groups");

    // 2. 处理 Gamma / Beta
    // 如果没有提供，我们需要处理空指针。在 Kernel 里如果不传指针，需要默认 Gamma=1, Beta=0。
    // 为了简单起见，如果 PyTorch 端没传，我们在 Python Wrapper 层最好保证传进来，或者在这里处理。
    // 这里我们允许 nullptr 传递给 kernel。
    
    const void* weight_ptr = nullptr;
    const void* bias_ptr = nullptr;
    const float* weight_ptr_fp32 = nullptr;
    const float* bias_ptr_fp32 = nullptr;

    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->size(0) == C, "Weight shape mismatch");
        TORCH_CHECK(weight->dtype() == input.dtype(), "Weight dtype mismatch");
        if (input.dtype() == torch::kFloat32) weight_ptr_fp32 = weight->data_ptr<float>();
        else weight_ptr = weight->data_ptr();
    }

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->size(0) == C, "Bias shape mismatch");
        TORCH_CHECK(bias->dtype() == input.dtype(), "Bias dtype mismatch");
        if (input.dtype() == torch::kFloat32) bias_ptr_fp32 = bias->data_ptr<float>();
        else bias_ptr = bias->data_ptr();
    }

    // 3. 准备输出
    auto output = torch::empty_like(input);

    // 4. Dispatch
    if (input.dtype() == torch::kFloat32) {
        launch_groupnorm_fp32(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            weight_ptr_fp32,
            bias_ptr_fp32,
            N, C, (int)HxW, (int)num_groups, (float)eps, apply_silu
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_groupnorm_fp16(
            output.data_ptr<at::Half>(),
            input.data_ptr<at::Half>(),
            weight_ptr,
            bias_ptr,
            N, C, (int)HxW, (int)num_groups, (float)eps, apply_silu
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for group_norm");
    }

    return output;
}

// Wrapper for PyBind -- fused GroupNorm+SiLU, same op with apply_silu=true.
// Not a monkeypatch target (F.group_norm has no activation argument to key
// off of), so this is exposed only as an explicit opt-in op: call
// cmpext3.ops.group_norm_silu(...) directly in model code in place of the
// separate F.group_norm(...) + F.silu(...) pair to skip the extra
// global-memory round trip between them -- the exact GroupNorm -> SiLU step
// of the GroupNorm -> SiLU -> Conv block a diffusion UNet/VAE decoder repeats.
torch::Tensor custom_group_norm_silu_forward(
    torch::Tensor input,
    int64_t num_groups,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    double eps) {
    return custom_group_norm_forward(input, num_groups, weight, bias, eps, /*apply_silu=*/true);
}

// ------------------------------------------------------------------
// LayerNorm 实现
// ------------------------------------------------------------------

torch::Tensor custom_layernorm_forward(
    torch::Tensor input,
    std::vector<int64_t> normalized_shape,
    c10::optional<torch::Tensor> weight, // Gamma
    c10::optional<torch::Tensor> bias,   // Beta
    double eps) {

    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        c10::optional<torch::Tensor> weight_f32 = c10::nullopt;
        c10::optional<torch::Tensor> bias_f32 = c10::nullopt;
        if (weight.has_value() && weight->defined()) weight_f32 = weight->to(torch::kFloat32);
        if (bias.has_value() && bias->defined()) bias_f32 = bias->to(torch::kFloat32);
        torch::Tensor out = custom_layernorm_forward(input.to(torch::kFloat32), normalized_shape, weight_f32, bias_f32, eps);
        return out.to(torch::kBFloat16);
    }

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    // 1. 计算维度
    int64_t feature_dim = 1;
    for (auto dim : normalized_shape) {
        feature_dim *= dim;
    }

    int input_rank = input.dim();
    int norm_rank = normalized_shape.size();
    TORCH_CHECK(input_rank >= norm_rank, "Input dimension must be >= normalized_shape dimension");
    
    for (int i = 0; i < norm_rank; ++i) {
        TORCH_CHECK(input.size(input_rank - norm_rank + i) == normalized_shape[i], 
            "Normalized shape mismatch");
    }

    int rows = input.numel() / feature_dim;
    int cols = feature_dim;

    // 2. 处理 Gamma / Beta
    const void* gamma_ptr = nullptr;
    const void* beta_ptr = nullptr;           // 声明为 beta_ptr
    const float* gamma_ptr_fp32 = nullptr;
    const float* beta_ptr_fp32 = nullptr;     // 声明为 beta_ptr_fp32

    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->numel() == cols, "Weight (gamma) size mismatch");
        TORCH_CHECK(weight->dtype() == input.dtype(), "Weight dtype mismatch");
        if (input.dtype() == torch::kFloat32) gamma_ptr_fp32 = weight->data_ptr<float>();
        else gamma_ptr = weight->data_ptr();
    }

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == cols, "Bias (beta) size mismatch");
        TORCH_CHECK(bias->dtype() == input.dtype(), "Bias dtype mismatch");
        
        // --- 修正点开始 ---
        // 之前错误写成了 bias_ptr_fp32，应改为 beta_ptr_fp32
        if (input.dtype() == torch::kFloat32) beta_ptr_fp32 = bias->data_ptr<float>();
        else beta_ptr = bias->data_ptr();
        // --- 修正点结束 ---
    }

    // 3. 准备输出
    auto output = torch::empty_like(input);

    // 4. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_layernorm_fp32(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            gamma_ptr_fp32,
            beta_ptr_fp32, // 使用 beta_ptr_fp32
            rows, cols, (float)eps
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_layernorm_fp16(
            output.data_ptr<at::Half>(),
            input.data_ptr<at::Half>(),
            gamma_ptr,
            beta_ptr,      // 使用 beta_ptr
            rows, cols, (float)eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for LayerNorm");
    }

    return output;
}

// ------------------------------------------------------------------
// RMSNorm 实现 (Root Mean Square Layer Normalization)
// Formula: x / RootMeanSquare(x) * weight
// ------------------------------------------------------------------

torch::Tensor custom_rmsnorm_forward(
    torch::Tensor input,
    std::vector<int64_t> normalized_shape,
    torch::Tensor weight, // gamma
    double eps) {

    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        torch::Tensor out = custom_rmsnorm_forward(input.to(torch::kFloat32), normalized_shape, weight.to(torch::kFloat32), eps);
        return out.to(torch::kBFloat16);
    }

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");

    // 1. 计算维度
    int64_t feature_dim = 1;
    for (auto dim : normalized_shape) {
        feature_dim *= dim;
    }

    int input_rank = input.dim();
    int norm_rank = normalized_shape.size();
    
    // 检查维度匹配
    TORCH_CHECK(input_rank >= norm_rank, "Input dimension must be >= normalized_shape dimension");
    for (int i = 0; i < norm_rank; ++i) {
        TORCH_CHECK(input.size(input_rank - norm_rank + i) == normalized_shape[i], 
            "Normalized shape mismatch");
    }

    int rows = input.numel() / feature_dim;
    int cols = feature_dim;

    // 2. 检查 Weight 形状和类型
    TORCH_CHECK(weight.numel() == cols, "Weight (gamma) size mismatch");
    TORCH_CHECK(weight.dtype() == input.dtype(), "Weight dtype mismatch");

    // 确保 weight 连续
    weight = weight.contiguous();

    // 3. 准备输出
    auto output = torch::empty_like(input);

    // 4. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_rmsnorm_fp32(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            rows, cols, (float)eps
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_rmsnorm_fp16(
            output.data_ptr<at::Half>(),
            input.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            rows, cols, (float)eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RMSNorm");
    }

    return output;
}

// ------------------------------------------------------------------
// Attention 实现 (Scaled Dot Product)
// ------------------------------------------------------------------

torch::Tensor custom_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    c10::optional<double> scale_arg) {

    // bf16 has no native arithmetic on Turing; convert at the boundary and
    // reuse an existing FMA-free kernel. UNLIKE every other op in this
    // file, this routes through FP16, not FP32: launch_attention_fp32
    // (attention_fp32_strict_kernel, fp32_attention.cu) is a naive
    // one-thread-per-query-row kernel with no warp-level cooperation across
    // the head-dim reduction -- measured at ~21.6s for a single S=17402
    // attention call. launch_attention_fp16 (flash_attention_fp16_turing,
    // fp16_attention.cu) is the properly warp-cooperative tiled kernel --
    // it's the same design later forked into sageattention/xformers'
    // turing_fma_free ports, measured at ~6.5ms for the same shape there.
    // fp16's narrower exponent range (vs fp32) is an accepted tradeoff here
    // because sageattention/xformers already validated bf16->fp16->bf16
    // round-tripping through this exact kernel design without overflow.
    if (query.scalar_type() == torch::kBFloat16) {
        torch::Tensor out = custom_attention_forward(
            query.to(torch::kFloat16), key.to(torch::kFloat16), value.to(torch::kFloat16), scale_arg);
        return out.to(torch::kBFloat16);
    }

    // 1. 基础检查
    TORCH_CHECK(query.dim() == 4, "Query must be 4D (Batch, Head, Seq, Dim)");
    TORCH_CHECK(key.dim() == 4, "Key must be 4D");
    TORCH_CHECK(value.dim() == 4, "Value must be 4D");
    
    // 假设形状都是 [B, H, S, D]
    int B = query.size(0);
    int H = query.size(1);
    int S = query.size(2);
    int D = query.size(3);

    TORCH_CHECK(key.size(0)==B && key.size(1)==H && key.size(2)==S && key.size(3)==D, "Key shape mismatch");
    TORCH_CHECK(value.size(0)==B && value.size(1)==H && value.size(2)==S && value.size(3)==D, "Value shape mismatch");
    TORCH_CHECK(query.device() == key.device() && query.device() == value.device(), "Device mismatch");

    // NOTE: no S<=4096 limit here -- both launch_attention_fp32 and
    // launch_attention_fp16 tile over K/V in chunks of TC rows
    // (num_tiles = ceil(S/TC)), so arbitrary S is handled correctly. A
    // stale printf() warning used to fire on every call for S>4096 (the
    // real, expected case for e.g. video sequence lengths); besides being
    // log spam on every forward pass, host-side printf can add real
    // wall-clock latency when stdout is a slow pipe/terminal, which was
    // skewing benchmark measurements. Removed.

    // 2. 准备数据
    query = query.contiguous();
    key = key.contiguous();
    value = value.contiguous();

    auto options = torch::TensorOptions().dtype(query.dtype()).device(query.device());
    torch::Tensor output = torch::empty({B, H, S, D}, options);

    float scale = 1.0f / std::sqrt((float)D);
    if (scale_arg.has_value()) {
        scale = (float)scale_arg.value();
    }

    // 3. 调度
#ifdef CMPEXT3_WITH_TENSORRT
    bool attention_done_by_trt = false;
    if (cmpext3_trt_enabled() &&
        (query.dtype() == torch::kFloat16 || query.dtype() == torch::kFloat32)) {
        bool is_fp16 = query.dtype() == torch::kFloat16;
        const void* q_ptr = is_fp16 ? (const void*)query.data_ptr<at::Half>() : (const void*)query.data_ptr<float>();
        const void* k_ptr = is_fp16 ? (const void*)key.data_ptr<at::Half>() : (const void*)key.data_ptr<float>();
        const void* v_ptr = is_fp16 ? (const void*)value.data_ptr<at::Half>() : (const void*)value.data_ptr<float>();
        attention_done_by_trt = cmpext3_trt_attention_forward(
            q_ptr, k_ptr, v_ptr, output.data_ptr(),
            B, H, S, D, scale, is_fp16,
            at::cuda::getCurrentCUDAStream(query.device().index()).stream());
    }
    if (!attention_done_by_trt) {
#endif
    if (query.dtype() == torch::kFloat32) {
        launch_attention_fp32(
            query.data_ptr<float>(),
            key.data_ptr<float>(),
            value.data_ptr<float>(),
            output.data_ptr<float>(),
            B, H, S, D, scale
        );
    } else if (query.dtype() == torch::kFloat16) {
        // fp16_attention.cu's flash_attention_fp16_optimized has D_DIM=128
        // hard-coded into its shared-memory tiling (see that file's D_DIM/
        // SMEM_STRIDE #defines and its [N3] note) -- launch_attention_fp16()
        // silently declines to launch (`if (D != D_DIM) return;`) for any
        // other head_dim, leaving `output` completely unwritten. Without
        // this check, custom_attention_forward would return that unwritten
        // (leftover-allocator-memory) tensor as if it were a valid result
        // -- confirmed directly: head_dim=64 silently produced all-zero
        // output, and other shapes produced NaN or wildly-wrong magnitudes
        // from stale memory, with no exception at any point. TORCH_CHECK
        // here instead, so cmpext3/__init__.py's own `_patched_sdpa`
        // (`except RuntimeError: return orig(...)`) correctly falls back
        // to stock SDPA for any head_dim != 128, matching every other
        // "unsupported input -> raise, let the caller fall back" contract
        // in this file. (fp32_attention.cu / launch_attention_fp32 has no
        // such constraint -- this only ever affected the fp16 path.)
        TORCH_CHECK(D == 128, "cmpext3 fp16 attention kernel only supports head_dim=128 (got ", D, ")");
        launch_attention_fp16(
            query.data_ptr<at::Half>(),
            key.data_ptr<at::Half>(),
            value.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            B, H, S, D, scale
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for attention");
    }
#ifdef CMPEXT3_WITH_TENSORRT
    }
#endif

    return output;
}


// ------------------------------------------------------------------
// GELU 实现 (Exact version using erf)
// Formula: 0.5 * x * (1 + erf(x / sqrt(2)))
// ------------------------------------------------------------------

torch::Tensor custom_gelu_forward(torch::Tensor input) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        return custom_gelu_forward(input.to(torch::kFloat32)).to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    // 2. 准备输出
    // 确保连续，避免 stride 问题
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 3. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_gelu_fp32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_gelu_fp16(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom GELU");
    }

    return output;
}

// ------------------------------------------------------------------
// SiLU (Swish) 实现
// Formula: x * sigmoid(x) = x / (1 + exp(-x))
// ------------------------------------------------------------------

torch::Tensor custom_silu_forward(torch::Tensor input, bool inplace = false) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        torch::Tensor out = custom_silu_forward(input.to(torch::kFloat32), false).to(torch::kBFloat16);
        if (inplace) { input.copy_(out); return input; }
        return out;
    }

    // 1. 检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    // Keep a handle to the tensor the caller actually passed in, so an
    // inplace=True request writes back through it even if we had to make
    // a contiguous copy below (F.silu(x, inplace=True) is a real call
    // pattern -- torch.nn.functional's own silu accepts it -- and previously
    // this native binding had no `inplace` parameter at all, so pybind11's
    // overload-resolution failure raised a TypeError whose message is built
    // by repr()'ing every argument, including the tensor; repr() on a CUDA
    // tensor runs masked_select/nonzero plus a .item() per printed element,
    // ~1000x slower than the op itself. Accepting the kwarg here means that
    // path is never taken.
    torch::Tensor caller_input = input;

    // 2. 准备输出
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 3. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_silu_fp32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_silu_fp16(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom SiLU");
    }

    if (inplace) {
        caller_input.copy_(output);
        return caller_input;
    }
    return output;
}

// ------------------------------------------------------------------
// Swish 实现 (x * sigmoid(beta * x))
// 修正版：beta 参数支持 float 或 Tensor
// ------------------------------------------------------------------

torch::Tensor custom_swish_forward(torch::Tensor input, py::object beta_arg) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        py::object beta_arg_f32 = beta_arg;
        if (py::isinstance<torch::Tensor>(beta_arg)) {
            torch::Tensor beta_t = beta_arg.cast<torch::Tensor>();
            if (beta_t.scalar_type() == torch::kBFloat16) {
                beta_arg_f32 = py::cast(beta_t.to(torch::kFloat32));
            }
        }
        return custom_swish_forward(input.to(torch::kFloat32), beta_arg_f32).to(torch::kBFloat16);
    }

    // 1. 检查 Input
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    // 2. 处理 Beta 参数
    torch::Tensor beta;

    if (py::isinstance<py::float_>(beta_arg) || py::isinstance<py::int_>(beta_arg)) {
        // 情况 A: 传入的是 Python float/int
        double val = beta_arg.cast<double>();
        // 在 Input 所在的设备上创建一个标量 Tensor
        beta = torch::tensor(val, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    } 
    else if (py::isinstance<torch::Tensor>(beta_arg)) {
        // 情况 B: 传入的是 Tensor
        beta = beta_arg.cast<torch::Tensor>();
        TORCH_CHECK(beta.is_cuda(), "Beta tensor must be on CUDA");
        TORCH_CHECK(beta.dtype() == input.dtype(), "Input and Beta dtype mismatch");
        TORCH_CHECK(beta.numel() == 1, "Beta must be a scalar (1-element tensor)");
    } 
    else {
        TORCH_CHECK(false, "Beta must be a float, int, or scalar Tensor");
    }

    // 3. 准备内存
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    // 刚刚创建的 scalar tensor 肯定是 contiguous 的，但为了安全：
    if (!beta.is_contiguous()) {
        beta = beta.contiguous();
    }
    
    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 4. 调度 (传入 beta.data_ptr)
    // 注意：无论 beta 是从 float 转换来的还是原本就是 Tensor，现在它都在 GPU 上
    if (input.dtype() == torch::kFloat32) {
        launch_swish_fp32(
            input.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_swish_fp16(
            input.data_ptr<at::Half>(),
            beta.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom Swish");
    }

    return output;
}

// ------------------------------------------------------------------
// Mish 实现
// Formula: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
// ------------------------------------------------------------------

torch::Tensor custom_mish_forward(torch::Tensor input, bool inplace = false) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        torch::Tensor out = custom_mish_forward(input.to(torch::kFloat32), false).to(torch::kBFloat16);
        if (inplace) { input.copy_(out); return input; }
        return out;
    }

    // 1. 检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    // See custom_silu_forward for why `inplace` must be a real accepted
    // parameter here rather than left to raise a TypeError.
    torch::Tensor caller_input = input;

    // 2. 准备内存
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 3. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_mish_fp32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_mish_fp16(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom Mish");
    }

    if (inplace) {
        caller_input.copy_(output);
        return caller_input;
    }
    return output;
}

// ------------------------------------------------------------------
// Softmax 实现
// ------------------------------------------------------------------

torch::Tensor custom_softmax_forward(torch::Tensor input, int64_t dim, bool half_to_float) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        return custom_softmax_forward(input.to(torch::kFloat32), dim, half_to_float).to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    // 2. 处理维度: 将目标 dim 移到最后，并展平为 2D [Batch_Size, Softmax_Dim]
    // 这样 Kernel 只需要处理 [Rows, Cols] 的情况
    // transpose 不拷贝内存，但 contiguous 会拷贝，确保内存连续
    int64_t ndim = input.dim();
    // 处理负数 dim (e.g. -1)
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "Dim out of range");

    torch::Tensor input_view = input;
    if (dim != ndim - 1) {
        input_view = input.transpose(dim, ndim - 1);
    }
    input_view = input_view.contiguous();

    int cols = input_view.size(-1);
    int rows = input_view.numel() / cols;

    // 3. 准备输出
    torch::Tensor output = torch::empty_like(input_view);

    // 4. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_softmax_fp32(
            input_view.data_ptr<float>(),
            output.data_ptr<float>(),
            rows, cols
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_softmax_fp16(
            input_view.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            rows, cols
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom softmax");
    }

    // 5. 恢复形状
    // 如果之前 transpose 过，输出也要 transpose 回去
    if (dim != ndim - 1) {
        output = output.transpose(dim, ndim - 1);
    }
    
    return output.contiguous(); // 确保返回连续内存
}

// ------------------------------------------------------------------
// Softplus 实现
// Formula: (1/beta) * log(1 + exp(beta * x))
// For numerical stability: return x if (input * beta) > threshold
// ------------------------------------------------------------------

torch::Tensor custom_softplus_forward(torch::Tensor input, double beta, double threshold) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        return custom_softplus_forward(input.to(torch::kFloat32), beta, threshold).to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    // 2. 准备输出
    // 确保连续
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 3. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_softplus_fp32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements,
            (float)beta,
            (float)threshold
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_softplus_fp16(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements,
            (float)beta,
            (float)threshold
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom Softplus");
    }

    return output;
}


// ------------------------------------------------------------------
// Softsign 实现
// ------------------------------------------------------------------

torch::Tensor custom_softsign_forward(torch::Tensor input) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        return custom_softsign_forward(input.to(torch::kFloat32)).to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    // 2. 准备输出
    // 确保连续
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 3. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_softsign_fp32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_softsign_fp16(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom Softsign");
    }

    return output;
}


// ------------------------------------------------------------------
// Softshrink 实现
// ------------------------------------------------------------------

torch::Tensor custom_softshrink_forward(torch::Tensor input, double lambd) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        return custom_softshrink_forward(input.to(torch::kFloat32), lambd).to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(lambd >= 0, "Lambda must be no less than zero");

    // 2. 准备输出
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 3. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_softshrink_fp32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements,
            (float)lambd
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_softshrink_fp16(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements,
            (float)lambd
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom Softshrink");
    }

    return output;
}

// ------------------------------------------------------------------
// UpSample实现
// ------------------------------------------------------------------


// ------------------------------------------------------------------
// Tanh 实现
// ------------------------------------------------------------------

torch::Tensor custom_tanh_forward(torch::Tensor input) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        return custom_tanh_forward(input.to(torch::kFloat32)).to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    // 2. 准备输出
    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 3. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_tanh_fp32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_tanh_fp16(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom_tanh");
    }

    return output;
}



// ------------------------------------------------------------------
// ERF (Error Function) 实现
// ------------------------------------------------------------------

torch::Tensor custom_erf_forward(torch::Tensor input) {
    // bf16 has no native arithmetic on Turing; convert to fp32 at the
    // boundary and reuse the FMA-free fp32 kernel below.
    if (input.scalar_type() == torch::kBFloat16) {
        return custom_erf_forward(input.to(torch::kFloat32)).to(torch::kBFloat16);
    }

    // 1. 检查
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous for custom ERF");
    
    // 2. 准备输出
    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // 3. 调度
    if (input.dtype() == torch::kFloat32) {
        launch_erf_fp32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            total_elements
        );
    } else if (input.dtype() == torch::kFloat16) {
        launch_erf_fp16(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom ERF");
    }

    return output;
}


// ------------------------------------------------------------------
// 绑定模块 
// ------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //绑定Linear
    m.def("linear", &custom_linear_wrapper, "Custom linear layer implementation",
          py::arg("input"), 
          py::arg("weight"), 
          py::arg("bias") = py::none());

    // BMM
    m.def("bmm", &custom_bmm_forward, "Custom Batch Matrix Multiplication",
          py::arg("input"),
          py::arg("mat2"));

    // Conv2d
    m.def("conv2d", &custom_conv2d_wrapper, "Custom conv2d layer",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          // 注意：这里默认值稍微改一下写法，或者在 wrapper 里处理默认值
          // 为了简单，我们可以在 Python 端设默认值，或者在这里给默认 Int
          py::arg("stride") = 1,   
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("groups") = 1);

    // Conv3d
    m.def("conv3d", &custom_conv3d_wrapper, "Custom conv3d layer",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("groups") = 1);

    // ConvTranspose2d
    m.def("conv_transpose2d", &custom_conv_transpose2d_wrapper, "Custom conv_transpose2d layer",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("dilation") = 1,
          py::arg("groups") = 1);

    // 使用 ConvTranspose2d 进行 Upsample
    // 这个版本通常比手写的非优化 Kernel 快
    m.def("upsample_scaling", &custom_upsample_smart, 
          "Upsample using optimized ConvTranspose2d (Nearest Neighbor). Supports int scale or (H, W) size.",
          py::arg("input"),
          py::arg("size_or_scale"));
          

    //Attention      
    m.def("attention", &custom_attention_forward, "Custom SDPA Attention",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("scale") = py::none());

    // Embedding
    m.def("embedding", &custom_embedding_forward, "Custom Embedding Layer",
          py::arg("input"),
          py::arg("weight"),
          py::arg("padding_idx") = -1,
          py::arg("scale_grad_by_freq") = false,
          py::arg("sparse") = false);

    // GroupNorm
    m.def("group_norm", &custom_group_norm_forward, "Custom GroupNorm",
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("weight") = py::none(),
          py::arg("bias") = py::none(),
          py::arg("eps") = 1e-5,
          py::arg("apply_silu") = false);

    // GroupNorm+SiLU (fused) -- explicit opt-in op, see the comment above
    // custom_group_norm_silu_forward. Not monkeypatched onto anything;
    // call directly in place of F.group_norm(...) followed by F.silu(...).
    m.def("group_norm_silu", &custom_group_norm_silu_forward, "Custom fused GroupNorm+SiLU",
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("weight") = py::none(),
          py::arg("bias") = py::none(),
          py::arg("eps") = 1e-5);

    // LayerNorm
    m.def("layer_norm", &custom_layernorm_forward, "Custom LayerNorm",
          py::arg("input"),
          py::arg("normalized_shape"),
          py::arg("weight") = py::none(),
          py::arg("bias") = py::none(),
          py::arg("eps") = 1e-5);

    // RMSNorm
    m.def("rmsnorm", &custom_rmsnorm_forward, "Custom RMSNorm",
          py::arg("input"),
          py::arg("normalized_shape"),
          py::arg("weight"),
          py::arg("eps") = 1e-6); // RMSNorm 常用默认 eps

    // GELU
    m.def("gelu", &custom_gelu_forward, "Custom GELU activation",
          py::arg("input"));

    // SiLU
    m.def("silu", &custom_silu_forward, "Custom SiLU activation",
          py::arg("input"),
          py::arg("inplace") = false);

    // Swish with beta
    m.def("swish", &custom_swish_forward, "Custom Swish activation with learnable beta",
          py::arg("input"),
          py::arg("beta"));

    // Mish
    m.def("mish", &custom_mish_forward, "Custom Mish activation",
          py::arg("input"),
          py::arg("inplace") = false);

    // Softmax
    m.def("softmax", &custom_softmax_forward, "Custom Softmax",
          py::arg("input"),
          py::arg("dim") = -1,
          py::arg("half_to_float") = false); // 预留接口，暂未在Kernel中使用

    // Softplus
    m.def("softplus", &custom_softplus_forward, "Custom Softplus activation",
          py::arg("input"),
          py::arg("beta") = 1.0,
          py::arg("threshold") = 20.0);

    // Softsign
    m.def("softsign", &custom_softsign_forward, "Custom Softsign activation",
          py::arg("input"));

    // Softshrink
    m.def("softshrink", &custom_softshrink_forward, "Custom Softshrink activation",
          py::arg("input"),
          py::arg("lambd") = 0.5);

    // Tanh
    m.def("tanh", &custom_tanh_forward, "Custom Tanh based on exp expansion and FP16 RCP",
          py::arg("input"));

    // ERF
    m.def("erf", &custom_erf_forward, "Custom ERF approximation",
          py::arg("input"));
}