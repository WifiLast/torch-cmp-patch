#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#define MAX_BLOCK_SIZE 256
#define WARP_SIZE 32

// 辅助函数：严格禁止FMA的FP32加法
__device__ __forceinline__ float add_no_fma(float a, float b) {
    return __fadd_rn(a, b);
}

// 辅助函数：严格禁止FMA的FP32乘法
__device__ __forceinline__ float mul_no_fma(float a, float b) {
    return __fmul_rn(a, b);
}

// Warp Reduce Max (float)
__device__ float warpReduceMaxF(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp Reduce Sum (float) - using explicit add
__device__ float warpReduceSumF(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = add_no_fma(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__global__ void softmax_kernel_opt_ga100(const __half* __restrict__ input, __half* __restrict__ output, int rows, int cols) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    // 计算当前行的偏移量
    const __half* row_input = input + row_idx * cols;
    __half* row_output = output + row_idx * cols;

    // half2 向量化访问要求 4 字节对齐。当 cols 为奇数时，row_idx >= 1 的行
    // 相对 base 的偏移量（单位: half）为奇数，导致地址非 4 字节对齐，
    // 引发 CUDA_ERROR_MISALIGNED_ADDRESS。按行校验对齐，未对齐的行整体
    // 退化为标量路径（vec_limit=0，下面的尾部标量循环会覆盖全部 cols）。
    bool row_aligned =
        ((reinterpret_cast<uintptr_t>(row_input) & 0x3) == 0) &&
        ((reinterpret_cast<uintptr_t>(row_output) & 0x3) == 0);

    // Shared Memory用于Block规约
    __shared__ float s_data[32]; // 假设最大1024线程，即32个Warp

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // ==========================================
    // 1. 寻找最大值 (Find Max)
    // ==========================================
    float local_max = -1e37f;

    // 向量化部分 (以half2读取)。若本行地址未对齐 (cols为奇数导致的行), 则
    // vec_limit=0, 下面的尾部循环会以标量方式覆盖全部 cols 个元素。
    int vec_limit = row_aligned ? (cols / 2) : 0;
    const __half2* row_input_h2 = (const __half2*)row_input;

    for (int i = tid; i < vec_limit; i += blockDim.x) {
        __half2 v2 = row_input_h2[i];
        float2 vf2 = __half22float2(v2);
        local_max = fmaxf(local_max, fmaxf(vf2.x, vf2.y));
    }

    // 尾部标量循环：覆盖 [vec_limit*2, cols) 范围内的所有剩余元素
    // (向量化路径关闭时即为 [0, cols)，而不只是最多一个元素)。
    for (int i = vec_limit * 2 + tid; i < cols; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        local_max = fmaxf(local_max, val);
    }

    // Warp内规约
    local_max = warpReduceMaxF(local_max);

    // 将Warp结果写入Shared Memory
    if (lane == 0) s_data[wid] = local_max;
    __syncthreads();

    // Block内规约 (由第一个Warp完成)
    float block_max = -1e37f;
    if (wid == 0) {
        if (lane < num_warps) block_max = s_data[lane];
        else block_max = -1e37f;
        block_max = warpReduceMaxF(block_max);
    }

    // 广播最大值
    if (tid == 0) s_data[0] = block_max;
    __syncthreads();
    float global_max = s_data[0];

    // ==========================================
    // 2. 计算指数和 (Sum Exp)
    // ==========================================
    float local_sum = 0.0f;

    // 向量化循环
    for (int i = tid; i < vec_limit; i += blockDim.x) {
        __half2 v2 = row_input_h2[i];
        float2 vf2 = __half22float2(v2);

        // 约束1 & 4: 不用FMA，必须转float用__expf
        // x - max
        float diff1 = add_no_fma(vf2.x, -global_max);
        float diff2 = add_no_fma(vf2.y, -global_max); // -x 等同于 +(-x)

        // exp(x - max)
        float val1 = __expf(diff1);
        float val2 = __expf(diff2);

        // sum += val (使用 __fadd_rn)
        local_sum = add_no_fma(local_sum, add_no_fma(val1, val2));
    }

    // 尾部标量循环
    for (int i = vec_limit * 2 + tid; i < cols; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        float diff = add_no_fma(val, -global_max);
        local_sum = add_no_fma(local_sum, __expf(diff));
    }

    // Warp内规约
    local_sum = warpReduceSumF(local_sum);

    // 存入Shared Memory
    if (lane == 0) s_data[wid] = local_sum;
    __syncthreads();

    // Block内规约
    float block_sum = 0.0f;
    if (wid == 0) {
        if (lane < num_warps) block_sum = s_data[lane];
        else block_sum = 0.0f;
        block_sum = warpReduceSumF(block_sum);
    }

    // ==========================================
    // 3. 计算倒数 (Constraint 7)
    // ==========================================
    // 要求：不要使用FP32 __frcp_rn，必须转为FP16向量使用h2rcp
    if (tid == 0) {
        // 复制sum到half2的两个通道 (S, S)
        __half2 sum_h2 = __float2half2_rn(block_sum);
        // 使用向量化FP16倒数指令
        sum_h2 = h2rcp(sum_h2);
        // 转回float，取出其中一个
        s_data[0] = __low2float(sum_h2);
    }
    __syncthreads();
    float global_inv_sum = s_data[0];

    // ==========================================
    // 4. 计算并写回 (Write Output)
    // ==========================================
    __half2* row_output_h2 = (__half2*)row_output;

    for (int i = tid; i < vec_limit; i += blockDim.x) {
        // 重新读取 (为了寄存器压力通常重读，或者如果寄存器够可以缓存)
        __half2 v2 = row_input_h2[i];
        float2 vf2 = __half22float2(v2);

        // 计算 exp(x - max)
        float e1 = __expf(add_no_fma(vf2.x, -global_max));
        float e2 = __expf(add_no_fma(vf2.y, -global_max));

        // 乘倒数: e * inv_sum (No FMA)
        float res1 = mul_no_fma(e1, global_inv_sum);
        float res2 = mul_no_fma(e2, global_inv_sum);

        // 转回half2并写入
        row_output_h2[i] = __float22half2_rn({res1, res2});
    }

    // 尾部标量写回
    for (int i = vec_limit * 2 + tid; i < cols; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        float e = __expf(add_no_fma(val, -global_max));
        float res = mul_no_fma(e, global_inv_sum);
        row_output[i] = __float2half(res);
    }
}

void launch_softmax_fp16(const void* input, void* output, int rows, int cols) {
    // 针对每个Row启动一个Block
    int block_size = 256;
    // 根据cols大小调整block_size通常更好，但此处固定256以匹配原逻辑
    if (cols < 256) block_size = 128;
    if (cols < 64) block_size = 32; // Warp size

    softmax_kernel_opt_ga100<<<rows, block_size>>>(
        (const __half*)input,
        (__half*)output,
        rows, cols
    );
}
//[8192x8192]：0.467 ms