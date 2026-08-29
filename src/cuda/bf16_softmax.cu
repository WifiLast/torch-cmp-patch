#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

// 辅助函数：将 val 的符号位反转，实现 -val，避免隐式转换
__device__ __forceinline__ float neg_f(float x) {
    return -x;
}

// 辅助规约：使用 fmaxf 替代 max
__device__ __forceinline__ float warpReduceMaxB(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// 辅助规约：使用 __fadd_rn 替代 +
__device__ __forceinline__ float warpReduceSumB(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = __fadd_rn(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__global__ void softmax_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int rows, int cols) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    // 计算行偏移
    const int row_offset = row_idx * cols;
    const __nv_bfloat16* row_input = input + row_offset;
    __nv_bfloat16* row_output = output + row_offset;

    // float4 向量化访问 (LDG.128/STG.128) 要求 16 字节对齐。row_input/row_output
    // = base + row_idx*cols*sizeof(bf16)。只有当 cols 是 8 的倍数时，每一行相对
    // base 的偏移量才必然是 16 字节对齐的；否则 row_idx >= 1 的行会产生
    // CUDA_ERROR_MISALIGNED_ADDRESS。按行校验实际指针对齐后再决定是否走向量化
    // 路径 -- 未对齐的行 vec_end=0，整体退化为下面的标量循环（覆盖全部 cols）。
    bool row_aligned =
        ((reinterpret_cast<uintptr_t>(row_input) & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(row_output) & 0xF) == 0);
    int vec_end = row_aligned ? (cols / 8) * 8 : 0;

    // -------------------------------------------------------------------------
    // 1. Find Max
    // -------------------------------------------------------------------------
    float local_max = -1e37f; // 或者使用 -INFINITY

    // 向量化部分：每次处理 8 个 BF16 (128 bits / 16 bytes)
    // 使用 float4 类型作为 128 位容器
    int i = threadIdx.x * 8;
    int stride = blockDim.x * 8;

    for (; i < vec_end; i += stride) {
        // 使用 float4 加载 16 字节 (8个BF16)
        float4 vec_data = *reinterpret_cast<const float4*>(row_input + i);

        // 将 float4 (128 bits) 重新解释为 8 个 __nv_bfloat16
        // 我们通过 union 或 指针转换来解包
        const __nv_bfloat16* pack = reinterpret_cast<const __nv_bfloat16*>(&vec_data);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val_f = __bfloat162float(pack[k]);
            local_max = fmaxf(local_max, val_f);
        }
    }

    // 标量循环：处理 [vec_end, cols) 范围内的所有剩余元素
    // (向量化路径关闭时 vec_end=0，即为 [0, cols)，而不只是尾部)。
    for (int j = vec_end + threadIdx.x; j < cols; j += blockDim.x) {
        local_max = fmaxf(local_max, __bfloat162float(row_input[j]));
    }

    // Warp Reduction
    local_max = warpReduceMaxB(local_max);

    __shared__ float shared_val[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) shared_val[wid] = local_max;
    __syncthreads();

    // Block Reduction (假设 blockDim.x <= 1024, max 32 warps)
    float global_max = local_max;
    if (threadIdx.x == 0) {
        float block_max = -1e37f;
        int N_warps = (blockDim.x + 31) / 32;
        for (int k = 0; k < N_warps; ++k) {
            block_max = fmaxf(block_max, shared_val[k]);
        }
        shared_val[0] = block_max; // Reuse shared[0] for broadcast
    }
    __syncthreads();
    global_max = shared_val[0];

    // -------------------------------------------------------------------------
    // 2. Sum (Exp(val - max))
    // -------------------------------------------------------------------------
    float local_sum = 0.0f;
    float neg_global_max = neg_f(global_max); // -global_max

    // Vectorized Loop
    i = threadIdx.x * 8;
    for (; i < vec_end; i += stride) {
        float4 vec_data = *reinterpret_cast<const float4*>(row_input + i);
        const __nv_bfloat16* pack = reinterpret_cast<const __nv_bfloat16*>(&vec_data);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val_f = __bfloat162float(pack[k]);
            // Constraint: No FMA, Explicit __fadd_rn
            float diff = __fadd_rn(val_f, neg_global_max);
            // Constraint: internal expf -> __expf
            float e = __expf(diff);
            local_sum = __fadd_rn(local_sum, e);
        }
    }

    // Scalar Tail
    for (int j = vec_end + threadIdx.x; j < cols; j += blockDim.x) {
        float val_f = __bfloat162float(row_input[j]);
        float diff = __fadd_rn(val_f, neg_global_max);
        local_sum = __fadd_rn(local_sum, __expf(diff));
    }

    // Warp Reduction
    local_sum = warpReduceSumB(local_sum);
    
    if (lane == 0) shared_val[wid] = local_sum;
    __syncthreads();

    // Block Reduction
    float global_sum = local_sum;
    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        int N_warps = (blockDim.x + 31) / 32;
        for (int k = 0; k < N_warps; ++k) {
            block_sum = __fadd_rn(block_sum, shared_val[k]);
        }
        shared_val[0] = block_sum;
    }
    __syncthreads();
    global_sum = shared_val[0];

    // -------------------------------------------------------------------------
    // 3. Normalize & Write
    // -------------------------------------------------------------------------
    // Constraint: rcp must use __fdividef
    float inv_sum = __fdividef(1.0f, global_sum);

    // Vectorized Loop
    i = threadIdx.x * 8;
    for (; i < vec_end; i += stride) {
        // Load
        float4 vec_data = *reinterpret_cast<const float4*>(row_input + i);
        const __nv_bfloat16* input_pack = reinterpret_cast<const __nv_bfloat16*>(&vec_data);
        
        // Prepare output buffer in registers
        __nv_bfloat16 output_pack[8];

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val_f = __bfloat162float(input_pack[k]);
            float diff = __fadd_rn(val_f, neg_global_max);
            float e = __expf(diff);
            // Constraint: No div in loop, use mul with rcp
            float norm = __fmul_rn(e, inv_sum);
            output_pack[k] = __float2bfloat16(norm);
        }

        // Vector Store (128-bit)
        // Need to pack 8 bf16s back into float4 container to store efficiently
        *reinterpret_cast<float4*>(row_output + i) = *reinterpret_cast<float4*>(output_pack);
    }

    // Scalar Tail
    for (int j = vec_end + threadIdx.x; j < cols; j += blockDim.x) {
        float val_f = __bfloat162float(row_input[j]);
        float diff = __fadd_rn(val_f, neg_global_max);
        float e = __expf(diff);
        float norm = __fmul_rn(e, inv_sum);
        row_output[j] = __float2bfloat16(norm);
    }
}

void launch_softmax_bf16(const void* input, void* output, int rows, int cols) {
    int block_size = 256;
    // 确保 rows 有足够的 grid size
    softmax_kernel_bf16<<<rows, block_size>>>(
        (const __nv_bfloat16*)input, 
        (__nv_bfloat16*)output, 
        rows, cols
    );
}
//[8192x8192]： 0.231 ms, Avg Power:  77.23 W 22.2x
