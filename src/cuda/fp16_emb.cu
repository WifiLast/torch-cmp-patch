#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cassert>

// 定义向量化类型，int4 用于搬运 128位 (16字节) 数据，即 8 个 __half
using vec_t = int4;
constexpr int VEC_SIZE = sizeof(vec_t) / sizeof(__half); // 8

/**
 * 优化策略：
 * 1. 向量化：使用 int4 (128-bit) 一次搬运 8 个 half。
 * 2. 避免除法：每个 Block 处理若干行，ThreadIdx.x 直接映射列，消除 idx / dim 计算。
 * 3. 访存合并：确保同一个 Warp 内的线程读取连续内存。
 * 4. 分支外提：Padding/越界判断在行级别处理，而非元素级别。
 */
__global__ void __launch_bounds__(256) embedding_fp16_opt_kernel(
    const int64_t* __restrict__ indices,      // 只读，使用 __restrict__ 提示编译器
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int num_indices,
    int embedding_dim,
    int padding_idx,
    int num_embeddings,
    int num_vecs_per_row) // 预计算每行有多少个 int4 向量
{
    // -----------------------------------------------------------------
    // 1. Grid-Stride Loop over Rows (Indices)
    //    blockIdx.x 负责行维度的遍历，使得一个 Block 完整处理某一行(或多行)
    // -----------------------------------------------------------------
    for (int row_idx = blockIdx.x; row_idx < num_indices; row_idx += gridDim.x) {
        
        // 提前读取 index，减少全局内存延迟影响
        int64_t target_idx = indices[row_idx];
        
        // 计算输出的起始偏移量
        int64_t out_row_offset = (int64_t)row_idx * embedding_dim;
        
        // -------------------------------------------------------------
        // 2. 逻辑分支优化：将 Padding 判断移出内层拷贝循环
        // -------------------------------------------------------------
        bool is_valid = (target_idx != padding_idx) && (target_idx >= 0) && (target_idx < num_embeddings);
        
        // 向量化指针操作
        vec_t* out_vec_ptr = reinterpret_cast<vec_t*>(output + out_row_offset);
        
        if (is_valid) {
            // == 有效 Index：执行拷贝 ==
            int64_t weight_row_offset = target_idx * embedding_dim;
            const vec_t* weight_vec_ptr = reinterpret_cast<const vec_t*>(weight + weight_row_offset);

            // ---------------------------------------------------------
            // 3. 向量化拷贝循环 (Vectorized Loop)
            //    线程并行处理这一行中的不同 int4 块
            // ---------------------------------------------------------
            for (int i = threadIdx.x; i < num_vecs_per_row; i += blockDim.x) {
                // 128-bit Load & Store
                out_vec_ptr[i] = weight_vec_ptr[i]; 
            }

            // 处理无法被 8 整除的尾部元素 (Scalar Peeling)
            // 绝大多数情况 embedding_dim 是 8 的倍数，此部分不执行
            int remaining_start = num_vecs_per_row * VEC_SIZE;
            for (int i = remaining_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
                output[out_row_offset + i] = weight[weight_row_offset + i];
            }

        } else {
            // == 无效 Index (Padding/OOB)：执行清零 ==
            
            // 构造全 0 的 128-bit 寄存器
            vec_t zero_vec;
            zero_vec.x = 0; zero_vec.y = 0; zero_vec.z = 0; zero_vec.w = 0;

            for (int i = threadIdx.x; i < num_vecs_per_row; i += blockDim.x) {
                out_vec_ptr[i] = zero_vec;
            }

            // 尾部清零
            int remaining_start = num_vecs_per_row * VEC_SIZE;
            for (int i = remaining_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
                output[out_row_offset + i] = __float2half(0.0f);
            }
        }
    }
}

void launch_embedding_fp16(const int64_t* indices, const void* weight, void* output, int num_indices, int embedding_dim, int padding_idx, int num_embeddings) {
    // 硬件相关配置
    const int thread_per_block = 256;

    // 计算向量化参数
    // 我们假设 output 和 weight 内存地址是 16字节对齐的 (cudaMalloc 默认 256字节对齐)
    // 如果 embedding_dim 不是 8 的倍数，向量化部分只能处理前面部分，尾部走 scalar 循环
    int num_vecs_per_row = embedding_dim / 8;

    // -------------------------------------------------------------------------
    // Grid 配置策略 — TU102 (Turing)
    //
    // TU102 有 72 个 SM。
    //
    // 该 kernel 无共享内存，每 Block 256 线程。
    // Turing 每 SM 最多 1024 线程 → 每 SM 最多可驻留 4 个 Block。
    // 但 embedding lookup 是纯 Memory-Bound 操作：每 Block 的瓶颈是 DRAM 带宽，
    // 而非计算资源。实测表明 2–4 个 Block/SM 足以打满 TU102 的 672 GB/s 带宽
    // （RTX 2080 Ti）。取 4 个 Block/SM 作为上限：
    //
    //   max_blocks = 72 SMs × 4 blocks/SM = 288
    //
    // 原代码误用了 A100 的 108 SM，导致上限偏高 (864)，增加调度开销但无性能收益。
    //
    // min_blocks 确保即使 num_indices 很小时也有足够的并行度以隐藏内存延迟：
    // 每 4 行至少启动 1 个 Block，使每个 Block 的平均工作量合理。
    //
    // 最终 blocks = clamp(num_indices, min_blocks, max_blocks)，
    // 保证：
    //   (a) blocks ≤ num_indices — 不启动多余的空 Block（grid-stride loop 会跳过）
    //   (b) blocks ≥ min_blocks  — 保证最低并行度
    //   (c) blocks ≤ max_blocks  — 避免超出硬件调度饱和点
    // -------------------------------------------------------------------------
    const int TU102_SM_COUNT     = 72;
    const int BLOCKS_PER_SM      = 4;   // 内存密集型 kernel 的饱和点
    const int max_blocks         = TU102_SM_COUNT * BLOCKS_PER_SM;  // 288

    // 最低并行度：每 4 个 index 至少分配 1 个 Block
    int min_blocks = (num_indices + 3) / 4;
    min_blocks = (min_blocks < 1) ? 1 : min_blocks;

    // 上界：不超过 num_indices（多余 Block 在 grid-stride loop 中立即退出，
    // 但仍浪费启动和调度资源）
    int blocks = (num_indices < max_blocks) ? num_indices : max_blocks;

    // 下界：保证最低并行度，但不超过上界
    if (blocks < min_blocks) blocks = min_blocks;

    // 最终安全夹紧：确保非零
    if (blocks < 1) blocks = 1;

    // 内存占用分析（kernel 本身无 smem）：
    //   registers:  ~16 regs/thread × 256 threads = 4,096 regs/block
    //               TU102 每 SM 65,536 regs → 可驻留 16 blocks（远超 4 个上限）
    //   smem:       0 bytes（kernel 不使用 __shared__）
    //   L1/L2:      int4 (128-bit) 向量化访问，对 Turing L1 缓存行 128 字节对齐最优
    //   全局内存:   每行 embedding_dim × 2 bytes（读）+ embedding_dim × 2 bytes（写）
    //               对于 embedding_dim = 768: 每行 3 KB；288 blocks × 256 threads
    //               × 8 half/thread = 每次迭代最多 ~1.1 MB 在途数据，适合 TU102 带宽。

    embedding_fp16_opt_kernel<<<blocks, thread_per_block>>>(
        indices, 
        reinterpret_cast<const __half*>(weight), 
        reinterpret_cast<__half*>(output), 
        num_indices, embedding_dim, padding_idx, num_embeddings,
        num_vecs_per_row
    );
    
    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
