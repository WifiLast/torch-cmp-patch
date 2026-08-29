// =================================================================================
// matmul_trt_runtime.cpp -- see matmul_trt_runtime.h and trt_common.h
// (UNVERIFIED -- read trt_common.h's design-rationale comment before
// trusting this file).
// =================================================================================

#include "matmul_trt_runtime.h"
#include "trt_common.h"

#include <cstring>

using namespace nvinfer1;
using namespace cmpext3_trt;

namespace {

std::vector<char> build_engine(
    const std::vector<char>& weight_host,
    int M, int N, int K,
    bool is_fp16)
{
    Ptr<IBuilder> builder(createInferBuilder(logger()));
    if (!builder) return {};

    // TensorRT 10+ removed NetworkDefinitionCreationFlag::kEXPLICIT_BATCH --
    // createNetworkV2() supports only explicit batch + strongly-typed
    // networks now (see its own doc comment in NvInfer.h), so no flags are
    // needed at all.
    Ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) return {};

    DataType dtype = is_fp16 ? DataType::kHALF : DataType::kFLOAT;
    size_t elem_size = is_fp16 ? sizeof(uint16_t) : sizeof(float);

    Dims2 input_dims{M, K};
    ITensor* input_tensor = network->addInput("input", dtype, input_dims);
    if (!input_tensor) return {};

    // weight is baked in as a build-time constant (same treatment as
    // conv's weight/bias) -- it was already copied host-side by the
    // caller specifically so it can be passed here.
    Weights weight_w{dtype, weight_host.data(), static_cast<int64_t>(weight_host.size() / elem_size)};
    Dims2 weight_dims{K, N};
    IConstantLayer* weight_const = network->addConstant(weight_dims, weight_w);
    if (!weight_const) return {};

    IMatrixMultiplyLayer* mm = network->addMatrixMultiply(
        *input_tensor, MatrixOperation::kNONE,
        *weight_const->getOutput(0), MatrixOperation::kNONE);
    if (!mm) return {};

    ITensor* output_tensor = mm->getOutput(0);
    output_tensor->setName("output");
    network->markOutput(*output_tensor);

    Ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return {};
    // TensorRT 10+ removed BuilderFlag::kFP16 (and every other precision
    // hint flag) -- networks are always strongly typed now, so precision
    // is inferred purely from the DataType already set above on the input
    // tensor and Weights, nothing left to opt into here.
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

    Ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) return {};

    return std::vector<char>(
        static_cast<const char*>(serialized->data()),
        static_cast<const char*>(serialized->data()) + serialized->size());
}

}  // namespace

bool cmpext3_trt_matmul_forward(
    const void* input, const void* weight, void* output,
    int M, int N, int K,
    bool is_fp16,
    cudaStream_t stream)
{
    const size_t elem_size = is_fp16 ? sizeof(uint16_t) : sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(K) * N * elem_size;

    std::vector<char> weight_host(weight_bytes);
    if (!copy_to_host(weight, weight_host.data(), weight_bytes, stream)) return false;

    struct ShapeKey {
        int M, N, K;
        bool is_fp16;
    } shape_key;
    // Zero the WHOLE struct first, padding included -- see
    // conv3d_trt_runtime.cpp's identical fix for why: aggregate
    // list-initialization leaves padding bytes as uninitialized stack
    // garbage, which would otherwise make this hash non-reproducible for
    // the identical shape.
    std::memset(&shape_key, 0, sizeof(shape_key));
    shape_key.M = M; shape_key.N = N; shape_key.K = K; shape_key.is_fp16 = is_fp16;

    uint64_t h = fnv1a_update(&shape_key, sizeof(shape_key), FNV_OFFSET_BASIS);
    h = fnv1a_update(weight_host.data(), weight_host.size(), h);
    std::string key = "matmul_" + hash_to_hex(h);

    auto entry = get_or_build_engine(key, [&]() {
        return build_engine(weight_host, M, N, K, is_fp16);
    });
    if (!entry) return false;

    if (!entry->context->setTensorAddress("input", const_cast<void*>(input)) ||
        !entry->context->setTensorAddress("output", output)) {
        return false;
    }
    return entry->context->enqueueV3(stream);
}
