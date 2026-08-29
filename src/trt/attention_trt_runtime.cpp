// =================================================================================
// attention_trt_runtime.cpp -- see attention_trt_runtime.h and
// trt_common.h (UNVERIFIED -- read trt_common.h's design-rationale
// comment before trusting this file; this is the most structurally
// complex of the four runtime TRT paths -- three chained layers instead
// of one -- so it carries more risk of a subtle construction mistake than
// conv2d/conv3d/matmul do).
// =================================================================================

#include "attention_trt_runtime.h"
#include "trt_common.h"

#include <cstring>

#include <cstring>

using namespace nvinfer1;
using namespace cmpext3_trt;

namespace {

std::vector<char> build_engine(int B, int H, int S, int D, float scale, bool is_fp16) {
    Ptr<IBuilder> builder(createInferBuilder(logger()));
    if (!builder) return {};

    // TensorRT 10+ removed NetworkDefinitionCreationFlag::kEXPLICIT_BATCH --
    // createNetworkV2() supports only explicit batch + strongly-typed
    // networks now (see its own doc comment in NvInfer.h), so no flags are
    // needed at all.
    Ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) return {};

    DataType dtype = is_fp16 ? DataType::kHALF : DataType::kFLOAT;

    Dims4 qkv_dims{B, H, S, D};
    ITensor* q = network->addInput("q", dtype, qkv_dims);
    ITensor* k = network->addInput("k", dtype, qkv_dims);
    ITensor* v = network->addInput("v", dtype, qkv_dims);
    if (!q || !k || !v) return {};

    // scores = Q @ K^T -- addMatrixMultiply batches over the leading dims
    // (B,H here) and matrix-multiplies the trailing two, same semantics as
    // torch.matmul/numpy. kTRANSPOSE on the K operand transposes only its
    // own trailing two dims (S,D) -> (D,S), giving [B,H,S,S].
    IMatrixMultiplyLayer* qk = network->addMatrixMultiply(
        *q, MatrixOperation::kNONE, *k, MatrixOperation::kTRANSPOSE);
    if (!qk) return {};
    ITensor* scores = qk->getOutput(0);

    // scale: broadcast-multiply by a single scalar constant. A
    // {1,1,1,1}-shaped constant broadcasts against [B,H,S,S] under
    // TensorRT's numpy-style elementwise broadcasting rules.
    float scale_value = scale;
    std::vector<char> scale_bytes(sizeof(float));
    std::memcpy(scale_bytes.data(), &scale_value, sizeof(float));
    // Weights must stay alive until buildSerializedNetwork below (TensorRT
    // copies constant data internally at that point, not before), which
    // scale_bytes/scale_host (fp16 branch) already satisfy as locals of
    // this function.
    Weights scale_w{};
    std::vector<char> scale_host_fp16;
    if (is_fp16) {
        // Build the fp16 bit pattern for `scale` on the host without any
        // CUDA/half-intrinsic dependency in this file: reuse the identical
        // float->half conversion this project's own hand-tuned kernels
        // rely on being correct (IEEE754 half, round-to-nearest-even) via
        // a tiny local implementation, since pulling in <cuda_fp16.h>'s
        // __float2half here would need this file compiled by nvcc instead
        // of a plain C++ compiler.
        auto float_to_half_bits = [](float f) -> uint16_t {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            uint32_t sign = (bits >> 16) & 0x8000u;
            int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = bits & 0x7FFFFFu;
            if (exp <= 0) return static_cast<uint16_t>(sign);           // underflow to 0 -- fine for a scale factor in practice
            if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u); // overflow to inf
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
        };
        uint16_t half_bits = float_to_half_bits(scale_value);
        scale_host_fp16.resize(sizeof(uint16_t));
        std::memcpy(scale_host_fp16.data(), &half_bits, sizeof(uint16_t));
        scale_w = Weights{DataType::kHALF, scale_host_fp16.data(), 1};
    } else {
        scale_w = Weights{DataType::kFLOAT, scale_bytes.data(), 1};
    }
    Dims4 scale_dims{1, 1, 1, 1};
    IConstantLayer* scale_const = network->addConstant(scale_dims, scale_w);
    if (!scale_const) return {};

    IElementWiseLayer* scaled = network->addElementWise(
        *scores, *scale_const->getOutput(0), ElementWiseOperation::kPROD);
    if (!scaled) return {};

    // Softmax over the last dim (the key/sequence dim of the [B,H,S,S]
    // score matrix) -- setAxes takes a bitmask, bit 3 selects dim index 3
    // of this 4D tensor.
    ISoftMaxLayer* softmax = network->addSoftMax(*scaled->getOutput(0));
    if (!softmax) return {};
    softmax->setAxes(1U << 3);

    // output = softmax(scores) @ V -> [B,H,S,D]
    IMatrixMultiplyLayer* out_mm = network->addMatrixMultiply(
        *softmax->getOutput(0), MatrixOperation::kNONE, *v, MatrixOperation::kNONE);
    if (!out_mm) return {};

    ITensor* output_tensor = out_mm->getOutput(0);
    output_tensor->setName("output");
    network->markOutput(*output_tensor);

    Ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return {};
    // TensorRT 10+ removed BuilderFlag::kFP16 (and every other precision
    // hint flag) -- networks are always strongly typed now, so precision
    // is inferred purely from the DataType already set above on the input
    // tensors, nothing left to opt into here.
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

    Ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) return {};

    return std::vector<char>(
        static_cast<const char*>(serialized->data()),
        static_cast<const char*>(serialized->data()) + serialized->size());
}

}  // namespace

bool cmpext3_trt_attention_forward(
    const void* q, const void* k, const void* v, void* output,
    int B, int H, int S, int D,
    float scale,
    bool is_fp16,
    cudaStream_t stream)
{
    // Pure shape+scale+dtype cache key -- Q/K/V are runtime bindings, not
    // baked-in weights, so (unlike conv/matmul) their content never needs
    // hashing or a device->host copy here.
    struct ShapeKey {
        int B, H, S, D;
        float scale;
        bool is_fp16;
    } shape_key;
    // Zero the WHOLE struct first, padding included -- see
    // conv3d_trt_runtime.cpp's identical fix for why: aggregate
    // list-initialization leaves padding bytes as uninitialized stack
    // garbage, which would otherwise make this hash non-reproducible for
    // the identical shape.
    std::memset(&shape_key, 0, sizeof(shape_key));
    shape_key.B = B; shape_key.H = H; shape_key.S = S; shape_key.D = D;
    shape_key.scale = scale; shape_key.is_fp16 = is_fp16;

    uint64_t h = fnv1a_update(&shape_key, sizeof(shape_key), FNV_OFFSET_BASIS);
    std::string key = "attention_" + hash_to_hex(h);

    auto entry = get_or_build_engine(key, [&]() {
        return build_engine(B, H, S, D, scale, is_fp16);
    });
    if (!entry) return false;

    if (!entry->context->setTensorAddress("q", const_cast<void*>(q)) ||
        !entry->context->setTensorAddress("k", const_cast<void*>(k)) ||
        !entry->context->setTensorAddress("v", const_cast<void*>(v)) ||
        !entry->context->setTensorAddress("output", output)) {
        return false;
    }
    return entry->context->enqueueV3(stream);
}
