// =================================================================================
// conv2d_trt_runtime.cpp -- see conv3d_trt_runtime.cpp (this is the exact
// same structure, one dimension fewer) and trt_common.h (UNVERIFIED --
// read its design-rationale comment before trusting this file).
// =================================================================================

#include "conv2d_trt_runtime.h"
#include "trt_common.h"

#include <cstring>

using namespace nvinfer1;
using namespace cmpext3_trt;

namespace {

std::vector<char> build_engine(
    const std::vector<char>& weight_host, const std::vector<char>& bias_host,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w,
    bool is_fp16, bool has_bias)
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

    Dims4 input_dims{};
    input_dims.nbDims = 4;
    input_dims.d[0] = B; input_dims.d[1] = C_in; input_dims.d[2] = H_in; input_dims.d[3] = W_in;
    ITensor* input_tensor = network->addInput("input", dtype, input_dims);
    if (!input_tensor) return {};

    Weights weight_w{dtype, weight_host.data(), static_cast<int64_t>(weight_host.size() / elem_size)};
    Weights bias_w{dtype, nullptr, 0};
    if (has_bias) {
        bias_w = Weights{dtype, bias_host.data(), static_cast<int64_t>(bias_host.size() / elem_size)};
    }

    Dims2 kernel_size{K_H, K_W};
    IConvolutionLayer* conv = network->addConvolutionNd(*input_tensor, C_out, kernel_size, weight_w, bias_w);
    if (!conv) return {};
    conv->setStrideNd(Dims2{s_h, s_w});
    conv->setPaddingNd(Dims2{p_h, p_w});
    conv->setDilationNd(Dims2{d_h, d_w});

    ITensor* output_tensor = conv->getOutput(0);
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

bool cmpext3_trt_conv2d_forward(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w,
    int p_h, int p_w,
    int d_h, int d_w,
    bool is_fp16,
    bool has_bias,
    cudaStream_t stream)
{
    const size_t elem_size = is_fp16 ? sizeof(uint16_t) : sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(C_out) * C_in * K_H * K_W * elem_size;
    const size_t bias_bytes = has_bias ? static_cast<size_t>(C_out) * elem_size : 0;

    std::vector<char> weight_host(weight_bytes);
    if (!copy_to_host(weight, weight_host.data(), weight_bytes, stream)) return false;
    std::vector<char> bias_host;
    if (has_bias) {
        bias_host.resize(bias_bytes);
        if (!copy_to_host(bias, bias_host.data(), bias_bytes, stream)) return false;
    }

    struct ShapeKey {
        int B, C_in, H_in, W_in, C_out, K_H, K_W;
        int s_h, s_w, p_h, p_w, d_h, d_w;
        bool is_fp16, has_bias;
    } shape_key;
    // Zero the WHOLE struct first, padding included -- see
    // conv3d_trt_runtime.cpp's identical fix for why: aggregate
    // list-initialization leaves padding bytes as uninitialized stack
    // garbage, which would otherwise make this hash non-reproducible for
    // the identical shape.
    std::memset(&shape_key, 0, sizeof(shape_key));
    shape_key.B = B; shape_key.C_in = C_in; shape_key.H_in = H_in; shape_key.W_in = W_in;
    shape_key.C_out = C_out; shape_key.K_H = K_H; shape_key.K_W = K_W;
    shape_key.s_h = s_h; shape_key.s_w = s_w; shape_key.p_h = p_h; shape_key.p_w = p_w;
    shape_key.d_h = d_h; shape_key.d_w = d_w;
    shape_key.is_fp16 = is_fp16; shape_key.has_bias = has_bias;

    uint64_t h = fnv1a_update(&shape_key, sizeof(shape_key), FNV_OFFSET_BASIS);
    h = fnv1a_update(weight_host.data(), weight_host.size(), h);
    if (has_bias) h = fnv1a_update(bias_host.data(), bias_host.size(), h);
    std::string key = "conv2d_" + hash_to_hex(h);

    auto entry = get_or_build_engine(key, [&]() {
        return build_engine(weight_host, bias_host,
            B, C_in, H_in, W_in, C_out, K_H, K_W,
            s_h, s_w, p_h, p_w, d_h, d_w, is_fp16, has_bias);
    });
    if (!entry) return false;

    if (!entry->context->setTensorAddress("input", const_cast<void*>(input)) ||
        !entry->context->setTensorAddress("output", output)) {
        return false;
    }
    return entry->context->enqueueV3(stream);
}
