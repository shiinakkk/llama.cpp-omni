#include "marlin-op.cuh"

#include "ggml-cuda.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

static bool ggml_cuda_marlin_debug_enabled() {
    const char * value = std::getenv("LLAMA_AWQ_MARLIN_DEBUG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool ggml_cuda_marlin_disable_layer2_down_zp() {
    const char * value = std::getenv("LLAMA_AWQ_MARLIN_DISABLE_LAYER2_DOWN_ZP");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool ggml_cuda_marlin_debug_match_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }

    return std::strstr(name, "layers.2.mlp.gate_proj") != nullptr ||
           std::strstr(name, "layers.2.mlp.up_proj") != nullptr ||
           std::strstr(name, "layers.2.mlp.down_proj") != nullptr;
}

template <typename T>
static void ggml_cuda_marlin_debug_copy_prefix(
        std::vector<T> & host,
        const void * device_ptr,
        cudaStream_t stream) {
    if (host.empty() || device_ptr == nullptr) {
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(host.data(), device_ptr, host.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void ggml_cuda_marlin_debug_log_tensor_prefix(
        const char * stage,
        const ggml_tensor * tensor,
        cudaStream_t stream) {
    if (tensor == nullptr) {
        return;
    }

    const int64_t count = std::min<int64_t>(ggml_nelements(tensor), 8);
    const char * name = ggml_get_name(tensor);

    std::ostringstream oss;
    oss << "awq-marlin-runtime: stage=" << stage
        << " name=" << (name ? name : "<unnamed>")
        << " type=" << ggml_type_name(tensor->type)
        << " ne=[" << tensor->ne[0] << "," << tensor->ne[1] << "," << tensor->ne[2] << "," << tensor->ne[3] << "]"
        << " nb=[" << tensor->nb[0] << "," << tensor->nb[1] << "," << tensor->nb[2] << "," << tensor->nb[3] << "]";

    if (count <= 0 || tensor->data == nullptr) {
        GGML_LOG_INFO("%s\n", oss.str().c_str());
        return;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        oss << " first=<skipped during cuda graph capture>";
        GGML_LOG_INFO("%s\n", oss.str().c_str());
        return;
    }

    oss << " first=";
    switch (tensor->type) {
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> host(count);
            ggml_cuda_marlin_debug_copy_prefix(host, tensor->data, stream);
            for (int64_t i = 0; i < count; ++i) {
                if (i) oss << ",";
                oss << ggml_fp16_to_fp32(host[i]);
            }
        } break;
        case GGML_TYPE_F32: {
            std::vector<float> host(count);
            ggml_cuda_marlin_debug_copy_prefix(host, tensor->data, stream);
            for (int64_t i = 0; i < count; ++i) {
                if (i) oss << ",";
                oss << host[i];
            }
        } break;
        case GGML_TYPE_I32: {
            std::vector<int32_t> host(count);
            ggml_cuda_marlin_debug_copy_prefix(host, tensor->data, stream);
            for (int64_t i = 0; i < count; ++i) {
                if (i) oss << ",";
                oss << host[i];
            }
        } break;
        default:
            oss << "<unsupported>";
            break;
    }

    GGML_LOG_INFO("%s\n", oss.str().c_str());
}

void ggml_cuda_op_marlin_w4a16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a         = dst->src[0];
    const ggml_tensor * qweight   = dst->src[1];
    const ggml_tensor * scales    = dst->src[2];
    const ggml_tensor * qzeros    = dst->src[3];
    const ggml_tensor * workspace_src = dst->src[4];

    GGML_ASSERT(a != nullptr);
    GGML_ASSERT(qweight != nullptr);
    GGML_ASSERT(scales != nullptr);
    GGML_ASSERT(qzeros != nullptr);
    GGML_ASSERT(workspace_src != nullptr);

    const char * qweight_name = ggml_get_name(qweight);
    const bool want_debug = ggml_cuda_marlin_debug_enabled() && ggml_cuda_marlin_debug_match_name(qweight_name);
    const bool disable_down_zp = ggml_cuda_marlin_disable_layer2_down_zp() &&
            qweight_name != nullptr &&
            std::strstr(qweight_name, "layers.2.mlp.down_proj") != nullptr;
    const ggml_tensor * effective_qzeros = disable_down_zp ? nullptr : qzeros;

    if (want_debug) {
        std::ostringstream oss;
        oss << "awq-marlin-runtime: launch name=" << (qweight_name ? qweight_name : "<unnamed>")
            << " m=" << (ggml_nelements(a) / a->ne[0])
            << " k=" << a->ne[0]
            << " n=" << scales->ne[1]
            << " workspace_elements=" << ggml_cuda_marlin_min_workspace_elements(ctx.device)
            << " use_zp=" << (disable_down_zp ? 0 : 1);
        GGML_LOG_INFO("%s\n", oss.str().c_str());

        ggml_cuda_marlin_debug_log_tensor_prefix("input.pre", a, ctx.stream());
        ggml_cuda_marlin_debug_log_tensor_prefix("qweight.pre", qweight, ctx.stream());
        if (effective_qzeros != nullptr) {
            ggml_cuda_marlin_debug_log_tensor_prefix("qzeros.pre", effective_qzeros, ctx.stream());
        }
        ggml_cuda_marlin_debug_log_tensor_prefix("scales.pre", scales, ctx.stream());
    }

    const int workspace_elements = ggml_cuda_marlin_min_workspace_elements(ctx.device);
    ggml_cuda_pool_alloc<int> workspace_alloc(ctx.pool(), workspace_elements);

    ggml_tensor workspace = *workspace_src;
    workspace.ne[0] = workspace_elements;
    workspace.ne[1] = 1;
    workspace.ne[2] = 1;
    workspace.ne[3] = 1;
    workspace.nb[0] = sizeof(int);
    workspace.nb[1] = workspace.nb[0] * workspace.ne[0];
    workspace.nb[2] = workspace.nb[1];
    workspace.nb[3] = workspace.nb[2];
    workspace.buffer = nullptr;
    workspace.data = workspace_alloc.get();

    const bool ok = ggml_cuda_marlin_w4a16_gemm(
            a,
            qweight,
            scales,
            effective_qzeros,
            dst,
            &workspace,
            ctx.device,
            ctx.stream());

    GGML_ASSERT(ok);

    if (want_debug) {
        ggml_cuda_marlin_debug_log_tensor_prefix("output.post", dst, ctx.stream());
    }
}
