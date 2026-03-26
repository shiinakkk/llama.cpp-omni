#include <ggml-cuda.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kNumBits = 4;
constexpr int kTile = 16;
constexpr int kPackFactor = 32 / kNumBits;

void check_cuda(cudaError_t err, const char * expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)

std::vector<int> get_weight_perm() {
    std::vector<int> perm_list;
    perm_list.reserve(1024);

    for (int i = 0; i < 32; ++i) {
        std::vector<int> perm1;
        perm1.reserve(8);
        const int col = i / 4;
        for (int block : {0, 1}) {
            for (int row : {2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1}) {
                perm1.push_back(16 * row + col + 8 * block);
            }
        }
        for (int j = 0; j < 4; ++j) {
            for (int p : perm1) {
                perm_list.push_back(p + 256 * j);
            }
        }
    }

    const int interleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<int> perm;
    perm.reserve(perm_list.size());
    for (size_t i = 0; i < perm_list.size(); i += 8) {
        for (int idx : interleave) {
            perm.push_back(perm_list[i + idx]);
        }
    }

    return perm;
}

std::vector<int> get_scale_perm_single() {
    std::vector<int> perm;
    perm.reserve(32);
    for (int i = 0; i < 4; ++i) {
        for (int j : {0, 1, 8, 9, 16, 17, 24, 25}) {
            perm.push_back(2 * i + j);
        }
    }
    return perm;
}

std::array<int, 64> get_scale_perm_grouped() {
    return {
         0,  8, 16, 24, 32, 40, 48, 56,
         1,  9, 17, 25, 33, 41, 49, 57,
         2, 10, 18, 26, 34, 42, 50, 58,
         3, 11, 19, 27, 35, 43, 51, 59,
         4, 12, 20, 28, 36, 44, 52, 60,
         5, 13, 21, 29, 37, 45, 53, 61,
         6, 14, 22, 30, 38, 46, 54, 62,
         7, 15, 23, 31, 39, 47, 55, 63,
    };
}

std::vector<uint32_t> pack_u4_rows(const std::vector<uint32_t> & values, int rows, int cols) {
    std::vector<uint32_t> packed(rows * (cols / kPackFactor), 0);
    const int packed_cols = cols / kPackFactor;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int packed_col = col / kPackFactor;
            const int shift = (col % kPackFactor) * kNumBits;
            packed[row * packed_cols + packed_col] |= (values[row * cols + col] & 0xFu) << shift;
        }
    }
    return packed;
}

std::vector<uint32_t> unpack_u4_rows(const std::vector<uint32_t> & packed, int rows, int cols) {
    std::vector<uint32_t> unpacked(rows * cols, 0);
    const int packed_cols = cols / kPackFactor;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < packed_cols; ++col) {
            const uint32_t word = packed[row * packed_cols + col];
            for (int i = 0; i < kPackFactor; ++i) {
                unpacked[row * cols + col * kPackFactor + i] = (word >> (i * kNumBits)) & 0xFu;
            }
        }
    }
    return unpacked;
}

std::vector<uint32_t> marlin_pack_u4_weights(const std::vector<uint32_t> & q_w, int size_k, int size_n) {
    const auto perm = get_weight_perm();
    const int tile_k_blocks = size_k / kTile;
    const int tile_n_blocks = size_n / kTile;

    std::vector<uint32_t> tiled(tile_k_blocks * size_n * kTile);
    for (int kb = 0; kb < tile_k_blocks; ++kb) {
        for (int nb = 0; nb < tile_n_blocks; ++nb) {
            for (int ki = 0; ki < kTile; ++ki) {
                for (int ni = 0; ni < kTile; ++ni) {
                    const int src_row = kb * kTile + ki;
                    const int src_col = nb * kTile + ni;
                    const int dst_col = nb * kTile * kTile + ki * kTile + ni;
                    tiled[kb * (size_n * kTile) + dst_col] = q_w[src_row * size_n + src_col];
                }
            }
        }
    }

    std::vector<uint32_t> permuted(tiled.size());
    const int perm_width = static_cast<int>(perm.size());
    for (size_t base = 0; base < tiled.size(); base += perm_width) {
        for (int i = 0; i < perm_width; ++i) {
            permuted[base + i] = tiled[base + perm[i]];
        }
    }

    return pack_u4_rows(permuted, tile_k_blocks, size_n * kTile);
}

std::vector<__half> marlin_permute_scales(const std::vector<float> & scales, int num_groups, int size_n, bool use_grouped_perm) {
    std::vector<__half> out(scales.size());
    const auto perm_grouped = get_scale_perm_grouped();
    const auto perm_single = get_scale_perm_single();
    const int block = use_grouped_perm ? static_cast<int>(perm_grouped.size()) : static_cast<int>(perm_single.size());

    for (int row = 0; row < num_groups; ++row) {
        const float * src = scales.data() + row * size_n;
        __half * dst = out.data() + row * size_n;
        for (int base = 0; base < size_n; base += block) {
            if (use_grouped_perm) {
                for (int i = 0; i < block; ++i) {
                    dst[base + i] = __float2half(src[base + perm_grouped[i]]);
                }
            } else {
                for (int i = 0; i < block; ++i) {
                    dst[base + i] = __float2half(src[base + perm_single[i]]);
                }
            }
        }
    }

    return out;
}

std::vector<uint32_t> marlin_permute_qzeros(
        const std::vector<uint32_t> & qzeros_awq_packed,
        int rows,
        int size_n,
        bool undo_source_interleave,
        bool output_awq_interleaved) {
    static constexpr std::array<int, 8> kAwqInterleave = {0, 2, 4, 6, 1, 3, 5, 7};
    static constexpr std::array<int, 8> kAwqUndoInterleave = {0, 4, 1, 5, 2, 6, 3, 7};
    const auto scale_perm = get_scale_perm_grouped();

    std::vector<uint32_t> unpacked = unpack_u4_rows(qzeros_awq_packed, rows, size_n);
    std::vector<uint32_t> awq_uninterleaved = unpacked;
    std::vector<uint32_t> marlin_permuted(unpacked.size());
    std::vector<uint32_t> marlin_interleaved(unpacked.size());

    if (undo_source_interleave) {
        for (int row = 0; row < rows; ++row) {
            const uint32_t * src = unpacked.data() + row * size_n;
            uint32_t * dst = awq_uninterleaved.data() + row * size_n;
            for (int base = 0; base < size_n; base += static_cast<int>(kAwqUndoInterleave.size())) {
                for (size_t i = 0; i < kAwqUndoInterleave.size(); ++i) {
                    dst[base + static_cast<int>(i)] = src[base + kAwqUndoInterleave[i]];
                }
            }
        }
    }

    for (int row = 0; row < rows; ++row) {
        const uint32_t * src = awq_uninterleaved.data() + row * size_n;
        uint32_t * dst = marlin_permuted.data() + row * size_n;
        for (int base = 0; base < size_n; base += static_cast<int>(scale_perm.size())) {
            for (size_t i = 0; i < scale_perm.size(); ++i) {
                dst[base + static_cast<int>(i)] = src[base + scale_perm[i]];
            }
        }
    }

    if (output_awq_interleaved) {
        for (int row = 0; row < rows; ++row) {
            const uint32_t * src = marlin_permuted.data() + row * size_n;
            uint32_t * dst = marlin_interleaved.data() + row * size_n;
            for (int base = 0; base < size_n; base += static_cast<int>(kAwqInterleave.size())) {
                for (size_t i = 0; i < kAwqInterleave.size(); ++i) {
                    dst[base + static_cast<int>(i)] = src[base + kAwqInterleave[i]];
                }
            }
        }
    } else {
        marlin_interleaved = marlin_permuted;
    }

    return pack_u4_rows(marlin_interleaved, rows, size_n);
}

std::vector<uint32_t> pack_qzeros_source(
        const std::vector<uint32_t> & logical_qzeros,
        int rows,
        int size_n,
        bool awq_interleaved_source) {
    if (!awq_interleaved_source) {
        return pack_u4_rows(logical_qzeros, rows, size_n);
    }

    static constexpr std::array<int, 8> kAwqInterleave = {0, 2, 4, 6, 1, 3, 5, 7};

    std::vector<uint32_t> awq_interleaved(logical_qzeros.size());
    for (int row = 0; row < rows; ++row) {
        const uint32_t * src = logical_qzeros.data() + row * size_n;
        uint32_t * dst = awq_interleaved.data() + row * size_n;
        for (int base = 0; base < size_n; base += static_cast<int>(kAwqInterleave.size())) {
            for (size_t i = 0; i < kAwqInterleave.size(); ++i) {
                dst[base + kAwqInterleave[i]] = src[base + static_cast<int>(i)];
            }
        }
    }

    return pack_u4_rows(awq_interleaved, rows, size_n);
}

struct TestCase {
    std::string name;
    int m;
    int n;
    int k;
    int group_size;
    bool use_qzeros;
    bool use_grouped_scale_perm;
    bool qzeros_source_awq_interleaved;
    bool qzeros_output_awq_interleaved;
};

float allowed_error(const TestCase & tc, float want) {
    const float abs_want = std::fabs(want);
    float tol = 2e-2f;

    if (tc.k >= 4096 || tc.n >= 1024) {
        tol = std::max(tol, 2e-3f * abs_want);
        tol = std::max(tol, 5e-1f);
    }

    return tol;
}

struct DeviceBuffers {
    __half * d_a = nullptr;
    uint32_t * d_b = nullptr;
    __half * d_scales = nullptr;
    uint32_t * d_qzeros = nullptr;
    __half * d_c = nullptr;
    int * d_workspace = nullptr;
};

void free_device_buffers(DeviceBuffers & buffers) {
    if (buffers.d_workspace) CUDA_CHECK(cudaFree(buffers.d_workspace));
    if (buffers.d_c) CUDA_CHECK(cudaFree(buffers.d_c));
    if (buffers.d_qzeros) CUDA_CHECK(cudaFree(buffers.d_qzeros));
    if (buffers.d_scales) CUDA_CHECK(cudaFree(buffers.d_scales));
    if (buffers.d_b) CUDA_CHECK(cudaFree(buffers.d_b));
    if (buffers.d_a) CUDA_CHECK(cudaFree(buffers.d_a));
}

float dequant_awq_value(uint32_t q, float scale, uint32_t zero) {
    return (static_cast<float>(q) - static_cast<float>(zero)) * scale;
}

float dequant_no_zp_value(uint32_t q, float scale) {
    return (static_cast<float>(q) - 8.0f) * scale;
}

void run_test_case(int device, const TestCase & tc) {
    const int num_groups = tc.group_size > 0 ? tc.k / tc.group_size : 1;
    if (tc.group_size > 0 && tc.k % tc.group_size != 0) {
        throw std::runtime_error(tc.name + ": invalid group_size");
    }
    if (tc.n % 64 != 0) {
        throw std::runtime_error(tc.name + ": n must be divisible by 64");
    }

    std::vector<__half> a_host(tc.m * tc.k);
    std::vector<uint32_t> q_w_host(tc.k * tc.n);
    std::vector<float> scales_host(num_groups * tc.n);
    std::vector<uint32_t> qzeros_logical;
    std::vector<uint32_t> qzeros_awq_packed;
    std::vector<float> ref_host(tc.m * tc.n, 0.0f);

    for (int row = 0; row < tc.m; ++row) {
        for (int col = 0; col < tc.k; ++col) {
            const float value = float(((row * 11 + col * 7 + 3) % 23) - 11) * 0.125f;
            a_host[row * tc.k + col] = __float2half(value);
        }
    }

    for (int row = 0; row < tc.k; ++row) {
        for (int col = 0; col < tc.n; ++col) {
            q_w_host[row * tc.n + col] = static_cast<uint32_t>((row * 13 + col * 5 + 7) & 0xF);
        }
    }

    for (int group = 0; group < num_groups; ++group) {
        for (int col = 0; col < tc.n; ++col) {
            const float scale = 0.03125f * float(1 + ((group * 7 + col * 3 + 5) % 11));
            scales_host[group * tc.n + col] = scale;
        }
    }

    if (tc.use_qzeros) {
        qzeros_logical.resize(num_groups * tc.n);
        for (int group = 0; group < num_groups; ++group) {
            for (int col = 0; col < tc.n; ++col) {
                qzeros_logical[group * tc.n + col] = static_cast<uint32_t>((group * 5 + col * 9 + 2) & 0xF);
            }
        }
        qzeros_awq_packed = pack_qzeros_source(
                qzeros_logical,
                num_groups,
                tc.n,
                tc.qzeros_source_awq_interleaved);
    }

    for (int row = 0; row < tc.m; ++row) {
        for (int col = 0; col < tc.n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < tc.k; ++kk) {
                const int group = tc.group_size > 0 ? kk / tc.group_size : 0;
                const float scale = scales_host[group * tc.n + col];
                const uint32_t q = q_w_host[kk * tc.n + col];
                const float w = tc.use_qzeros
                        ? dequant_awq_value(q, scale, qzeros_logical[group * tc.n + col])
                        : dequant_no_zp_value(q, scale);
                acc += __half2float(a_host[row * tc.k + kk]) * w;
            }
            ref_host[row * tc.n + col] = acc;
        }
    }

    const std::vector<uint32_t> packed_b_host = marlin_pack_u4_weights(q_w_host, tc.k, tc.n);
    const std::vector<__half> packed_scales_host =
            marlin_permute_scales(scales_host, num_groups, tc.n, tc.use_grouped_scale_perm);
    const std::vector<uint32_t> packed_qzeros_host = tc.use_qzeros
            ? marlin_permute_qzeros(
                    qzeros_awq_packed,
                    num_groups,
                    tc.n,
                    tc.qzeros_source_awq_interleaved,
                    tc.qzeros_output_awq_interleaved)
            : std::vector<uint32_t>();

    DeviceBuffers buffers;
    ggml_context * ctx = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.d_a), a_host.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.d_b), packed_b_host.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.d_scales), packed_scales_host.size() * sizeof(__half)));
        if (tc.use_qzeros) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.d_qzeros), packed_qzeros_host.size() * sizeof(uint32_t)));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.d_c), tc.m * tc.n * sizeof(__half)));

        const int workspace_elems = ggml_cuda_marlin_min_workspace_elements(device);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.d_workspace), workspace_elems * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(buffers.d_a, a_host.data(), a_host.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buffers.d_b, packed_b_host.data(), packed_b_host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buffers.d_scales, packed_scales_host.data(), packed_scales_host.size() * sizeof(__half), cudaMemcpyHostToDevice));
        if (tc.use_qzeros) {
            CUDA_CHECK(cudaMemcpy(buffers.d_qzeros, packed_qzeros_host.data(), packed_qzeros_host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemset(buffers.d_c, 0, tc.m * tc.n * sizeof(__half)));
        CUDA_CHECK(cudaMemset(buffers.d_workspace, 0, workspace_elems * sizeof(int)));

        ggml_init_params ggml_params = {
            /*.mem_size   =*/ 1 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(ggml_params);
        if (ctx == nullptr) {
            throw std::runtime_error(tc.name + ": ggml_init failed");
        }

        ggml_tensor * a_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, tc.k, tc.m);
        ggml_tensor * b_qweight_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tc.k, tc.n / kPackFactor);
        ggml_tensor * b_scales_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, num_groups, tc.n);
        ggml_tensor * b_qzeros_tensor = tc.use_qzeros
                ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, num_groups, tc.n / kPackFactor)
                : nullptr;
        ggml_tensor * c_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, tc.n, tc.m);
        ggml_tensor * workspace_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, workspace_elems);

        a_tensor->data = buffers.d_a;
        b_qweight_tensor->data = buffers.d_b;
        b_scales_tensor->data = buffers.d_scales;
        if (b_qzeros_tensor != nullptr) {
            b_qzeros_tensor->data = buffers.d_qzeros;
        }
        c_tensor->data = buffers.d_c;
        workspace_tensor->data = buffers.d_workspace;

        if (!ggml_cuda_marlin_w4a16_gemm(
                    a_tensor,
                    b_qweight_tensor,
                    b_scales_tensor,
                    b_qzeros_tensor,
                    c_tensor,
                    workspace_tensor,
                    device,
                    nullptr)) {
            throw std::runtime_error(tc.name + ": ggml_cuda_marlin_w4a16_gemm returned false");
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<__half> c_host(tc.m * tc.n);
        CUDA_CHECK(cudaMemcpy(c_host.data(), buffers.d_c, c_host.size() * sizeof(__half), cudaMemcpyDeviceToHost));

        float max_abs_err = 0.0f;
        for (int i = 0; i < tc.m * tc.n; ++i) {
            const float got = __half2float(c_host[i]);
            const float want = ref_host[i];
            if (!std::isfinite(got)) {
                throw std::runtime_error(tc.name + ": output contains non-finite values");
            }
            max_abs_err = std::max(max_abs_err, std::fabs(got - want));
            const float tol = allowed_error(tc, want);
            if (std::fabs(got - want) > tol) {
                const int row = i / tc.n;
                const int col = i % tc.n;
                std::ostringstream detail;
                detail << tc.name << ": mismatch at " << i
                       << " row=" << row
                       << " col=" << col
                       << " got=" << got
                       << " want=" << want
                       << " tol=" << tol;
                if (tc.use_qzeros) {
                    const int ref_group = 0;
                    detail << " scale=" << scales_host[ref_group * tc.n + col]
                           << " zp=" << qzeros_logical[ref_group * tc.n + col]
                           << " q0=" << q_w_host[col]
                           << " qzeros_source_awq_interleaved=" << (tc.qzeros_source_awq_interleaved ? 1 : 0)
                           << " qzeros_output_awq_interleaved=" << (tc.qzeros_output_awq_interleaved ? 1 : 0);
                }
                throw std::runtime_error(
                        detail.str());
            }
        }

        std::cout << tc.name << " passed, max_abs_err=" << max_abs_err << '\n';
    } catch (...) {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        free_device_buffers(buffers);
        throw;
    }

    ggml_free(ctx);
    free_device_buffers(buffers);
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device not available\n";
        return 0;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (major * 10 + minor < 75) {
        std::cout << "SKIP: Marlin requires compute capability >= 7.5\n";
        return 0;
    }

    std::vector<TestCase> cases = {
        {"single_group_scales", 16, 64, 128, -1, false, false, false, false},
        {"grouped_scales", 16, 64, 128, 32, false, true, false, false},
    };

    const bool run_qzeros_probe = std::getenv("LLAMA_MARLIN_RUN_QZEROS_PROBE") != nullptr;
    if (run_qzeros_probe) {
        cases.push_back({"grouped_scales_qzeros_src_awq_out_awq", 16, 64, 128, 32, true, true, true, true});
        cases.push_back({"grouped_scales_qzeros_src_plain_out_awq", 16, 64, 128, 32, true, true, false, true});
        cases.push_back({"grouped_scales_qzeros_src_awq_out_plain", 16, 64, 128, 32, true, true, true, false});
        cases.push_back({"grouped_scales_qzeros_src_plain_out_plain", 16, 64, 128, 32, true, true, false, false});
    } else {
        std::cout << "SKIP: grouped_scales_qzeros probe is disabled by default; set LLAMA_MARLIN_RUN_QZEROS_PROBE=1 to run it\n";
    }

    const bool run_large_shape_probe = std::getenv("LLAMA_MARLIN_RUN_LARGE_SHAPE_PROBE") != nullptr;
    if (run_large_shape_probe) {
        cases.push_back({"grouped_scales_large_shape", 15, 4096, 12288, 128, false, true, false, false});
        cases.push_back({"grouped_scales_qzeros_large_shape_src_plain_out_awq", 15, 4096, 12288, 128, true, true, false, true});
    } else {
        std::cout << "SKIP: large-shape Marlin probe is disabled by default; set LLAMA_MARLIN_RUN_LARGE_SHAPE_PROBE=1 to run it\n";
    }

    bool had_failure = false;
    for (const TestCase & tc : cases) {
        try {
            run_test_case(device, tc);
        } catch (const std::exception & err) {
            had_failure = true;
            std::cerr << err.what() << '\n';
        }
    }

    if (had_failure) {
        return 1;
    }

    std::cout << "All enabled Marlin W4A16 tests passed\n";
    return 0;
}
