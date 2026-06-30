#include "trt_vocoder.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdlib>

namespace omni {
namespace vocoder {

// ====== iSTFT constants (matching hg2_stft16_params) ======
static constexpr int N_FFT = 16;
static constexpr int HOP   = 4;
static constexpr int F_    = 9;
static constexpr int PAD   = 8;
static constexpr int SAMPLES_PER_MEL = 480;
static constexpr float AUDIO_LIMIT = 0.99f;

// ====== Logger ======
namespace {
    struct HostStats {
        size_t n       = 0;
        size_t finite  = 0;
        size_t nan     = 0;
        size_t pos_inf = 0;
        size_t neg_inf = 0;
        size_t zero    = 0;
        float  min     = std::numeric_limits<float>::infinity();
        float  max     = -std::numeric_limits<float>::infinity();
        double mean_abs = 0.0;
    };

    static HostStats calc_stats(const float * data, size_t n) {
        HostStats s{};
        s.n = n;
        if (!data || n == 0) {
            s.min = 0.0f;
            s.max = 0.0f;
            return s;
        }
        double sum_abs = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const float v = data[i];
            if (std::isnan(v)) {
                s.nan++;
                continue;
            }
            if (std::isinf(v)) {
                if (v > 0.0f) {
                    s.pos_inf++;
                } else {
                    s.neg_inf++;
                }
                continue;
            }
            s.finite++;
            if (v == 0.0f) {
                s.zero++;
            }
            s.min = std::min(s.min, v);
            s.max = std::max(s.max, v);
            sum_abs += std::fabs(v);
        }
        if (s.finite == 0) {
            s.min = 0.0f;
            s.max = 0.0f;
        } else {
            s.mean_abs = sum_abs / (double) s.finite;
        }
        return s;
    }

    static void print_stats(const char * label, const float * data, size_t n) {
        const HostStats s = calc_stats(data, n);
        std::fprintf(stderr,
                     "[TRT-DBG] %s n=%zu finite=%zu nan=%zu +inf=%zu -inf=%zu zero=%zu min=%.6g max=%.6g mean_abs=%.6g\n",
                     label, s.n, s.finite, s.nan, s.pos_inf, s.neg_inf, s.zero, s.min, s.max, s.mean_abs);
    }

    static bool dump_f32_raw(const char * dir, const char * tag, int idx, int T, int C, const std::vector<float> & data) {
        if (!dir || !*dir || !tag || !*tag) {
            return false;
        }
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/%s_%03d_T%d_C%d.f32", dir, tag, idx, T, C);
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            std::fprintf(stderr, "OMNI_VOC_DUMP: failed to open %s\n", path);
            return false;
        }
        out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
        std::fprintf(stderr, "OMNI_VOC_DUMP: wrote %s (%zu floats)\n", path, data.size());
        return true;
    }

    class TrtLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char * msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::fprintf(stderr, "[TRT] %s\n", msg);
        }
    };
    static TrtLogger g_trt_logger;
}

// ====== impl ======

TRTVocoder::TRTVocoder() = default;

TRTVocoder::~TRTVocoder() {
    if (d_mel_)  cudaFree(d_mel_);
    if (d_src_)  cudaFree(d_src_);
    if (d_stft_) cudaFree(d_stft_);
    if (stream_) cudaStreamDestroy(stream_);
    delete ctx_;
    delete engine_;
    delete runtime_;
}

bool TRTVocoder::init(const TRTVocoderConfig & cfg) {
    cfg_ = cfg;

    // Load serialized engine
    std::ifstream f(cfg_.engine_path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[TRT] Cannot open engine file: %s\n", cfg_.engine_path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> plan(sz);
    f.read(plan.data(), sz);
    fprintf(stderr, "[TRT] Loaded engine: %s (%.1f MB)\n", cfg_.engine_path.c_str(), sz / 1e6);

    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_) return false;

    engine_ = runtime_->deserializeCudaEngine(plan.data(), sz);
    if (!engine_) { fprintf(stderr, "[TRT] deserializeCudaEngine failed\n"); return false; }

    ctx_ = engine_->createExecutionContext();
    if (!ctx_) { fprintf(stderr, "[TRT] createExecutionContext failed\n"); return false; }

    // Log engine I/O info (for debug), and keep the actual output name.
    int n_io = engine_->getNbIOTensors();
    int n_inputs = 0;
    fprintf(stderr, "[TRT] Engine has %d IO tensors:\n", n_io);
    for (int i = 0; i < n_io; i++) {
        const char * name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = engine_->getTensorShape(name);
        const char * mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "IN" : "OUT";
        fprintf(stderr, "[TRT]   [%d] %s (%s) dims=", i, name, mode_str);
        for (int d = 0; d < dims.nbDims; d++) fprintf(stderr, "%lld ", (long long) dims.d[d]);
        fprintf(stderr, "\n");
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            n_inputs++;
            if (std::strcmp(name, "mel") == 0) {
                mel_name_ = name;
                dynamic_mel_ = (dims.nbDims == 3 && dims.d[2] < 0);
            } else if (std::strcmp(name, "source_stft") == 0) {
                src_name_ = name;
                has_src_input_ = true;
                dynamic_src_ = (dims.nbDims == 3 && dims.d[2] < 0);
            }
        } else {
            stft_name_ = name;
            dynamic_out_ = (dims.nbDims == 3 && dims.d[2] < 0);
        }
    }

    nvinfer1::Dims mel_shape;
    mel_shape.nbDims = 3;
    mel_shape.d[0] = 1;
    mel_shape.d[1] = 80;
    mel_shape.d[2] = cfg_.T_mel;
    if (!ctx_->setInputShape(mel_name_.c_str(), mel_shape)) {
        fprintf(stderr, "[TRT] setInputShape(%s) failed\n", mel_name_.c_str());
        return false;
    }

    src_frames_ = cfg_.T_mel * SAMPLES_PER_MEL / HOP + 1;
    if (has_src_input_) {
        nvinfer1::Dims src_shape = engine_->getTensorShape(src_name_.c_str());
        if (!dynamic_src_ && src_shape.nbDims == 3 && src_shape.d[2] > 0) {
            src_frames_ = src_shape.d[2];
        }
        src_shape.nbDims = 3;
        src_shape.d[0] = 1;
        src_shape.d[1] = 18;
        src_shape.d[2] = src_frames_;
        if (!ctx_->setInputShape(src_name_.c_str(), src_shape)) {
            fprintf(stderr, "[TRT] setInputShape(%s) failed\n", src_name_.c_str());
            return false;
        }
    }

    // Allocate I/O
    cudaStreamCreate(&stream_);
    mel_bytes_  = 1 * 80 * cfg_.T_mel * sizeof(float);
    src_bytes_  = has_src_input_ ? (size_t) 1 * 18 * src_frames_ * sizeof(float) : 0;
    nvinfer1::Dims out_shape = ctx_->getTensorShape(stft_name_.c_str());
    stft_frames_ = has_src_input_ ? src_frames_ : (cfg_.T_mel * SAMPLES_PER_MEL / HOP + 1);
    if (out_shape.nbDims == 3 && out_shape.d[2] > 0) {
        stft_frames_ = out_shape.d[2];
    }
    stft_bytes_ = (size_t) 1 * 18 * stft_frames_ * sizeof(float);

    if (cudaMalloc(&d_mel_, mel_bytes_) != cudaSuccess) {
        fprintf(stderr, "[TRT] cudaMalloc mel failed\n");
        return false;
    }
    if (has_src_input_ && cudaMalloc(&d_src_, src_bytes_) != cudaSuccess) {
        fprintf(stderr, "[TRT] cudaMalloc source_stft failed\n");
        return false;
    }
    if (cudaMalloc(&d_stft_, stft_bytes_) != cudaSuccess) {
        fprintf(stderr, "[TRT] cudaMalloc stft output failed\n");
        return false;
    }
    if (!ctx_->setTensorAddress(mel_name_.c_str(), d_mel_) ||
        (has_src_input_ && !ctx_->setTensorAddress(src_name_.c_str(), d_src_)) ||
        !ctx_->setTensorAddress(stft_name_.c_str(), d_stft_)) {
        fprintf(stderr, "[TRT] setTensorAddress failed\n");
        return false;
    }

    // Pre-compute periodic Hann window, matching hg_stft16_fill_periodic_hann_f32.
    window_.resize(N_FFT);
    window_sq_.resize(N_FFT);
    for (int i = 0; i < N_FFT; ++i) {
        window_[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / (float) N_FFT);
        window_sq_[i] = window_[i] * window_[i];
    }

    ready_ = true;
    fprintf(stderr, "[TRT] Vocoder ready: T_mel=%d, src_frames=%d, stft_frames=%d, mel_bytes=%zu, src_bytes=%zu, stft_bytes=%zu\n",
            cfg_.T_mel, src_frames_, stft_frames_, mel_bytes_, src_bytes_, stft_bytes_);
    return true;
}

bool TRTVocoder::infer(const float * mel_bct, int T_mel,
                        std::vector<float> & wave_bt_out, int64_t & out_T_audio) {
    return infer(mel_bct, T_mel, nullptr, 0, wave_bt_out, out_T_audio);
}

bool TRTVocoder::infer(const float * mel_bct, int T_mel,
                        const float * source_stft_tcb, int T_source_frame,
                        std::vector<float> & wave_bt_out, int64_t & out_T_audio) {
    if (!ready_) return false;
    if (!mel_bct || T_mel <= 0) return false;
    if (T_mel > cfg_.T_mel) {
        fprintf(stderr, "[TRT] T_mel=%d exceeds max/profile T_mel=%d\n", T_mel, cfg_.T_mel);
        return false;
    }

    const int T_mel_engine = dynamic_mel_ ? T_mel : cfg_.T_mel;
    nvinfer1::Dims mel_shape;
    mel_shape.nbDims = 3;
    mel_shape.d[0] = 1;
    mel_shape.d[1] = 80;
    mel_shape.d[2] = T_mel_engine;
    if (!ctx_->setInputShape(mel_name_.c_str(), mel_shape)) {
        fprintf(stderr, "[TRT] setInputShape(%s, T=%d) failed\n", mel_name_.c_str(), T_mel_engine);
        return false;
    }

    int T_src_engine = src_frames_;
    if (has_src_input_ && dynamic_src_) {
        T_src_engine = (source_stft_tcb && T_source_frame > 0) ?
                           T_source_frame :
                           (T_mel_engine * SAMPLES_PER_MEL / HOP + 1);
        if (T_src_engine > src_frames_) {
            fprintf(stderr, "[TRT] T_source=%d exceeds max/profile T_source=%d\n", T_src_engine, src_frames_);
            return false;
        }
    }
    if (has_src_input_) {
        nvinfer1::Dims src_shape;
        src_shape.nbDims = 3;
        src_shape.d[0] = 1;
        src_shape.d[1] = 18;
        src_shape.d[2] = T_src_engine;
        if (!ctx_->setInputShape(src_name_.c_str(), src_shape)) {
            fprintf(stderr, "[TRT] setInputShape(%s, T=%d) failed\n", src_name_.c_str(), T_src_engine);
            return false;
        }
    }

    nvinfer1::Dims out_shape = ctx_->getTensorShape(stft_name_.c_str());
    int T_frame = stft_frames_;
    if (out_shape.nbDims == 3 && out_shape.d[2] > 0) {
        T_frame = out_shape.d[2];
    }
    if (T_frame <= 0 || T_frame > stft_frames_) {
        fprintf(stderr, "[TRT] invalid output T_frame=%d (max=%d)\n", T_frame, stft_frames_);
        return false;
    }

    const size_t mel_cur_bytes  = (size_t) 1 * 80 * T_mel_engine * sizeof(float);
    const size_t src_cur_bytes  = has_src_input_ ? (size_t) 1 * 18 * T_src_engine * sizeof(float) : 0;
    const size_t stft_cur_bytes = (size_t) 1 * 18 * T_frame * sizeof(float);

    // Pad/truncate only for fixed-shape engines. Dynamic engines use the real T_mel.
    // mel_bct is BCT layout for B=1: row-major [C=80, T_mel].
    std::vector<float> mel_tr((size_t) 80 * T_mel_engine, 0.0f);
    int T_copy = std::min(T_mel, T_mel_engine);
    for (int c = 0; c < 80; ++c) {
        std::memcpy(mel_tr.data() + (size_t) c * T_mel_engine,
                    mel_bct + (size_t) c * T_mel,
                    (size_t) T_copy * sizeof(float));
    }

    // Copy mel to GPU
    cudaMemcpyAsync(d_mel_, mel_tr.data(), mel_cur_bytes, cudaMemcpyHostToDevice, stream_);

    std::vector<float> src_tr;
    if (has_src_input_) {
        if (source_stft_tcb && T_source_frame > 0) {
            src_tr.assign((size_t) 18 * T_src_engine, 0.0f);
            const int T_src_copy = std::min(T_source_frame, T_src_engine);
            for (int c = 0; c < 18; ++c) {
                std::memcpy(src_tr.data() + (size_t) c * T_src_engine,
                            source_stft_tcb + (size_t) c * T_source_frame,
                            (size_t) T_src_copy * sizeof(float));
            }
            cudaMemcpyAsync(d_src_, src_tr.data(), src_cur_bytes, cudaMemcpyHostToDevice, stream_);
        } else {
            cudaMemsetAsync(d_src_, 0, src_cur_bytes, stream_);
        }
    }

    bool enq_ok = ctx_->enqueueV3(stream_);
    cudaError_t cu_err = cudaStreamSynchronize(stream_);
    if (!enq_ok) {
        fprintf(stderr, "[TRT] enqueueV3 FAILED\n");
    }
    if (cu_err != cudaSuccess) {
        fprintf(stderr, "[TRT] cu sync error: %s\n", cudaGetErrorString(cu_err));
    }
    if (!enq_ok || cu_err != cudaSuccess) {
        return false;
    }

    std::vector<float> stft_host(18 * T_frame);
    cudaMemcpy(stft_host.data(), d_stft_, stft_cur_bytes, cudaMemcpyDeviceToHost);

    if (const char * dump_dir = std::getenv("OMNI_VOC_DUMP_DIR")) {
        static int dump_idx = 0;
        const int idx = dump_idx++;
        if (idx < 8) {
            dump_f32_raw(dump_dir, "trt_mel", idx, T_mel_engine, 80, mel_tr);
            if (!src_tr.empty()) {
                dump_f32_raw(dump_dir, "trt_src", idx, T_src_engine, 18, src_tr);
            }
            dump_f32_raw(dump_dir, "trt_post", idx, T_frame, 18, stft_host);
        }
    }

    static int dbg_count = 0;
    if (std::getenv("OMNI_TRT_VOCODER_DEBUG") && dbg_count < 3) {
        fprintf(stderr, "[TRT-DBG] call=%d T_mel_in=%d T_mel_engine=%d T_source_in=%d T_source_engine=%d T_frame=%d\n",
                dbg_count, T_mel, T_mel_engine, T_source_frame, T_src_engine, T_frame);
        print_stats("mel", mel_tr.data(), mel_tr.size());
        if (source_stft_tcb && T_source_frame > 0) {
            print_stats("source_stft_in", source_stft_tcb, (size_t) 18 * T_source_frame);
        }
        if (!src_tr.empty()) {
            print_stats("source_stft_trt", src_tr.data(), src_tr.size());
        }
        print_stats("stft_out", stft_host.data(), stft_host.size());
        fprintf(stderr, "[TRT-DBG] stft first 20:");
        for (int i = 0; i < 20 && i < (int) stft_host.size(); i++) {
            fprintf(stderr, " %.6g", stft_host[(size_t) i]);
        }
        fprintf(stderr, "\n");
        dbg_count++;
    }

    // ---- Mag/Phase processing + iSTFT (matching ggml build_graph_decode) ----
    // ONNX outputs 18ch raw: first 9ch = magnitude_log, last 9ch = raw_phase
    // ggml applies: mag = clamp(exp(mag_log), -1e30, 1e2)
    //               phase = sin(raw_phase)
    //               real_ifft = mag * cos(phase)
    //               imag_ifft = mag * sin(phase)
    // Then IDFT -> window -> overlap-add -> window-square normalization -> pad trim

    int T_frame_valid = T_frame;
    if (has_src_input_ && T_source_frame > 0) {
        T_frame_valid = std::min(T_frame, T_source_frame);
    } else if (T_mel > 0) {
        T_frame_valid = std::min(T_frame, (T_mel * SAMPLES_PER_MEL) / HOP + 1);
    }
    if (T_frame_valid <= 0) {
        return false;
    }

    T_frame_out_ = T_frame_valid;
    const int audio_len_full = (T_frame_valid - 1) * HOP + N_FFT;
    std::vector<float> ola((size_t) audio_len_full, 0.0f);
    std::vector<float> wsum((size_t) audio_len_full, 0.0f);

    for (int frame = 0; frame < T_frame_valid; ++frame) {
        float real_vals[9], imag_vals[9];
        for (int f = 0; f < F_; ++f) {
            const float mag_log   = stft_host[(size_t) f * T_frame + frame];
            const float raw_phase = stft_host[(size_t) (F_ + f) * T_frame + frame];

            float mag = std::exp(mag_log);
            if (!std::isfinite(mag)) {
                mag = (mag_log > 0.0f) ? 100.0f : 0.0f;
            }
            mag = std::max(0.0f, std::min(mag, 100.0f));

            float phase = std::sin(raw_phase);
            if (!std::isfinite(phase)) {
                phase = 0.0f;
            }
            real_vals[f] = mag * std::cos(phase);
            imag_vals[f] = mag * std::sin(phase);
        }

        const int base = frame * HOP;
        for (int n = 0; n < N_FFT; ++n) {
            float v = real_vals[0] + real_vals[F_ - 1] * ((n & 1) ? -1.0f : 1.0f);
            for (int k = 1; k < F_ - 1; ++k) {
                const float angle = 2.0f * M_PI * (float) k * (float) n / (float) N_FFT;
                v += 2.0f * (real_vals[k] * std::cos(angle) - imag_vals[k] * std::sin(angle));
            }
            ola[(size_t) base + n] += (v / (float) N_FFT) * window_[n];
            wsum[(size_t) base + n] += window_sq_[n];
        }
    }

    const int trim = 2 * PAD;
    if (audio_len_full <= trim) {
        return false;
    }
    const int audio_len = audio_len_full - trim;
    wave_bt_out.assign((size_t) audio_len, 0.0f);
    for (int i = 0; i < audio_len; ++i) {
        const int src_i = i + PAD;
        const float denom = std::max(wsum[(size_t) src_i], 1e-8f);
        float v = ola[(size_t) src_i] / denom;
        if (!std::isfinite(v)) {
            v = 0.0f;
        }
        wave_bt_out[(size_t) i] = std::max(-AUDIO_LIMIT, std::min(v, AUDIO_LIMIT));
    }
    out_T_audio = (int64_t) wave_bt_out.size();

    return true;
}

} // namespace vocoder
} // namespace omni
