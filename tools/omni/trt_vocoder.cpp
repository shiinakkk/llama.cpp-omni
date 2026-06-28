#include "trt_vocoder.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdio>

namespace omni {
namespace vocoder {

// ====== iSTFT constants (matching hg2_stft16_params) ======
static constexpr int N_FFT = 16;
static constexpr int HOP   = 4;
static constexpr int F_    = 9;
static constexpr int PAD   = 8;

// ====== Logger ======
namespace {
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

    // Set fixed input shapes
    nvinfer1::Dims mel_dims;
    mel_dims.nbDims = 3;
    mel_dims.d[0] = 1;
    mel_dims.d[1] = 80;
    mel_dims.d[2] = cfg_.T_mel;
    ctx_->setInputShape("mel", mel_dims);

    // Allocate I/O (single-input engine: mel → stft_18ch)
    cudaStreamCreate(&stream_);
    mel_bytes_ = 1 * 80 * cfg_.T_mel * sizeof(float);
    int T_frame = cfg_.T_mel * 64;
    stft_bytes_ = 1 * 18 * T_frame * sizeof(float);

    cudaMalloc(&d_mel_, mel_bytes_);
    cudaMalloc(&d_stft_, stft_bytes_);

    // Pre-compute Hann window
    window_.resize(N_FFT);
    for (int i = 0; i < N_FFT; ++i)
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N_FFT - 1)));

    // Pre-compute IDFT matrix: [N_FFT=16, 18] maps 9 real + 9 imag → 16 time samples
    // STFT channel layout: [0..8]=real, [9..17]=imag (F_=9 freqs, skip DC)
    idft_matrix_.resize(N_FFT * 18);
    for (int n = 0; n < N_FFT; ++n) {
        for (int k = 0; k < F_; ++k) {
            float angle = 2.0f * M_PI * (k + 1) * n / (float)N_FFT;
            idft_matrix_[n * 18 + k]       =  std::cos(angle) * 2.0f / N_FFT;
            idft_matrix_[n * 18 + F_ + k]  = -std::sin(angle) * 2.0f / N_FFT;
        }
    }

    // Window applied to idft output: window_td[n] = window[n] for each frame
    // We apply this as a post-IDFT scale

    ready_ = true;
    fprintf(stderr, "[TRT] Vocoder ready: T_mel=%d, mel_bytes=%zu, stft_bytes=%zu\n",
            cfg_.T_mel, mel_bytes_, stft_bytes_);
    return true;
}

bool TRTVocoder::infer(const float * mel_bct, int T_mel,
                        std::vector<float> & wave_bt_out, int64_t & out_T_audio) {
    if (!ready_) return false;

    // Pad/truncate mel to fixed T_mel
    // mel_bct is BCT layout [channels=80, T_mel] row-major
    std::vector<float> mel_padded(80 * cfg_.T_mel, 0.0f);
    int T_copy = std::min(T_mel, cfg_.T_mel);
    for (int t = 0; t < T_copy; ++t)
        for (int c = 0; c < 80; ++c)
            mel_padded[t * 80 + c] = mel_bct[t * 80 + c];  // BCT → T×C layout

    // Transpose to PyTorch layout [B=1, C=80, T]
    std::vector<float> mel_tr(80 * cfg_.T_mel);
    for (int c = 0; c < 80; ++c)
        for (int t = 0; t < cfg_.T_mel; ++t)
            mel_tr[c * cfg_.T_mel + t] = mel_padded[t * 80 + c];

    // Copy mel to GPU
    cudaMemcpyAsync(d_mel_, mel_tr.data(), mel_bytes_, cudaMemcpyHostToDevice, stream_);

    // source_stft: zero-fill (computed externally by ggml compute_source_stft)
    // TODO: accept source_stft from caller
    cudaMemsetAsync(d_src_, 0, src_bytes_, stream_);

    // Run TRT
    ctx_->setTensorAddress("mel", d_mel_);
    ctx_->setTensorAddress("source_stft", d_src_);
    ctx_->setTensorAddress("stft_18ch", d_stft_);
    ctx_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);

    // Copy STFT output back
    int T_frame = cfg_.T_mel * 64; // approximate; actual depends on conv strides
    std::vector<float> stft_host(18 * T_frame);
    cudaMemcpy(stft_host.data(), d_stft_, stft_bytes_, cudaMemcpyDeviceToHost);

    // ---- Mag/Phase processing + iSTFT (matching ggml build_graph_decode) ----
    // ONNX outputs 18ch raw: first 9ch = magnitude_log, last 9ch = raw_phase
    // ggml applies: mag = exp(clamp(mag_log, -1e30, 1e2))
    //               phase = sin(raw_phase)
    //               real_ifft = mag * cos(phase)
    //               imag_ifft = mag * sin(phase)
    // Then IDFT → window → overlap-add

    T_frame_out_ = T_frame;
    int audio_len = (T_frame - 1) * HOP + N_FFT;
    wave_bt_out.assign(audio_len, 0.0f);

    for (int frame = 0; frame < T_frame; ++frame) {
        // Process 18ch through mag/phase nonlinearity, compute IFFT real+imag
        float real_vals[9], imag_vals[9];
        for (int f = 0; f < F_; ++f) {
            float mag_log  = stft_host[f       * T_frame + frame];  // first 9ch
            float raw_phase = stft_host[(F_ + f) * T_frame + frame]; // last 9ch

            float mag   = std::exp(std::max(-1e30f, std::min(mag_log, 1e2f)));
            float phase = std::sin(raw_phase);
            real_vals[f] = mag * std::cos(phase);
            imag_vals[f] = mag * std::sin(phase);
        }

        // IDFT: real[9] + imag[9] → time[16] using pre-computed matrix
        int base = frame * HOP;
        for (int n = 0; n < N_FFT; ++n) {
            float v = 0;
            const float * row = &idft_matrix_[n * 18];
            for (int k = 0; k < F_; ++k) {
                v += row[k] * real_vals[k] + row[F_ + k] * imag_vals[k];
            }
            wave_bt_out[base + n] += v * window_[n];
        }
    }

    // Trim padding (matching ggml: trim = 2 * PAD samples from end)
    out_T_audio = audio_len - 2 * PAD;
    if (out_T_audio < 0) out_T_audio = 0;

    return true;
}

} // namespace vocoder
} // namespace omni
