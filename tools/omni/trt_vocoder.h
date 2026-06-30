#pragma once
// TRT-accelerated HiFi-GAN vocoder for MiniCPM-o.
// TRT handles main conv/resblock path. Source excitation (F0+NSF+STFT) is computed
// externally via ggml compute_source_stft and passed as input.

#include <vector>
#include <string>
#include <cstdint>

namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}
typedef struct CUstream_st *cudaStream_t;

namespace omni {
namespace vocoder {

struct TRTVocoderConfig {
    std::string engine_path;  // path to .plan file (2-input full ONNX: mel + source_stft → 18ch)
    int         T_mel = 100;
};

class TRTVocoder {
public:
    TRTVocoder();
    ~TRTVocoder();

    bool init(const TRTVocoderConfig & cfg);
    bool is_ready() const { return ready_; }

    /// Infer vocoder: mel[80*T_mel] (BCT, B=1) → wave_bt_out.
    /// Source STFT must be computed externally via voc_hg2_runner::compute_source_stft.
    bool infer(const float * mel_bct, int T_mel,
               std::vector<float> & wave_bt_out, int64_t & out_T_audio);
    bool infer(const float * mel_bct, int T_mel,
               const float * source_stft_tcb, int T_source_frame,
               std::vector<float> & wave_bt_out, int64_t & out_T_audio);

private:
    bool ready_ = false;
    TRTVocoderConfig cfg_;

    nvinfer1::IRuntime          * runtime_ = nullptr;
    nvinfer1::ICudaEngine       * engine_  = nullptr;
    nvinfer1::IExecutionContext * ctx_     = nullptr;
    cudaStream_t                  stream_  = 0;

    void * d_mel_  = nullptr;
    void * d_src_  = nullptr;
    void * d_stft_ = nullptr;
    size_t mel_bytes_  = 0;
    size_t src_bytes_  = 0;
    size_t stft_bytes_ = 0;
    bool has_src_input_ = false;
    bool dynamic_mel_   = false;
    bool dynamic_src_   = false;
    bool dynamic_out_   = false;
    int  src_frames_    = 0;
    int  stft_frames_   = 0;
    std::string mel_name_  = "mel";
    std::string src_name_  = "source_stft";
    std::string stft_name_ = "stft_18ch";

    std::vector<float> window_;
    std::vector<float> window_sq_;
    int T_frame_out_ = 0;
};

} // namespace vocoder
} // namespace omni
