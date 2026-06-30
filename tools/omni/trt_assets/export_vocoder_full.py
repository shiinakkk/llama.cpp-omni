#!/usr/bin/env python3
"""Export FULL HiFi-GAN vocoder (mel + source_stft → 18ch), matching ggml build_graph_decode exactly."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../gguf-py"))
from gguf import GGUFReader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GGUF_PATH = os.environ.get("GGUF_PATH",
    "/models/MiniCPM-o-4_5-gguf/token2wav-gguf/hifigan2.gguf")

reader = GGUFReader(GGUF_PATH)
def g2t(t):
    s = list(t.shape)
    if len(s) <= 1:
        return torch.from_numpy(np.frombuffer(t.data, dtype=np.float32).copy())
    return torch.from_numpy(np.frombuffer(t.data, dtype=np.float32).reshape(list(reversed(s))).copy())

W = {t.name: g2t(t) for t in reader.tensors}
print(f"Loaded {len(W)} tensors, {sum(w.numel() for w in W.values()):,} params")

lrelu_slope = 0.1
HG2_SAMPLES_PER_MEL = 480
HG2_STFT_HOP = 4

class Snake(nn.Module):
    def __init__(s, c): super().__init__(); s.alpha = nn.Parameter(torch.ones(c))
    def forward(s, x): a = s.alpha.view(1, -1, 1); return x + (torch.sin(a * x) ** 2) / (a + 1e-9)

class ResBlock(nn.Module):
    def __init__(s, c, k, dl):
        super().__init__()
        s.c1 = nn.ModuleList([nn.Conv1d(c, c, k, dilation=d, padding=(k//2)*d) for d in dl])
        s.c2 = nn.ModuleList([nn.Conv1d(c, c, k, dilation=1, padding=k//2) for _ in dl])
        s.s1 = nn.ModuleList([Snake(c) for _ in dl])
        s.s2 = nn.ModuleList([Snake(c) for _ in dl])
    def forward(s, x):
        for c1, c2, s1, s2 in zip(s.c1, s.c2, s.s1, s.s2):
            y = c1(s1(x))
            y = c2(s2(y))
            x = x + y
        return x

class FullHiFiGAN(nn.Module):
    """Exact match to ggml hg2_hift_generator::build_graph_decode."""
    def __init__(s):
        super().__init__()
        # F0 predictor: 5 conv1d layers + linear
        s.f0_conv = nn.ModuleList([nn.Conv1d(80 if i == 0 else 512, 512, 3, padding=1) for i in range(5)])
        s.f0_linear = nn.Linear(512, 1)

        # conv_pre
        s.conv_pre = nn.Conv1d(80, 512, 7, padding=3)

        # Upsamplers
        s.up0 = nn.ConvTranspose1d(512, 256, 16, stride=8, padding=4)
        s.up1 = nn.ConvTranspose1d(256, 128, 11, stride=5, padding=3)
        s.up2 = nn.ConvTranspose1d(128,  64,  7, stride=3, padding=2)

        # source_downs: merge STFT source at each resolution
        s.sd0 = nn.Conv1d(18, 256, 30, stride=15, padding=7)
        s.sd1 = nn.Conv1d(18, 128,  6, stride=3,  padding=1)
        s.sd2 = nn.Conv1d(18,  64,  1, stride=1,  padding=0)

        # source_resblocks
        s.srb0 = ResBlock(256,  7, [1, 3, 5])
        s.srb1 = ResBlock(128,  7, [1, 3, 5])
        s.srb2 = ResBlock( 64, 11, [1, 3, 5])

        # main resblocks: 9 blocks, 3 per resolution
        s.rb = nn.ModuleList()
        for ch, ks in [(256, [3,7,11]), (128, [3,7,11]), (64, [3,7,11])]:
            for k in ks:
                s.rb.append(ResBlock(ch, k, [1, 3, 5]))

        # conv_post
        s.conv_post = nn.Conv1d(64, 18, 7, padding=3)

    def load(s):
        d = {}
        cp = lambda src, dst: d.update({dst: W[src]})

        # F0 predictor
        for i in range(5):
            cp(f'f0_predictor.condnet.{i*2}.weight', f'f0_conv.{i}.weight')
            cp(f'f0_predictor.condnet.{i*2}.bias',   f'f0_conv.{i}.bias')
        cp('f0_predictor.classifier.weight', 'f0_linear.weight')
        cp('f0_predictor.classifier.bias',   'f0_linear.bias')

        # Main path
        cp('conv_pre.weight', 'conv_pre.weight'); cp('conv_pre.bias', 'conv_pre.bias')
        cp('ups.0.weight', 'up0.weight');  cp('ups.0.bias', 'up0.bias')
        cp('ups.1.weight', 'up1.weight');  cp('ups.1.bias', 'up1.bias')
        cp('ups.2.weight', 'up2.weight');  cp('ups.2.bias', 'up2.bias')

        # source_downs + source_resblocks
        for i, sd_n in enumerate(['sd0', 'sd1', 'sd2']):
            cp(f'source_downs.{i}.weight', f'{sd_n}.weight')
            cp(f'source_downs.{i}.bias',   f'{sd_n}.bias')
        for i, srb_n in enumerate(['srb0', 'srb1', 'srb2']):
            for j in range(3):
                for sgguf, spt, sn in [('convs1','c1','s1'), ('convs2','c2','s2')]:
                    cp(f'source_resblocks.{i}.{sgguf}.{j}.weight', f'{srb_n}.{spt}.{j}.weight')
                    cp(f'source_resblocks.{i}.{sgguf}.{j}.bias',  f'{srb_n}.{spt}.{j}.bias')
                    act = '2' if sn.startswith('s2') else '1'
                    cp(f'source_resblocks.{i}.activations{act}.{j}.alpha', f'{srb_n}.{sn}.{j}.alpha')

        # 9 main resblocks
        for i in range(9):
            for j in range(3):
                cp(f'resblocks.{i}.convs1.{j}.weight', f'rb.{i}.c1.{j}.weight')
                cp(f'resblocks.{i}.convs1.{j}.bias',   f'rb.{i}.c1.{j}.bias')
                cp(f'resblocks.{i}.convs2.{j}.weight', f'rb.{i}.c2.{j}.weight')
                cp(f'resblocks.{i}.convs2.{j}.bias',   f'rb.{i}.c2.{j}.bias')
                cp(f'resblocks.{i}.activations1.{j}.alpha', f'rb.{i}.s1.{j}.alpha')
                cp(f'resblocks.{i}.activations2.{j}.alpha', f'rb.{i}.s2.{j}.alpha')

        # conv_post
        cp('conv_post.weight', 'conv_post.weight'); cp('conv_post.bias', 'conv_post.bias')

        s.load_state_dict(d, strict=True)
        print("Weights loaded (full ggml match)")

    def forward(s, mel, source_stft):
        """
        mel:  [B=1, 80, T_mel]  — mel spectrogram from flow matching
        source_stft: [B=1, 18, T_stft]  — STFT of NSF harmonic source (pre-computed in C++)
        Returns: [B=1, 18, T_frame] — raw 18ch STFT
        """
        B, Cm, Tm = mel.shape
        _, Cs, Ts = source_stft.shape

        # Main path: conv_pre → leaky_relu
        x = s.conv_pre(mel)
        x = F.leaky_relu(x, lrelu_slope)

        # Level 0: 256 channels
        x = s.up0(x)
        si0 = s.sd0(source_stft)
        si0 = s.srb0(si0)
        x = x + si0
        x = sum(s.rb[i](x) for i in range(3)) / 3.0
        x = F.leaky_relu(x, lrelu_slope)

        # Level 1: 128 channels
        x = s.up1(x)
        si1 = s.sd1(source_stft)
        si1 = s.srb1(si1)
        x = x + si1
        x = sum(s.rb[i](x) for i in range(3, 6)) / 3.0
        x = F.leaky_relu(x, lrelu_slope)

        # Level 2: 64 channels
        x = s.up2(x)
        x = F.pad(x, (1, 0), mode="reflect")
        si2 = s.sd2(source_stft)
        si2 = s.srb2(si2)
        x = x + si2
        x = sum(s.rb[i](x) for i in range(6, 9)) / 3.0
        x = F.leaky_relu(x, 0.01)

        return s.conv_post(x)  # [B, 18, Tout]


# ====== Export ======
m = FullHiFiGAN()
m.load()
m.eval()

TMEL = 100
TSRC = TMEL * HG2_SAMPLES_PER_MEL // HG2_STFT_HOP + 1

dummy_mel = torch.randn(1, 80, TMEL)
dummy_src = torch.randn(1, 18, TSRC)

with torch.no_grad():
    out = m(dummy_mel, dummy_src)
print(f"mel {list(dummy_mel.shape)} + src {list(dummy_src.shape)} → stft {list(out.shape)}")

ONNX = os.environ.get("ONNX_OUTPUT", "/workspace/vocoder_full.onnx")
torch.onnx.export(
    m, (dummy_mel, dummy_src), ONNX,
    input_names=["mel", "source_stft"],
    output_names=["stft_18ch"],
    opset_version=17,
    dynamo=False,
    dynamic_axes={
        "mel": {2: "T_mel"},
        "source_stft": {2: "T_source"},
        "stft_18ch": {2: "T_frame"},
    },
)
sz = os.path.getsize(ONNX) / 1048576
print(f"Exported: {ONNX} ({sz:.1f} MB)")
