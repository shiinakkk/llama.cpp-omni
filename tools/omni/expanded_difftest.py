#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np

# Prefer local gguf package in this repo.
sys.path.insert(0, "/home/i_liuxinyu/llama.cpp-omni/gguf-py")
import gguf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expanded AWQ runtime-vs-offline diff test: attention + multi-layer + random sampling"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/home/i_liuxinyu/llama.cpp-omni/difftest_runtime_repack_full.log"),
        help="Runtime debug log path",
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path("/home/i_liuxinyu/models/minicpm-o-4_5-awq-w4a16-gguf/MiniCPM-o-4_5-AWQ-W4A16-marlin.gguf"),
        help="Converted marlin GGUF path",
    )
    parser.add_argument("--seed", type=int, default=20260326, help="Random seed for sampling")
    parser.add_argument("--rand-n", type=int, default=60, help="Random sample size")
    args = parser.parse_args()

    pat = re.compile(
        r"awq-marlin-debug: stage=(qweight\.post_repack|qzeros\.post_permute|scales\.post_permute) name=([^ ]+) .*? first=([^ ]+)"
    )
    layer_pat = re.compile(r"layers\.(\d+)\.")

    runtime: dict[tuple[str, str], list[str]] = {}
    for line in args.log.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        stage, name, firsts = m.groups()
        runtime[(stage, name)] = [v.strip() for v in firsts.split(",") if v.strip()]

    reader = gguf.GGUFReader(args.gguf, "r")
    tensors = {t.name: t for t in reader.tensors}

    # attention + mlp only
    cands: list[tuple[str, str, int, str]] = []
    for (stage, name), vals in runtime.items():
        if ".self_attn." not in name and ".mlp." not in name:
            continue
        if name not in tensors:
            continue
        m = layer_pat.search(name)
        layer = int(m.group(1)) if m else -1
        kind = "attn" if ".self_attn." in name else "mlp"
        cands.append((stage, name, layer, kind))

    focus_layers = {0, 2, 17, 35}
    focus = [x for x in cands if x[2] in focus_layers]

    rng = random.Random(args.seed)
    rand_n = min(args.rand_n, len(cands))
    rand_sample = rng.sample(cands, rand_n)

    # merge unique
    keys: dict[tuple[str, str], tuple[int, str]] = {(s, n): (l, k) for s, n, l, k in focus}
    for s, n, l, k in rand_sample:
        keys.setdefault((s, n), (l, k))

    stats = {
        "total": 0,
        "int_exact_ok": 0,
        "int_exact_fail": 0,
        "f16_allclose_ok": 0,
        "f16_allclose_fail": 0,
        "f16_max_abs": 0.0,
        "attn_total": 0,
        "mlp_total": 0,
    }
    fails: list[tuple[str, str, str]] = []

    for (stage, name), (_layer, kind) in sorted(keys.items(), key=lambda x: (x[1][0], x[0][1], x[0][0])):
        stats["total"] += 1
        stats[f"{kind}_total"] += 1

        rt_vals = runtime[(stage, name)][:8]
        gg_vals = np.asarray(tensors[name].data).reshape(-1)[:8]

        if stage.startswith("scales"):
            rt = np.asarray([float(x) for x in rt_vals], dtype=np.float64)
            gg = gg_vals.astype(np.float64)
            max_abs = float(np.max(np.abs(rt - gg))) if rt.size else 0.0
            stats["f16_max_abs"] = max(stats["f16_max_abs"], max_abs)
            ok = np.allclose(rt, gg, rtol=1e-3, atol=1e-5)
            if ok:
                stats["f16_allclose_ok"] += 1
            else:
                stats["f16_allclose_fail"] += 1
                fails.append((stage, name, f"max_abs={max_abs:.6g}"))
        else:
            rt = np.asarray([int(x) for x in rt_vals], dtype=np.int64)
            gg = gg_vals.astype(np.int64)
            neq = int(np.sum(rt != gg))
            if neq == 0:
                stats["int_exact_ok"] += 1
            else:
                stats["int_exact_fail"] += 1
                fails.append((stage, name, f"mismatch={neq}/8"))

    print("EXPANDED DIFFTEST REPORT")
    print(f"log_entries={len(runtime)} candidates(attn+mlp)={len(cands)}")
    print(f"focus_layers={sorted(focus_layers)} focus_count={len(focus)} random_sample={rand_n}")
    print(f"checked_unique={stats['total']} (attn={stats['attn_total']}, mlp={stats['mlp_total']})")
    print(f"int_exact_ok={stats['int_exact_ok']} int_exact_fail={stats['int_exact_fail']}")
    print(
        f"f16_allclose_ok={stats['f16_allclose_ok']} "
        f"f16_allclose_fail={stats['f16_allclose_fail']} "
        f"f16_max_abs={stats['f16_max_abs']:.6g}"
    )

    if fails:
        print("FAILURES:")
        for stage, name, msg in fails[:30]:
            print(f"- {stage} {name}: {msg}")
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
