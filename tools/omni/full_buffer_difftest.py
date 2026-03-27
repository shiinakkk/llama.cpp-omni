#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, "/home/i_liuxinyu/llama.cpp-omni/gguf-py")
import gguf

FNV64_OFFSET = np.uint64(1469598103934665603)
FNV64_PRIME = np.uint64(1099511628211)


@njit(cache=True)
def fnv1a64_u8(data: np.ndarray) -> np.uint64:
    h = FNV64_OFFSET
    for i in range(data.size):
        h = h ^ np.uint64(data[i])
        h = h * FNV64_PRIME
    return h


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full-buffer diff test for all AWQ post-repack tensors via runtime hash vs converted GGUF hash"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/home/i_liuxinyu/llama.cpp-omni/difftest_runtime_fullhash.log"),
        help="Runtime log with full_fnv1a64 entries",
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path("/home/i_liuxinyu/models/minicpm-o-4_5-awq-w4a16-gguf/MiniCPM-o-4_5-AWQ-W4A16-marlin.gguf"),
        help="Converted marlin GGUF path",
    )
    args = parser.parse_args()

    pat = re.compile(
        r"awq-marlin-debug: stage=(qweight\.post_repack|qzeros\.post_permute|scales\.post_permute) "
        r"name=([^ ]+) .*? full_fnv1a64=0x([0-9a-fA-F]+) nbytes=([0-9]+)"
    )

    runtime: dict[tuple[str, str], tuple[int, int]] = {}
    for line in args.log.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        stage, name, h, nbytes = m.groups()
        runtime[(stage, name)] = (int(h, 16), int(nbytes))

    reader = gguf.GGUFReader(args.gguf, "r")
    tensors = {t.name: t for t in reader.tensors}

    mismatch: list[tuple[str, str, int, int, int, int]] = []
    missing: list[tuple[str, str, str]] = []
    checked = 0
    total = len(runtime)

    for idx, ((stage, name), (h_rt, n_rt)) in enumerate(sorted(runtime.items()), start=1):
        if name not in tensors:
            missing.append((stage, name, "tensor_missing_in_converted_gguf"))
            continue

        arr = np.ascontiguousarray(np.asarray(tensors[name].data))
        raw_u8 = arr.view(np.uint8).reshape(-1)
        n_gg = int(raw_u8.size)
        checked += 1

        if n_rt != n_gg:
            mismatch.append((stage, name, h_rt, -1, n_rt, n_gg))
            continue

        h = int(fnv1a64_u8(raw_u8))
        if h_rt != h:
            mismatch.append((stage, name, h_rt, h, n_rt, n_gg))

        if idx % 50 == 0:
            print(f"progress: {idx}/{total} checked={checked} mismatch={len(mismatch)} missing={len(missing)}", flush=True)

    print("FULL BUFFER DIFFTEST (252 triplets / 756 tensors)")
    print(f"runtime_entries={len(runtime)} checked={checked} mismatches={len(mismatch)} missing={len(missing)}")

    if missing:
        print("MISSING SAMPLE:")
        for st, nm, msg in missing[:10]:
            print(f"- {st} {nm}: {msg}")

    if mismatch:
        print("MISMATCH SAMPLE:")
        for st, nm, h1, h2, n1, n2 in mismatch[:20]:
            h2s = "<size-mismatch>" if h2 < 0 else f"0x{h2:016x}"
            print(f"- {st} {nm}: runtime=0x{h1:016x} gguf={h2s} nbytes_rt={n1} nbytes_gg={n2}")

    ok = not mismatch and not missing
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
