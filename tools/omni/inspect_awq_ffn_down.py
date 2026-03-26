#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader  # noqa: E402


PACK_FACTOR = 8


@dataclass
class DebugTensorRecord:
    name: str
    op: str
    shape: tuple[int, int, int, int]
    values: list[float]


@dataclass
class DumpTensorRecord:
    name: str
    op: str
    ggml_type: str
    shape: tuple[int, int, int, int]
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MiniCPM AWQ ffn_down tensors and debug-log prefixes."
    )
    parser.add_argument(
        "--gguf",
        default="/models/minicpm-o-4_5-awq-w4a16-gguf/MiniCPM-o-4_5-AWQ-W4A16.gguf",
        help="Path to the AWQ GGUF.",
    )
    parser.add_argument(
        "--log",
        default="/home/i_liuxinyu/llama_omni_cli_debug_tensor.log",
        help="Path to the debug tensor log.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=2,
        help="Transformer layer index to inspect.",
    )
    parser.add_argument(
        "--prefix",
        type=int,
        default=16,
        help="How many prefix values to print for decoded tensors.",
    )
    parser.add_argument(
        "--top-cols",
        type=int,
        default=8,
        help="How many output columns to summarize.",
    )
    parser.add_argument(
        "--dump-dir",
        help="Directory containing LLAMA_DEBUG_TENSOR_DUMP_DIR tensor dumps.",
    )
    parser.add_argument(
        "--dump-token",
        type=int,
        default=0,
        help="Token column inside the dumped [ne0, ne1] tensor to compare.",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=8,
        help="How many mismatching output positions to print.",
    )
    return parser.parse_args()


def find_tensor(reader: GGUFReader, name: str):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    raise KeyError(f"tensor not found: {name}")


def unpack_u4_cols(packed: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.empty((rows, cols), dtype=np.uint8)
    for i in range(PACK_FACTOR):
        out[:, i::PACK_FACTOR] = ((packed >> (4 * i)) & 0xF).astype(np.uint8)
    return out


def parse_debug_log(path: Path) -> list[DebugTensorRecord]:
    header_re = re.compile(
        r"^DEBUG_TENSOR: name=(?P<name>.+?) op=(?P<op>\S+) type=(?P<type>\S+) "
        r"shape=\s*(?P<n0>\d+),\s*(?P<n1>\d+),\s*(?P<n2>\d+),\s*(?P<n3>\d+) n=(?P<n>\d+)"
    )

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[DebugTensorRecord] = []
    idx = 0
    while idx + 1 < len(lines):
        match = header_re.match(lines[idx])
        if not match:
            idx += 1
            continue
        values_line = lines[idx + 1]
        if not values_line.startswith("DEBUG_TENSOR: values="):
            idx += 1
            continue
        values_str = values_line.split("=", 1)[1].strip()
        values: list[float] = []
        for item in values_str.split(","):
            token = item.strip()
            if token.lower() == "nan":
                values.append(math.nan)
            elif token.lower() == "inf":
                values.append(math.inf)
            elif token.lower() == "-inf":
                values.append(-math.inf)
            else:
                values.append(float(token))
        records.append(
            DebugTensorRecord(
                name=match.group("name"),
                op=match.group("op"),
                shape=tuple(int(match.group(f"n{i}")) for i in range(4)),
                values=values,
            )
        )
        idx += 2
    return records


def select_last(records: list[DebugTensorRecord], name: str, shape0: int | None = None) -> DebugTensorRecord | None:
    for record in reversed(records):
        if record.name != name:
            continue
        if shape0 is not None and record.shape[0] != shape0:
            continue
        return record
    return None


def parse_meta_file(path: Path) -> DumpTensorRecord:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    shape = tuple(int(values[f"ne{i}"]) for i in range(4))
    return DumpTensorRecord(
        name=values["name"],
        op=values["op"],
        ggml_type=values["type"],
        shape=shape,
        path=path.with_suffix(".bin"),
    )


def load_dump_records(dump_dir: Path) -> list[DumpTensorRecord]:
    return [parse_meta_file(path) for path in sorted(dump_dir.glob("*.meta"))]


def find_dump_tensor(
    records: list[DumpTensorRecord],
    name: str,
    shape0: int | None = None,
    shape1: int | None = None,
) -> DumpTensorRecord | None:
    for record in reversed(records):
        if record.name != name:
            continue
        if shape0 is not None and record.shape[0] != shape0:
            continue
        if shape1 is not None and record.shape[1] != shape1:
            continue
        return record
    return None


def load_dump_tensor(record: DumpTensorRecord) -> np.ndarray:
    dtype_map = {
        "f32": np.float32,
        "f16": np.float16,
        "bf16": np.uint16,
        "i32": np.int32,
    }
    if record.ggml_type not in dtype_map:
        raise ValueError(f"unsupported dump tensor type: {record.ggml_type}")
    raw = np.fromfile(record.path, dtype=dtype_map[record.ggml_type])
    expected = math.prod(record.shape)
    if raw.size != expected:
        raise ValueError(f"dump size mismatch for {record.path}: got {raw.size}, expected {expected}")
    if record.ggml_type == "bf16":
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    tensor = raw.reshape(record.shape[3], record.shape[2], record.shape[1], record.shape[0]).transpose(3, 2, 1, 0)
    return tensor[:, :, 0, 0]


def summarize_compare(cpu: np.ndarray, runtime: np.ndarray, max_report: int) -> None:
    diff = cpu - runtime
    abs_diff = np.abs(diff)
    finite_mask = np.isfinite(runtime)
    print(f"cpu_vs_runtime finite_count={int(np.count_nonzero(finite_mask))}/{runtime.size}")
    if np.any(finite_mask):
        finite_abs = abs_diff[finite_mask]
        print(
            "cpu_vs_runtime "
            f"max_abs_err={float(np.max(finite_abs)):.6f} "
            f"mean_abs_err={float(np.mean(finite_abs)):.6f}"
        )

        worst = np.argsort(finite_abs)[::-1][:max_report]
        finite_pos = np.flatnonzero(finite_mask)
        print("top_mismatches:")
        for idx in worst:
            flat = int(finite_pos[idx])
            row = flat % runtime.shape[0]
            col = flat // runtime.shape[0]
            print(
                f"  out[{row}, {col}] cpu={float(cpu[row, col]):.6f} "
                f"runtime={float(runtime[row, col]):.6f} abs_err={float(abs_diff[row, col]):.6f}"
            )

    nonfinite_mask = ~np.isfinite(runtime)
    if np.any(nonfinite_mask):
        flat = np.flatnonzero(nonfinite_mask)[:max_report]
        print("runtime_nonfinite:")
        for pos in flat:
            row = int(pos % runtime.shape[0])
            col = int(pos // runtime.shape[0])
            print(
                f"  out[{row}, {col}] cpu={float(cpu[row, col]):.6f} "
                f"runtime={runtime[row, col]}"
            )


def main() -> int:
    args = parse_args()

    gguf_path = Path(args.gguf)
    log_path = Path(args.log)

    layer_prefix = f"llm.model.layers.{args.layer}.mlp.down_proj"
    qweight_name = f"{layer_prefix}.qweight"
    qzeros_name = f"{layer_prefix}.qzeros"
    scales_name = f"{layer_prefix}.scales"

    reader = GGUFReader(gguf_path)
    qweight_tensor = find_tensor(reader, qweight_name)
    qzeros_tensor = find_tensor(reader, qzeros_name)
    scales_tensor = find_tensor(reader, scales_name)

    size_k = int(qweight_tensor.shape[0])
    packed_n = int(qweight_tensor.shape[1])
    size_n = packed_n * PACK_FACTOR
    num_groups = int(qzeros_tensor.shape[0])
    group_size = size_k // num_groups

    qweight_packed = np.asarray(qweight_tensor.data, dtype=np.uint32).T.copy()
    qzeros_packed = np.asarray(qzeros_tensor.data, dtype=np.uint32).T.copy()
    scales = np.asarray(scales_tensor.data, dtype=np.float32).T.copy()

    qweight_u4 = unpack_u4_cols(qweight_packed, size_k, size_n)
    qzeros_u4 = unpack_u4_cols(qzeros_packed, num_groups, size_n)

    repeated_scales = np.repeat(scales, group_size, axis=0)
    repeated_qzeros = np.repeat(qzeros_u4.astype(np.float32), group_size, axis=0)
    weight_f32 = (qweight_u4.astype(np.float32) - repeated_qzeros) * repeated_scales

    print(f"GGUF: {gguf_path}")
    print(f"layer={args.layer} size_k={size_k} size_n={size_n} num_groups={num_groups} group_size={group_size}")
    print(f"qweight logical shape={qweight_packed.shape} raw data shape={qweight_tensor.data.shape}")
    print(f"qzeros logical shape={qzeros_packed.shape} raw data shape={qzeros_tensor.data.shape}")
    print(f"scales logical shape={scales.shape} raw data shape={scales_tensor.data.shape}")
    print()

    prefix = args.prefix
    top_cols = args.top_cols

    print("qweight_u4[0, :prefix] =", qweight_u4[0, :prefix].tolist())
    print("qzeros_u4[0, :prefix]  =", qzeros_u4[0, :prefix].tolist())
    print("scales[0, :prefix]     =", [float(x) for x in scales[0, :prefix]])
    print("dequant_w[0, :prefix]  =", [float(x) for x in weight_f32[0, :prefix]])
    print()

    col_max = np.max(np.abs(weight_f32[:, :top_cols]), axis=0)
    col_mean = np.mean(weight_f32[:, :top_cols], axis=0)
    col_std = np.std(weight_f32[:, :top_cols], axis=0)
    for col in range(top_cols):
        print(
            f"col={col:2d} "
            f"zp0={int(qzeros_u4[0, col]):2d} "
            f"scale0={float(scales[0, col]):.6f} "
            f"w_max_abs={float(col_max[col]):.6f} "
            f"w_mean={float(col_mean[col]):.6f} "
            f"w_std={float(col_std[col]):.6f}"
        )

    if not log_path.exists():
        return 0

    records = parse_debug_log(log_path)
    gate_in = select_last(records, f"awq_marlin_input-{args.layer}", shape0=4096)
    swiglu = select_last(records, f"ffn_swiglu-{args.layer}")
    down_in = select_last(records, f"awq_marlin_input-{args.layer}", shape0=12288)
    down_out = select_last(records, f"awq_marlin_mm-{args.layer}", shape0=4096)

    print()
    print(f"Log: {log_path}")
    if gate_in is not None:
        print(f"last gate/up input prefix = {gate_in.values}")
    if swiglu is not None:
        print(f"ffn_swiglu prefix         = {swiglu.values}")
    if down_in is not None:
        print(f"down_proj input prefix    = {down_in.values}")
    if down_out is not None:
        print(f"down_proj output prefix   = {down_out.values}")

    if down_in is not None:
        print()
        print(
            "Note: debug log only contains tensor prefixes, so this tool currently "
            "reports GGUF/dequant statistics and log prefixes, but cannot yet do a "
            "full CPU matmul without a complete dumped down_proj input tensor."
        )

    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        dump_records = load_dump_records(dump_dir)
        down_in_dump = find_dump_tensor(dump_records, f"awq_marlin_input-{args.layer}", shape0=size_k)
        down_out_dump = find_dump_tensor(dump_records, f"awq_marlin_mm-{args.layer}", shape0=size_n)

        print()
        print(f"Dump dir: {dump_dir}")
        if down_in_dump is None or down_out_dump is None:
            print("Missing dumped down_proj input/output tensors; set LLAMA_DEBUG_TENSOR_DUMP_DIR and rerun.")
            return 0

        down_input = load_dump_tensor(down_in_dump).astype(np.float32)
        runtime_output = load_dump_tensor(down_out_dump).astype(np.float32)

        if args.dump_token < 0 or args.dump_token >= down_input.shape[1]:
            raise ValueError(f"--dump-token out of range: {args.dump_token}, input has {down_input.shape[1]} columns")

        cpu_output = weight_f32.T @ down_input
        print(
            f"cpu_matmul shape={cpu_output.shape} "
            f"input_shape={down_input.shape} runtime_shape={runtime_output.shape}"
        )
        print(
            f"token={args.dump_token} "
            f"cpu_prefix={[float(x) for x in cpu_output[:prefix, args.dump_token]]}"
        )
        print(
            f"token={args.dump_token} "
            f"runtime_prefix={[float(x) for x in runtime_output[:prefix, args.dump_token]]}"
        )
        summarize_compare(cpu_output, runtime_output, args.max_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
