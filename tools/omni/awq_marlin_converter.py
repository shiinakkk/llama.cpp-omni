#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

# Prefer local gguf package in this repo.
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).resolve().parents[2] / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

import gguf

logger = logging.getLogger("awq-marlin-converter")

DEFAULT_INPUT = Path(
    "/home/i_liuxinyu/models/minicpm-o-4_5-awq-w4a16-gguf/MiniCPM-o-4_5-AWQ-W4A16.gguf"
)
DEFAULT_OUTPUT = Path(
    "/home/i_liuxinyu/models/minicpm-o-4_5-awq-w4a16-gguf/MiniCPM-o-4_5-AWQ-W4A16-marlin.gguf"
)

SCALE_PERM_64 = np.asarray([
    0, 8, 16, 24, 32, 40, 48, 56,
    1, 9, 17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58,
    3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60,
    5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62,
    7, 15, 23, 31, 39, 47, 55, 63,
], dtype=np.int64)

SCALE_PERM_32 = np.asarray([
    0, 1, 8, 9, 16, 17, 24, 25,
    2, 3, 10, 11, 18, 19, 26, 27,
    4, 5, 12, 13, 20, 21, 28, 29,
    6, 7, 14, 15, 22, 23, 30, 31,
], dtype=np.int64)

AWQ_INTERLEAVE = np.asarray([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)
AWQ_UNDO_INTERLEAVE = np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
MARLIN_TILE_K = 16
MARLIN_TILE_N = 64


@njit(cache=True)
def _repack_qweight_marlin_numba(qweight_u32: np.ndarray, size_n: int, num_bits: int) -> np.ndarray:
    pack_factor = 32 // num_bits
    size_k = qweight_u32.shape[0]
    packed_cols = qweight_u32.shape[1]
    k_tiles = size_k // MARLIN_TILE_K
    n_tiles = size_n // MARLIN_TILE_N
    tile_words = (MARLIN_TILE_K * MARLIN_TILE_N) // pack_factor  # 128 for int4

    out = np.empty((size_k * packed_cols,), dtype=np.uint32)

    for k_tile in range(k_tiles):
        row0 = k_tile * MARLIN_TILE_K
        for n_tile in range(n_tiles):
            col0_packed = n_tile * (MARLIN_TILE_N // pack_factor)
            out_off = (k_tile * n_tiles + n_tile) * tile_words

            for warp_id in range(4):
                for th_id in range(32):
                    tc_col = th_id // 4
                    tc_row = (th_id % 4) * 2

                    cur_n = warp_id * 16 + tc_col
                    cur_n_packed = cur_n // pack_factor
                    cur_n_pos = cur_n % pack_factor

                    if cur_n_pos == 0:
                        cur_n_pos_unpacked = 0
                    elif cur_n_pos == 1:
                        cur_n_pos_unpacked = 4
                    elif cur_n_pos == 2:
                        cur_n_pos_unpacked = 1
                    elif cur_n_pos == 3:
                        cur_n_pos_unpacked = 5
                    elif cur_n_pos == 4:
                        cur_n_pos_unpacked = 2
                    elif cur_n_pos == 5:
                        cur_n_pos_unpacked = 6
                    elif cur_n_pos == 6:
                        cur_n_pos_unpacked = 3
                    else:
                        cur_n_pos_unpacked = 7

                    vals = np.empty((8,), dtype=np.uint32)
                    for i in range(4):
                        if i == 0:
                            offset = 0
                        elif i == 1:
                            offset = 1
                        elif i == 2:
                            offset = 8
                        else:
                            offset = 9
                        cur_elem = tc_row + offset
                        packed_src_0 = qweight_u32[row0 + cur_elem, col0_packed + cur_n_packed]
                        packed_src_1 = qweight_u32[row0 + cur_elem, col0_packed + cur_n_packed + 1]
                        vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & 0xF
                        vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & 0xF

                    res = np.uint32(0)
                    # pack_idx = (0,2,4,6,1,3,5,7)
                    res |= (vals[0] & 0xF) << 0
                    res |= (vals[2] & 0xF) << 4
                    res |= (vals[4] & 0xF) << 8
                    res |= (vals[6] & 0xF) << 12
                    res |= (vals[1] & 0xF) << 16
                    res |= (vals[3] & 0xF) << 20
                    res |= (vals[5] & 0xF) << 24
                    res |= (vals[7] & 0xF) << 28

                    out[out_off + th_id * 4 + warp_id] = res

    return out


@dataclass
class AWQTriplet:
    base: str
    qweight: str
    qzeros: str
    scales: str


@dataclass
class ConvertStats:
    total_tensors: int = 0
    transformed_tensors: int = 0
    transformed_triplets: int = 0


def ggml_2d_shape(tensor: Any) -> tuple[int, int]:
    if len(tensor.shape) < 2:
        raise ValueError(f"Expected tensor with >=2 dims, got shape={tensor.shape}")
    return int(tensor.shape[0]), int(tensor.shape[1])


def tensor_np_to_ggml_2d(tensor: Any, arr: np.ndarray) -> np.ndarray:
    ne0, ne1 = ggml_2d_shape(tensor)
    return np.ascontiguousarray(np.asarray(arr).reshape(-1).reshape(ne0, ne1))


def ggml_2d_to_tensor_np_shape(tensor: Any, arr_ggml_2d: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(arr_ggml_2d).reshape(-1).reshape(tensor.data.shape))


def restore_awq_layout_to_hf_2d(ggml_2d: np.ndarray) -> np.ndarray:
    if ggml_2d.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {ggml_2d.shape}")
    rows, cols = ggml_2d.shape
    src_flat = np.asarray(ggml_2d).reshape(-1)
    src_idx = np.arange(rows * cols, dtype=np.int64).reshape(cols, rows).T.reshape(-1)
    dst_flat = src_flat[src_idx]
    return np.ascontiguousarray(dst_flat.reshape(rows, cols))


def permute_scales(scales_f32: np.ndarray, size_k: int, group_size: int) -> np.ndarray:
    if scales_f32.dtype != np.float32:
        raise ValueError(f"scales dtype must be float32 before permutation, got {scales_f32.dtype}")
    if scales_f32.ndim != 2:
        raise ValueError(f"scales must be 2D, got {scales_f32.shape}")

    use_full_perm = group_size < size_k and group_size != -1
    perm = SCALE_PERM_64 if use_full_perm else SCALE_PERM_32
    block = perm.size

    rows, cols = scales_f32.shape
    if cols % block != 0:
        raise ValueError(f"scales cols ({cols}) is not divisible by perm block ({block})")

    reshaped = scales_f32.reshape(rows, cols // block, block)
    permuted = reshaped[:, :, perm].reshape(rows, cols)
    return np.ascontiguousarray(permuted.astype(np.float16))


def _unpack_int4(packed_i32: np.ndarray, num_bits: int = 4) -> np.ndarray:
    if packed_i32.dtype != np.int32:
        raise ValueError(f"packed dtype must be int32, got {packed_i32.dtype}")
    pack_factor = 32 // num_bits

    packed_u32 = packed_i32.view(np.uint32)
    rows, packed_cols = packed_u32.shape
    shifts = (np.arange(pack_factor, dtype=np.uint32) * num_bits).reshape(1, 1, pack_factor)
    unpacked = ((packed_u32[:, :, None] >> shifts) & np.uint32(0xF)).astype(np.uint32)
    return unpacked.reshape(rows, packed_cols * pack_factor)


def _repack_int4(values_u32: np.ndarray, num_bits: int = 4) -> np.ndarray:
    if values_u32.dtype != np.uint32:
        raise ValueError(f"values dtype must be uint32, got {values_u32.dtype}")
    pack_factor = 32 // num_bits

    rows, size_n = values_u32.shape
    if size_n % pack_factor != 0:
        raise ValueError(f"size_n ({size_n}) is not divisible by pack_factor ({pack_factor})")

    packed_cols = size_n // pack_factor
    vals = values_u32.reshape(rows, packed_cols, pack_factor)
    shifts = (np.arange(pack_factor, dtype=np.uint32) * num_bits).reshape(1, 1, pack_factor)
    packed = np.bitwise_or.reduce((vals & np.uint32(0xF)) << shifts, axis=2)
    return np.ascontiguousarray(packed.view(np.int32))


def convert_qzeros_to_marlin(qzeros_i32: np.ndarray, size_n: int, num_bits: int = 4) -> np.ndarray:
    if qzeros_i32.ndim != 2:
        raise ValueError(f"qzeros must be 2D, got {qzeros_i32.shape}")
    if num_bits != 4:
        raise ValueError(f"Only 4-bit qzeros is supported, got {num_bits}")

    unpacked = _unpack_int4(qzeros_i32, num_bits=num_bits)
    rows, cols = unpacked.shape
    if cols != size_n:
        raise ValueError(f"qzeros unpacked cols ({cols}) != size_n ({size_n})")
    if cols % SCALE_PERM_64.size != 0:
        raise ValueError(f"qzeros cols ({cols}) must be divisible by {SCALE_PERM_64.size}")
    if cols % AWQ_INTERLEAVE.size != 0:
        raise ValueError(f"qzeros cols ({cols}) must be divisible by {AWQ_INTERLEAVE.size}")

    undo = unpacked.reshape(rows, cols // AWQ_UNDO_INTERLEAVE.size, AWQ_UNDO_INTERLEAVE.size)
    undo = undo[:, :, AWQ_UNDO_INTERLEAVE].reshape(rows, cols)

    perm = undo.reshape(rows, cols // SCALE_PERM_64.size, SCALE_PERM_64.size)
    perm = perm[:, :, SCALE_PERM_64].reshape(rows, cols)

    interleave = perm.reshape(rows, cols // AWQ_INTERLEAVE.size, AWQ_INTERLEAVE.size)
    interleave = interleave[:, :, AWQ_INTERLEAVE].reshape(rows, cols)

    return _repack_int4(np.ascontiguousarray(interleave.astype(np.uint32)), num_bits=num_bits)


def repack_qweight_to_marlin(qweight_i32: np.ndarray, size_n: int, num_bits: int = 4) -> np.ndarray:
    if qweight_i32.ndim != 2:
        raise ValueError(f"qweight must be 2D, got {qweight_i32.shape}")
    if qweight_i32.dtype != np.int32:
        raise ValueError(f"qweight dtype must be int32, got {qweight_i32.dtype}")
    if num_bits != 4:
        raise ValueError(f"Only 4-bit qweight is supported, got {num_bits}")

    size_k, packed_cols = qweight_i32.shape
    pack_factor = 32 // num_bits
    if packed_cols * pack_factor != size_n:
        raise ValueError(f"qweight packed cols ({packed_cols}) do not match size_n ({size_n})")
    if size_k % MARLIN_TILE_K != 0 or size_n % MARLIN_TILE_N != 0:
        raise ValueError(
            f"qweight size_k/size_n must be divisible by ({MARLIN_TILE_K},{MARLIN_TILE_N}), got k={size_k}, n={size_n}"
        )

    qweight_u32 = np.ascontiguousarray(qweight_i32.view(np.uint32))
    out = _repack_qweight_marlin_numba(qweight_u32, size_n=size_n, num_bits=num_bits)
    return np.ascontiguousarray(out.reshape(qweight_i32.shape).view(np.int32))


def convert_awq_triplet_to_marlin(
    qweight_ggml2d: np.ndarray,
    qzeros_ggml2d: np.ndarray,
    scales_ggml2d: np.ndarray,
    quant_bits: int,
    quant_group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qweight_hf = restore_awq_layout_to_hf_2d(np.asarray(qweight_ggml2d, dtype=np.int32))
    qzeros_hf = restore_awq_layout_to_hf_2d(np.asarray(qzeros_ggml2d, dtype=np.int32))
    scales_hf = restore_awq_layout_to_hf_2d(np.asarray(scales_ggml2d, dtype=np.float32))

    qweight_marlin = repack_qweight_to_marlin(
        qweight_hf,
        size_n=scales_hf.shape[1],
        num_bits=quant_bits,
    )
    qzeros_marlin = convert_qzeros_to_marlin(
        qzeros_hf,
        size_n=scales_hf.shape[1],
        num_bits=quant_bits,
    )
    scales_marlin = permute_scales(
        scales_hf,
        size_k=qweight_hf.shape[0],
        group_size=quant_group_size,
    )

    return qweight_marlin, qzeros_marlin, scales_marlin


def detect_awq_triplets(tensors_by_name: dict[str, Any]) -> list[AWQTriplet]:
    triplets: list[AWQTriplet] = []
    for name in tensors_by_name:
        if not name.endswith(".qweight"):
            continue
        base = name[:-len(".qweight")]
        qzeros = f"{base}.qzeros"
        scales = f"{base}.scales"
        if qzeros in tensors_by_name and scales in tensors_by_name:
            triplets.append(AWQTriplet(base=base, qweight=name, qzeros=qzeros, scales=scales))
    return triplets


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def convert_gguf_awq_to_marlin(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    quant_bits: int = 4,
    quant_group_size: int = 128,
    dry_run: bool = False,
    force: bool = False,
    arch_override: str | None = None,
) -> ConvertStats:
    if quant_bits != 4:
        raise ValueError("Only AWQ 4-bit offline conversion is currently supported")

    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not force and not dry_run:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    logger.info(f"Loading GGUF: {input_path}")
    reader = gguf.GGUFReader(input_path, "r")

    tensors_by_name = {t.name: t for t in reader.tensors}
    triplets = detect_awq_triplets(tensors_by_name)
    logger.info(f"Detected AWQ triplets: {len(triplets)}")
    if len(triplets) == 0:
        logger.warning("No .qweight/.qzeros/.scales triplets detected; output will be a plain copy")

    if dry_run:
        for tr in triplets:
            logger.info(f"Would transform triplet: {tr.base}")
        return ConvertStats(total_tensors=len(reader.tensors))

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_override if arch_override is not None else (arch_field.contents() if arch_field else None)
    if arch is None:
        raise ValueError("Cannot infer architecture metadata from input; please pass --arch")

    writer = gguf.GGUFWriter(output_path, arch=arch, endianess=reader.endianess)

    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = alignment_field.contents()

    copy_metadata(reader, writer)

    stats = ConvertStats(total_tensors=len(reader.tensors))

    converted_data_by_name: dict[str, np.ndarray] = {}
    converted_dtype_by_name: dict[str, Any] = {}

    for tr in triplets:
        qweight_t = tensors_by_name[tr.qweight]
        qzeros_t = tensors_by_name[tr.qzeros]
        scales_t = tensors_by_name[tr.scales]

        qweight_ggml = tensor_np_to_ggml_2d(qweight_t, np.asarray(qweight_t.data))
        qzeros_ggml = tensor_np_to_ggml_2d(qzeros_t, np.asarray(qzeros_t.data))
        scales_ggml = tensor_np_to_ggml_2d(scales_t, np.asarray(scales_t.data))

        qweight_marlin, qzeros_marlin, scales_marlin = convert_awq_triplet_to_marlin(
            qweight_ggml,
            qzeros_ggml,
            scales_ggml,
            quant_bits=quant_bits,
            quant_group_size=quant_group_size,
        )

        converted_data_by_name[tr.qweight] = ggml_2d_to_tensor_np_shape(qweight_t, qweight_marlin)
        converted_data_by_name[tr.qzeros] = ggml_2d_to_tensor_np_shape(qzeros_t, qzeros_marlin)
        converted_data_by_name[tr.scales] = ggml_2d_to_tensor_np_shape(scales_t, scales_marlin)

        converted_dtype_by_name[tr.qweight] = qweight_t.tensor_type
        converted_dtype_by_name[tr.qzeros] = qzeros_t.tensor_type
        converted_dtype_by_name[tr.scales] = gguf.GGMLQuantizationType.F16

        stats.transformed_triplets += 1
        stats.transformed_tensors += 3

    for tensor in reader.tensors:
        data = converted_data_by_name.get(tensor.name, tensor.data)
        raw_dtype = converted_dtype_by_name.get(tensor.name, tensor.tensor_type)
        writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, raw_dtype=raw_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        data = converted_data_by_name.get(tensor.name, tensor.data)
        writer.write_tensor_data(np.ascontiguousarray(data))

    writer.close()
    logger.info("Done.")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total tensors: {stats.total_tensors}")
    logger.info(f"Transformed triplets: {stats.transformed_triplets}")
    logger.info(f"Transformed tensors: {stats.transformed_tensors}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline convert AWQ GGUF tensors to Marlin layout (standalone converter)"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input GGUF path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output GGUF path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--quant-bits", type=int, default=4, help="AWQ quant bits (default: 4)")
    parser.add_argument("--quant-group-size", type=int, default=128, help="AWQ group size; use -1 for channel-wise")
    parser.add_argument("--arch", type=str, default=None, help="Override output arch metadata")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and print actions, do not write output")
    parser.add_argument("--force", action="store_true", help="Overwrite output if exists")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    convert_gguf_awq_to_marlin(
        input_path=args.input,
        output_path=args.output,
        quant_bits=args.quant_bits,
        quant_group_size=args.quant_group_size,
        dry_run=args.dry_run,
        force=args.force,
        arch_override=args.arch,
    )


if __name__ == "__main__":
    main()
