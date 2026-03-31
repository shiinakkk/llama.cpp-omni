#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_ncu_top10_tmp.py <ncu_csv_path>")
        return 2

    path = sys.argv[1]
    rows = []
    with open(path, "r", newline="") as f:
        for line in f:
            if line.startswith('"'):
                rows.append(line)

    reader = csv.reader(rows)
    header = None
    for row in reader:
        if row and row[0] == "ID":
            header = row
            break

    if header is None:
        print("header not found")
        return 1

    idx_kernel = header.index("Kernel Name")
    idx_time = header.index("gpu__time_duration.sum")

    by_time_us = defaultdict(float)
    by_count = defaultdict(int)
    for row in reader:
        if len(row) <= max(idx_kernel, idx_time):
            continue
        kernel = row[idx_kernel].strip()
        val = row[idx_time].strip().replace(",", "")
        if not kernel or not val:
            continue
        try:
            ns = float(val)
        except ValueError:
            continue
        by_time_us[kernel] += ns / 1000.0
        by_count[kernel] += 1

    total_us = sum(by_time_us.values())
    print(f"TOTAL_US,{total_us:.3f}")
    print("RANK,CALLS,TOTAL_US,TOTAL_MS,SHARE,KERNEL")
    for i, (kernel, us) in enumerate(sorted(by_time_us.items(), key=lambda kv: kv[1], reverse=True)[:10], 1):
        share = us / total_us * 100 if total_us else 0.0
        print(f"{i},{by_count[kernel]},{us:.3f},{us/1000.0:.3f},{share:.2f}%,{kernel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
