#!/usr/bin/env python3
"""
Run a command while collecting hardware/system metrics.

Metrics collected:
- GPU utilization, memory, power draw, temperature from nvidia-smi
- Approximate GPU energy in joules by integrating power over time
- Process-tree CPU RSS and /proc I/O read/write bytes when available
- Wall-clock runtime and child exit code

Example:
  CUDA_VISIBLE_DEVICES=0 python3 monitor_command.py \
    --name baseline_seed43 --out-dir metrics --gpu-index 0 -- \
    python3 train.py --seed 43 ...
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, check=False):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)
    except Exception as e:
        return None


def query_gpu(gpu_index):
    cmd = [
        "nvidia-smi",
        "-i", str(gpu_index),
        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    p = run_cmd(cmd)
    if p is None or p.returncode != 0:
        return None
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 8:
        return None
    def f(x):
        try:
            return float(x)
        except Exception:
            return None
    return {
        "timestamp": parts[0],
        "gpu_index": parts[1],
        "gpu_util_pct": f(parts[2]),
        "gpu_mem_util_pct": f(parts[3]),
        "gpu_mem_used_mb": f(parts[4]),
        "gpu_mem_total_mb": f(parts[5]),
        "gpu_power_w": f(parts[6]),
        "gpu_temp_c": f(parts[7]),
    }


def get_children_pids(pid):
    p = run_cmd(["pgrep", "-P", str(pid)])
    children = []
    if p is not None and p.returncode == 0:
        for s in p.stdout.split():
            try:
                c = int(s)
                children.append(c)
                children.extend(get_children_pids(c))
            except Exception:
                pass
    return children


def read_proc_status(pid):
    status = {}
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    status["rss_kb"] = int(line.split()[1])
    except Exception:
        pass
    return status


def read_proc_io(pid):
    vals = {}
    try:
        with open(f"/proc/{pid}/io", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                k, v = line.split(":", 1)
                vals[k.strip()] = int(v.strip())
    except Exception:
        pass
    return vals


def process_tree_metrics(root_pid):
    pids = [root_pid] + get_children_pids(root_pid)
    rss_kb = 0
    read_bytes = 0
    write_bytes = 0
    syscr = 0
    syscw = 0
    alive = 0
    for pid in pids:
        if not os.path.exists(f"/proc/{pid}"):
            continue
        alive += 1
        rss_kb += read_proc_status(pid).get("rss_kb", 0)
        io = read_proc_io(pid)
        read_bytes += io.get("read_bytes", 0)
        write_bytes += io.get("write_bytes", 0)
        syscr += io.get("syscr", 0)
        syscw += io.get("syscw", 0)
    return {
        "num_pids": alive,
        "rss_mb": rss_kb / 1024.0,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "read_syscalls": syscr,
        "write_syscalls": syscw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        raise SystemExit("No command provided after --")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / f"{args.name}_metrics.csv"
    summary_json = out_dir / f"{args.name}_metrics_summary.json"

    print(f"[monitor] launching: {' '.join(args.cmd)}")
    print(f"[monitor] metrics: {metrics_csv}")

    proc = subprocess.Popen(args.cmd, preexec_fn=os.setsid)
    start = time.time()
    last_t = start
    energy_j = 0.0
    rows = []
    peak_mem_mb = 0.0
    peak_gpu_util = 0.0
    peak_rss_mb = 0.0

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "elapsed_sec", "gpu_timestamp", "gpu_index", "gpu_util_pct", "gpu_mem_util_pct",
            "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_power_w", "gpu_temp_c",
            "energy_j_est", "num_pids", "rss_mb", "read_bytes", "write_bytes",
            "read_syscalls", "write_syscalls",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            now = time.time()
            gpu = query_gpu(args.gpu_index) or {}
            proc_m = process_tree_metrics(proc.pid)

            dt = now - last_t
            power = gpu.get("gpu_power_w")
            if power is not None:
                energy_j += float(power) * max(dt, 0.0)
            last_t = now

            peak_mem_mb = max(peak_mem_mb, gpu.get("gpu_mem_used_mb") or 0.0)
            peak_gpu_util = max(peak_gpu_util, gpu.get("gpu_util_pct") or 0.0)
            peak_rss_mb = max(peak_rss_mb, proc_m.get("rss_mb") or 0.0)

            row = {
                "elapsed_sec": now - start,
                "gpu_timestamp": gpu.get("timestamp"),
                "gpu_index": gpu.get("gpu_index"),
                "gpu_util_pct": gpu.get("gpu_util_pct"),
                "gpu_mem_util_pct": gpu.get("gpu_mem_util_pct"),
                "gpu_mem_used_mb": gpu.get("gpu_mem_used_mb"),
                "gpu_mem_total_mb": gpu.get("gpu_mem_total_mb"),
                "gpu_power_w": gpu.get("gpu_power_w"),
                "gpu_temp_c": gpu.get("gpu_temp_c"),
                "energy_j_est": energy_j,
                **proc_m,
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)

            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(args.interval)

    end = time.time()
    final_io = rows[-1] if rows else {}
    summary = {
        "name": args.name,
        "cmd": args.cmd,
        "returncode": proc.returncode,
        "wall_time_sec": end - start,
        "gpu_index": args.gpu_index,
        "energy_j_est": energy_j,
        "energy_wh_est": energy_j / 3600.0,
        "peak_gpu_mem_mb": peak_mem_mb,
        "peak_gpu_util_pct": peak_gpu_util,
        "peak_process_tree_rss_mb": peak_rss_mb,
        "final_read_bytes": final_io.get("read_bytes"),
        "final_write_bytes": final_io.get("write_bytes"),
        "metrics_csv": str(metrics_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
