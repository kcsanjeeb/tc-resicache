#!/usr/bin/env python3
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

def run_cmd(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return None

def query_gpu(gpu_index):
    cmd = ["nvidia-smi", "-i", str(gpu_index),
           "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
           "--format=csv,noheader,nounits"]
    p = run_cmd(cmd)
    if p is None or p.returncode != 0 or not p.stdout.strip():
        return {}
    parts = [x.strip() for x in p.stdout.strip().splitlines()[-1].split(",")]
    if len(parts) < 8:
        return {}
    def f(x):
        try: return float(x)
        except Exception: return None
    return {"timestamp": parts[0], "gpu_index": parts[1], "gpu_util_pct": f(parts[2]),
            "gpu_mem_util_pct": f(parts[3]), "gpu_mem_used_mb": f(parts[4]),
            "gpu_mem_total_mb": f(parts[5]), "gpu_power_w": f(parts[6]), "gpu_temp_c": f(parts[7])}

def get_children(pid):
    p = run_cmd(["pgrep", "-P", str(pid)])
    out = []
    if p and p.returncode == 0:
        for s in p.stdout.split():
            try:
                c = int(s); out.append(c); out.extend(get_children(c))
            except Exception:
                pass
    return out

def proc_metrics(root):
    pids = [root] + get_children(root)
    rss = rb = wb = 0
    alive = 0
    for pid in pids:
        if not os.path.exists(f"/proc/{pid}"): continue
        alive += 1
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss += int(line.split()[1])
        except Exception:
            pass
        try:
            with open(f"/proc/{pid}/io") as f:
                vals = {}
                for line in f:
                    k,v = line.split(":",1)
                    vals[k.strip()] = int(v.strip())
                rb += vals.get("read_bytes",0)
                wb += vals.get("write_bytes",0)
        except Exception:
            pass
    return {"num_pids": alive, "rss_mb": rss/1024.0, "read_bytes": rb, "write_bytes": wb}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        raise SystemExit("No command after --")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{args.name}_metrics.csv"
    json_path = out / f"{args.name}_metrics_summary.json"
    proc = subprocess.Popen(cmd)
    start = last = time.time()
    energy = peak_mem = peak_util = peak_rss = 0.0
    last_row = {}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["elapsed_sec","gpu_timestamp","gpu_index","gpu_util_pct","gpu_mem_used_mb","gpu_power_w","gpu_temp_c","energy_j_est","num_pids","rss_mb","read_bytes","write_bytes"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        while True:
            now = time.time()
            gpu = query_gpu(args.gpu_index)
            pm = proc_metrics(proc.pid)
            dt = now - last; last = now
            power = gpu.get("gpu_power_w")
            if power is not None: energy += power * max(dt, 0.0)
            peak_mem = max(peak_mem, gpu.get("gpu_mem_used_mb") or 0.0)
            peak_util = max(peak_util, gpu.get("gpu_util_pct") or 0.0)
            peak_rss = max(peak_rss, pm.get("rss_mb") or 0.0)
            row = {"elapsed_sec": now-start, "gpu_timestamp": gpu.get("timestamp"), "gpu_index": gpu.get("gpu_index"),
                   "gpu_util_pct": gpu.get("gpu_util_pct"), "gpu_mem_used_mb": gpu.get("gpu_mem_used_mb"),
                   "gpu_power_w": gpu.get("gpu_power_w"), "gpu_temp_c": gpu.get("gpu_temp_c"),
                   "energy_j_est": energy, **pm}
            w.writerow(row); f.flush(); last_row = row
            if proc.poll() is not None: break
            time.sleep(args.interval)
    summary = {"name": args.name, "cmd": cmd, "returncode": proc.returncode, "wall_time_sec": time.time()-start,
               "gpu_index": args.gpu_index, "energy_j_est": energy, "energy_wh_est": energy/3600.0,
               "peak_gpu_mem_mb": peak_mem, "peak_gpu_util_pct": peak_util, "peak_process_tree_rss_mb": peak_rss,
               "final_read_bytes": last_row.get("read_bytes"), "final_write_bytes": last_row.get("write_bytes"),
               "metrics_csv": str(csv_path)}
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
