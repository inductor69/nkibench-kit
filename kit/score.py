#!/usr/bin/env python3
"""Re-measure every run's final kernel.py and summarize per arm.

  python score.py --workspaces ~/nkib-ws --runs ~/nkib-runs [--repeats 3] [--out results.csv]

For each run: tamper check on protected files, correctness check, then `nki_runner.py bench` repeated in fresh processes
(median). Baselines are re-measured on this machine from the pristine workspaces. An incorrect or tampered final
kernel, or one measured faster than the theoretical bound, scores as the baseline (effective speedup 1.0).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

PROTECTED = ["reference.py", "nki_runner.py", "TASK.md", "baseline.json"]
LAT = re.compile(r"latency_ms=([0-9.]+)")
PY = os.environ.get("NKI_PYTHON", "python")  # the Neuron venv interpreter


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def run(cmd: str, cwd: Path, timeout: int = 1800) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def measure(ws: Path, repeats: int) -> tuple[float | None, str]:
    code, out = run(f"{PY} nki_runner.py check", ws)
    if code != 0:
        return None, {1: "incorrect", 2: "compile_error", 124: "timeout"}.get(code, f"error{code}")
    lats = []
    for _ in range(repeats):
        code, out = run(f"{PY} nki_runner.py bench", ws)
        m = LAT.search(out)
        if code != 0 or not m:
            return None, "benchmark_error"
        lats.append(float(m.group(1)))
    return statistics.median(lats), "ok"


def gmean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspaces", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("results.csv"))
    args = ap.parse_args()

    baselines: dict[str, dict] = {}
    rows = []
    for run_dir in sorted(args.runs.glob("*/*/seed*")):
        arm, case = run_dir.parts[-3], run_dir.parts[-2]
        pristine = args.workspaces / case
        meta = json.loads((pristine / "baseline.json").read_text())
        if case not in baselines:
            lat, status = measure(pristine, args.repeats)
            if lat is None:
                raise SystemExit(f"pristine baseline failed for {case}: {status}")
            if meta["theoretical_bound_ms"] / lat > 1.0:
                raise SystemExit(f"pristine baseline for {case} ({lat:.4f} ms) beats the theoretical bound; check timing or the peak model")
            baselines[case] = {"latency": lat}
            print(f"baseline {case}: {lat:.4f} ms")
        base = baselines[case]["latency"]

        tampered = [f for f in PROTECTED if sha(run_dir / f) != sha(pristine / f)]
        lat, status = (None, "tampered") if tampered else measure(run_dir, args.repeats)
        if lat is not None and meta["theoretical_bound_ms"] / lat > 1.0:
            # Faster than the hardware bound: skipped work or broken timing, never a win.
            lat, status = None, "faster_than_bound"
        valid = lat is not None
        speedup = base / lat if valid else None
        effective = speedup if valid else 1.0
        best_latency = lat if valid else base
        info = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
        rows.append({
            "arm": arm, "case": case, "seed": run_dir.name, "status": status, "tampered": ",".join(tampered),
            "baseline_ms": round(base, 5), "final_ms": round(lat, 5) if valid else "",
            "speedup": round(speedup, 4) if valid else "", "effective_speedup": round(effective, 4),
            "pct_peak": round(meta["theoretical_bound_ms"] / best_latency, 4),
            "baseline_pct_peak": round(meta["theoretical_bound_ms"] / base, 4),
            "accelopt_best_speedup": meta.get("accelopt_best_speedup"),
            "wall_seconds": info.get("wall_seconds"), "model": info.get("model"),
        })
        print(f"{arm:14s} {case:42s} {run_dir.name:6s} {status:14s} x{effective:.3f}")

    if not rows:
        raise SystemExit(f"no runs found under {args.runs} (expected <arm>/<case>/seed<N>/)")
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    print(f"\n{'arm':14s} runs  geomean_speedup  geomean_%peak  improved(>2%)  incorrect  tampered  median_wall_s")
    for arm, rs in sorted(by_arm.items()):
        eff = [r["effective_speedup"] for r in rs]
        peak = [r["pct_peak"] for r in rs]
        walls = [r["wall_seconds"] for r in rs if r["wall_seconds"] is not None]
        print(f"{arm:14s} {len(rs):4d}  {gmean(eff):15.3f}  {gmean(peak) * 100:12.1f}%  {sum(e > 1.02 for e in eff):13d}  "
              f"{sum(r['status'] in ('incorrect', 'compile_error', 'benchmark_error') for r in rs):9d}  "
              f"{sum(bool(r['tampered']) for r in rs):8d}  {statistics.median(walls) if walls else float('nan'):13.0f}")
    ref = [r for r in rows if r["accelopt_best_speedup"]]
    if ref:
        uniq = {r["case"]: r["accelopt_best_speedup"] for r in ref}
        print(f"\nreference: AccelOpt repo run (gpt-oss-120b) geomean best speedup {gmean(list(uniq.values())):.3f} over {len(uniq)} cases")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
