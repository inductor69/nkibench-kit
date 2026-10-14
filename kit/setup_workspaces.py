#!/usr/bin/env python3
"""Build one pristine, git-initialized workspace per NKIBench kernel.

Each workspace holds kernel.py (the baseline kernel, the only file an agent may edit), reference.py, nki_runner.py,
TASK.md (identical for every agent), and baseline.json. Kernels and references come from AccelOpt's NKIBench.

  python setup_workspaces.py --accelopt ~/AccelOpt --out ~/nkib-ws [--minutes 60] [--cases matmul,silu]
"""
import argparse
import ast
import csv
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PEAK_BW_BYTES = 410 * 1024**3  # AccelOpt's Trn1 per-core peaks
PEAK_MM_FLOPS = 23.75e12
PEAK_VEC_FLOPS = 2 * 128 * 1.12e9
PROTECTED = ["reference.py", "nki_runner.py", "TASK.md", "baseline.json"]


def load_work_model(accelopt: Path):
    """Return work(problem, config) -> (vec_flops, mm_flops, bytes) from AccelOpt's own peak formulas.

    The script runs its analysis at import time, so we extract calc_theoretical_peak via the AST and make it return
    the raw work terms instead of the bound latency.
    """
    src = (accelopt / "experiments/full_complete_local/calculate_percentage_of_peak.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "calc_theoretical_peak")
    idx = next(i for i, s in enumerate(fn.body) if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "latency")
    ret = ast.parse("return (vec_flops, mm_flops, total_bytes)").body[0]
    fn.body = fn.body[:idx] + [ret]
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {}
    exec(compile(mod, "calc_theoretical_peak", "exec"), ns)
    return ns["calc_theoretical_peak"]


def case_name(problem: str, values: dict) -> str:
    return problem + "_" + "_".join(f"{k}{values[k]}" for k in sorted(values))


def accelopt_baselines(accelopt: Path) -> dict:
    path = accelopt / "experiments/full_complete_local/profile_results.csv"
    out = {}
    for row in csv.DictReader(path.open()):
        out[row["case_name"]] = json.loads(row["profile"])["latency"]
    return out


def accelopt_best(accelopt: Path) -> dict:
    path = accelopt / "experiments/full_complete_local/results/11110904_best_plan_id.csv"
    return {r["case_name"]: float(r["best_speedup"]) for r in csv.DictReader(path.open())} if path.exists() else {}


TASK = """# Task: optimize an NKI kernel for AWS Trainium 1 (one NeuronCore)

Make `kernel` in `kernel.py` as fast as possible while producing the same result as `forward` in `reference.py`.

Correctness (checked by `python nki_runner.py check`): 5 random seeds, L2 relative error < 2e-5, float16/bfloat16 not allowed.
Speed (measured by `python nki_runner.py bench`): mean latency over profiled runs (AccelOpt/NKIBench protocol).
Counters (from `python nki_runner.py profile`): HBM/SBUF/PSUM bytes, spills, engine utilization.
Quick estimate (from `python nki_runner.py quick`): one profiled run plus counters, no correctness check, no warmup; much cheaper than `bench` for comparing ideas. The scorer uses the full `bench` protocol.

Rules
- Edit only `kernel.py`. Do not modify `reference.py`, `nki_runner.py`, `TASK.md`, or `baseline.json`; tampering invalidates the run.
- Keep the kernel entry point named `kernel` with the same signature.
- Stop within {minutes} minutes. When you stop, `kernel.py` must hold your best correct kernel; the scorer re-measures it.

Reference points (for orientation; the scorer measures the baseline on this machine)
- Case: {case} ({problem}, {values})
- Baseline latency reported by AccelOpt on Trn1: {baseline_ms} ms
- Theoretical lower bound from AccelOpt's peak model: {bound_ms:.4f} ms ({bound_key}-bound; peaks: 23.75 TFLOP/s matmul, 286.8 GFLOP/s vector+scalar, 410 GiB/s HBM)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelopt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--cases", default="", help="Comma-separated problem names; default all")
    args = ap.parse_args()

    bench = args.accelopt / "NKIBench"
    summary = json.loads((bench / "summary.json").read_text())
    work = load_work_model(args.accelopt)
    baselines = accelopt_baselines(args.accelopt)
    best = accelopt_best(args.accelopt)
    wanted = set(filter(None, args.cases.split(",")))
    args.out.mkdir(parents=True, exist_ok=True)
    index = []

    for problem, entry in summary.items():
        if wanted and problem not in wanted:
            continue
        for case in entry["cases"].values():
            values = case["values"]
            impl = case["impls"][0]
            name = case_name(problem, values)
            ws = args.out / name
            if ws.exists():
                shutil.rmtree(ws)
            ws.mkdir()
            shutil.copy(bench / impl["kernel"], ws / "kernel.py")
            shutil.copy(bench / impl["task"], ws / "reference.py")
            shutil.copy(HERE / "nki_runner.py", ws / "nki_runner.py")

            vec, mm, nbytes = work(problem, values)
            terms = {"vec": vec / PEAK_VEC_FLOPS, "mm": mm / PEAK_MM_FLOPS, "memory": nbytes / PEAK_BW_BYTES}
            bound_key = max(terms, key=terms.get)
            bound_ms = terms[bound_key] * 1e3
            baseline = {
                "case": name, "problem": problem, "values": values,
                "accelopt_baseline_ms": baselines.get(name), "accelopt_best_speedup": best.get(name),
                "theoretical_bound_ms": bound_ms, "bound_key": bound_key,
                "work": {"vec_flops": vec, "mm_flops": mm, "bytes": nbytes},
            }
            (ws / "baseline.json").write_text(json.dumps(baseline, indent=1) + "\n")
            (ws / "TASK.md").write_text(TASK.format(
                minutes=args.minutes, case=name, problem=problem, values=values,
                baseline_ms=baselines.get(name, "unknown"), bound_ms=bound_ms, bound_key=bound_key,
            ))

            (ws / ".gitignore").write_text("__pycache__/\n")

            subprocess.run("git init -q && git add -A && git -c user.email=bench@local -c user.name=bench commit -qm pristine",
                           shell=True, cwd=ws, check=True)
            index.append(baseline)
            print(f"{name:45s} bound {bound_ms:8.4f} ms ({bound_key})  accelopt baseline {baselines.get(name)} ms")

    (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"{len(index)} workspaces in {args.out}")


if __name__ == "__main__":
    main()
