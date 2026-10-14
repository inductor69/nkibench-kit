#!/usr/bin/env python3
"""Correctness checker and benchmark runner for one NKIBench workspace.

  python nki_runner.py check     compile + correctness (5 seeds, L2 rel tol 2e-5, no fp16)   exit 0/1/2
  python nki_runner.py bench     check, then latency via neuron-profile (AccelOpt protocol)   prints latency_ms=...
  python nki_runner.py profile   bench plus neuron-profile counters (HBM/SBUF/PSUM bytes, spills, engine utilization)
  python nki_runner.py quick     quick estimate: compile once, one profiled run, counters; NO correctness check, no warmup

Exit codes: 0 ok, 1 incorrect, 2 compile error, 3 benchmark error.
Wraps accelopt.kernel_wrapper.NKIKernel so measurements match the published NKIBench numbers.
"""
import argparse
import json
import os
import sys

# Agent shells may not inherit the Neuron venv's PATH; neuronx-cc lives next to this interpreter.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

from accelopt import kernel_wrapper
from accelopt.kernel_wrapper import NKIKernel

FIELDS = [
    "hbm_read_bytes", "hbm_write_bytes", "psum_read_bytes", "psum_write_bytes", "sbuf_read_bytes", "sbuf_write_bytes",
    "spill_reload_bytes", "spill_save_bytes", "hardware_flops", "transpose_flops", "peak_flops_bandwidth_ratio",
    "mm_arithmetic_intensity", "hfu_estimated_percent", "scalar_engine_active_time_percent",
    "vector_engine_active_time_percent", "gpsimd_engine_active_time_percent",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "bench", "profile", "quick"])
    p.add_argument("--kernel", default="kernel.py")
    p.add_argument("--reference", default="reference.py")
    args = p.parse_args()
    if args.mode == "quick":
        return quick(args)

    if args.mode == "check":
        # NKIKernel.profile always benchmarks after the correctness check; skip that for a fast check.
        kernel_wrapper.benchmark_latency = lambda *a, **k: {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "rel_diffs": 0.0}

    # Record every profiled latency so callers get the 10 per-run samples of the batch AccelOpt keeps, not only the mean.
    seen: list = []
    batches: list = []
    orig_get_latency = kernel_wrapper.get_latency
    orig_benchmark = kernel_wrapper.benchmark_latency

    def recording_get_latency(*a, **kw):
        v = orig_get_latency(*a, **kw)
        seen.append(v)
        return v

    def recording_benchmark(*a, **kw):
        start = len(seen)
        stats = orig_benchmark(*a, **kw)
        batches.append((stats, seen[start:]))
        return stats

    kernel_wrapper.get_latency = recording_get_latency
    if args.mode != "check":
        kernel_wrapper.benchmark_latency = recording_benchmark

    k = NKIKernel(args.kernel, args.reference)
    k.profile(FIELDS if args.mode == "profile" else [])
    res, md = k.res, k.res.metadata

    if not res.compiled:
        print("COMPILE_ERROR", md.get("compilation_error", ""))
        print(md.get("compilation_traceback", ""), file=sys.stderr)
        return 2
    if not res.correct:
        print("MISMATCH", json.dumps(md.get("correctness_error", "")))
        return 1
    if args.mode == "check":
        print("OK correct on 5 seeds")
        return 0
    if "latency" not in md:
        print("BENCHMARK_ERROR", md.get("benchmarking_error", ""))
        return 3

    print(f"latency_ms={md['latency']:.6f}")
    kept = next((b for stats, b in batches if stats["mean_ms"] == md["latency"]), [])
    for v in kept:
        print(f"sample_ms={v:.6f}")
    print(f"min_ms={md['min_ms']:.6f} max_ms={md['max_ms']:.6f} rel_spread={md['rel_diffs']:.4f}")
    if args.mode == "profile":
        counters = {f: md[f] for f in FIELDS if f in md}
        print(json.dumps(counters, indent=1))
        spills = counters.get("spill_save_bytes", 0) + counters.get("spill_reload_bytes", 0)
        if spills:
            print(f"note: {spills} bytes of SBUF spill traffic")
    return 0


def quick(args) -> int:
    """One profiled run of the kernel on the reference inputs (seed 42): a cheap latency estimate for screening ideas.
    Not a correctness check and not the scoring protocol (which is `bench`)."""
    import glob
    import tempfile

    import numpy as np
    from accelopt.eval_numpy import load_module_from_path

    os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    np.random.seed(42)
    task = load_module_from_path(args.reference)
    try:
        mod = load_module_from_path(args.kernel)
        fn = getattr(mod, "kernel")
        inputs = task.transform_to_nki_inputs(task.get_inputs())
    except Exception as e:  # noqa: BLE001
        print("COMPILE_ERROR", e)
        return 2
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="nki_quick_") as d:
        try:
            lat = orig_get_latency_quick(fn, inputs, d)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            print(("COMPILE_ERROR " if "compil" in msg.lower() else "BENCHMARK_ERROR ") + msg[:2000])
            return 2 if "compil" in msg.lower() else 3
        counters = {}
        for f in glob.glob(os.path.join(d, "profile_*.json")):
            summary = json.load(open(f))
            prof = summary[next(iter(summary))]
            counters = {k: prof[k] for k in FIELDS if k in prof}
    print(f"latency_ms={lat:.6f}")
    print(f"sample_ms={lat:.6f}")
    print("note: quick estimate: 1 profiled run, no warmup, NO correctness check (use `check` for that)")
    if counters:
        print(json.dumps(counters, indent=1))
    return 0


orig_get_latency_quick = kernel_wrapper.get_latency


if __name__ == "__main__":
    sys.exit(main())
