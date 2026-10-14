# Codex results on NKIBench (trn1.2xlarge)

Codex (gpt-5.6-luna, reasoning high), 30-minute budget per run, 3 independent runs per kernel. Every final kernel was
re-measured by `kit/score.py` (median of 3 `bench` runs, baseline re-measured on the same machine).

| Kernel | Baseline | Median speedup | Range | % of peak (baseline -> median run) |
|---|---|---|---|---|
| matmul_K5120_M4096_N12288 | 31.54 ms | 1.073x | 0.994-1.096x | 68.8% -> 73.8% |
| silu_M4096_N7168 | 1.215 ms | 1.721x | 1.720-1.730x | 43.9% -> 75.6% |
| transpose_matmul_K2048_M4096_N10944 | 9.475 ms | 1.000x | 1.000-1.029x | 81.6% -> 81.6% |

Geomean of per-kernel medians: 1.23x. All 9 runs passed correctness; none modified protected files.

- `results.csv`: every run, as written by `score.py`.
- `kernels/<case>/seed<N>/`: the final `kernel.py`, its diff against the baseline, and run metadata.

Latency varies a few percent between runs and sessions; re-measure the baseline on your instance before comparing.
