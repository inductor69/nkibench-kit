# NKIBench kit

Run coding agents (Codex, Claude Code) on [NKIBench](https://github.com/zhang677/AccelOpt) kernels on AWS Trainium and
score them the same way: one numerical correctness check, latency measured by the AccelOpt protocol, and an independent
re-measurement of every final kernel.

## What's included

NKIBench has 14 kernels. The published results cover 3 of them:

| Kernel | Type | Baseline | Theoretical bound | Baseline % of peak |
|---|---|---|---|---|
| `matmul_K5120_M4096_N12288` | compute-bound matmul | 31.54 ms | 21.70 ms | 68.8% |
| `silu_M4096_N7168` | memory-bound elementwise | 1.215 ms | 0.534 ms | 43.9% |
| `transpose_matmul_K2048_M4096_N10944` | matmul starting close to its bound | 9.475 ms | 7.73 ms | 81.6% |

- **Codex results:** 3 independent runs per kernel at a 30-minute budget, 9 runs in total. Every run has its final
  kernel, its diff, and its metadata in `results/kernels/`. Summary in `results/README.md`.
- **Scripts:** `setup_workspaces.py` builds all 14 kernels by default, so the kit runs on the full suite, not only these 3.
- **Bounds:** the baselines above were measured on our instance. Re-measure on yours, since latency varies a few
  percent between sessions. The bound comes from AccelOpt's peak model.

## Repository layout

```
kit/
  setup_workspaces.py   builds one git-initialized workspace per kernel from a local AccelOpt checkout
  nki_runner.py         the only tool agents run: check / bench / profile / quick
  run_agent.sh          runs Codex or Claude Code on a fresh copy of a workspace under a wall-clock budget
  score.py              re-measures every final kernel and writes results.csv plus a per-agent summary
  README.md             details on each script
  NOTICE                AccelOpt attribution (Apache 2.0)
results/
  README.md             Codex results summary (30-minute budget, 3 runs per kernel)
  results.csv           every run, as written by score.py
  kernels/<case>/seed<N>/
    kernel.py           final kernel the agent left
    final.diff          diff against the baseline kernel
    run.json            agent, model, reasoning effort, budget, wall time
```

Each workspace that `setup_workspaces.py` creates contains:

| File | Purpose |
|---|---|
| `kernel.py` | Baseline kernel. The only file an agent may edit |
| `reference.py` | Reference implementation used for the correctness check |
| `nki_runner.py` | Compile, check, benchmark, profile |
| `baseline.json` | Baseline latency and the theoretical bound from AccelOpt's peak model |
| `TASK.md` | The prompt, identical for every agent |

## Grading

- **Correctness** (`nki_runner.py check`): compile, then compare against `reference.py` on 5 random seeds. L2 relative
  error must be below 2e-5. float16/bfloat16 are not allowed.
- **Latency** (`nki_runner.py bench`): AccelOpt protocol, 2 warmups, then the mean of 10 profiled runs.
- **Scoring** (`score.py`):
  - Re-measures the baseline from the pristine workspaces on the same machine.
  - Runs each final kernel in fresh processes and takes the median of `--repeats` bench runs (default 3).
  - Scores as 1.0x (no gain) any final kernel that is incorrect, that changed a protected file
    (`reference.py`, `nki_runner.py`, `TASK.md`, `baseline.json`), or that measures faster than the theoretical bound.
  - Speedup = baseline latency / final latency.

## Running it

Requirements:
- A trn1.2xlarge with the Neuron Deep Learning AMI. One NeuronCore is used per kernel.
- The `codex` and/or `claude` CLI installed and logged in.

### 1. Set up

```bash
source /opt/aws_neuronx_venv_pytorch_2_7/bin/activate      # match your AMI's Neuron venv
git clone https://github.com/zhang677/AccelOpt ~/AccelOpt
(cd ~/AccelOpt && pip install pydantic && python setup.py install)

git clone https://github.com/inductor69/nkibench-kit ~/nkibench-kit
cd ~/nkibench-kit/kit
python setup_workspaces.py --accelopt ~/AccelOpt --out ~/nkib-ws --minutes 30
```

`--minutes` is the budget stated in `TASK.md`. Keep it equal to `MINUTES` in step 3. Use `--cases` to build only some
kernels. `ls ~/nkib-ws` lists the cases.

### 2. Smoke test

The baseline kernel must pass and benchmark:

```bash
(cd ~/nkib-ws/silu_M4096_N7168 && python nki_runner.py check && python nki_runner.py bench)
```

### 3. Run agents

Run one agent at a time per instance, so runs do not contend for the NeuronCore.

```bash
export MINUTES=30 REASONING=high
for seed in 1 2 3; do
  for case in matmul_K5120_M4096_N12288 silu_M4096_N7168 transpose_matmul_K2048_M4096_N10944; do
    ./run_agent.sh codex       ~/nkib-ws/$case ~/nkib-runs $seed <openai-model>
    ./run_agent.sh claude-code ~/nkib-ws/$case ~/nkib-runs $seed <claude-model>
  done
done
```

Each run writes to `~/nkib-runs/<agent>/<case>/seed<N>/`:
- the agent's final `kernel.py`;
- `transcript.log`;
- `final.diff`;
- `run.json`.

### 4. Score

```bash
NKI_PYTHON=$(which python) python score.py --workspaces ~/nkib-ws --runs ~/nkib-runs --repeats 3 --out results.csv
```

This prints a line per run and a per-agent summary, and writes `results.csv` in the same format as
`results/results.csv`. The summary reports geomean speedup, geomean % of peak, runs improved by more than 2%,
incorrect runs, tampered runs, and median wall time.

### Re-scoring the published Codex kernels

To check our numbers without rerunning Codex, copy the published kernels into run directories and score them:

```bash
for d in ~/nkibench-kit/results/kernels/*/seed*; do
  case=$(basename "$(dirname "$d")"); seed=$(basename "$d")
  run=~/nkib-runs-published/codex/$case/$seed
  mkdir -p "$(dirname "$run")" && cp -a ~/nkib-ws/$case "$run" && cp "$d/kernel.py" "$run/kernel.py"
done
NKI_PYTHON=$(which python) python score.py --workspaces ~/nkib-ws --runs ~/nkib-runs-published --repeats 3
```

## Notes

- **Seeds** are repeated runs of the same agent on the same task. Report the median across seeds and the spread, not
  a single run.
- **Noise:** latency varies a few percent between runs and sessions. Always compare against a baseline re-measured on
  your instance. Score a few unmodified kernels first to see the noise floor.
- **CLI flags:** check the flags in `run_agent.sh` against your installed Codex and Claude Code versions (`--help`).

## License

Kernels, reference implementations, and the peak model come from AccelOpt (Apache License 2.0). They are copied from
your local AccelOpt checkout, not redistributed here. See `kit/NOTICE`.
