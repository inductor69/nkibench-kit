# NKIBench evaluation kit

Scripts to run coding agents on [NKIBench](https://github.com/zhang677/AccelOpt) (AWS Trainium NKI kernels from LLM
workloads) and score them the same way: a numerical correctness check, latency by the AccelOpt protocol, and an
independent re-measurement of every final kernel.

The baseline kernels and their reference implementations come from AccelOpt's NKIBench. `setup_workspaces.py` copies
them into one workspace per kernel.

## Files

- `nki_runner.py` runs in each workspace:
  - `check`: compile, then correctness on 5 random seeds (L2 relative error < 2e-5; float16/bfloat16 not allowed).
  - `bench`: `check`, then latency via neuron-profile (AccelOpt protocol: 2 warmups, mean of 10 profiled runs).
  - `profile`: `bench` plus counters (HBM/SBUF/PSUM bytes, spills, engine utilization).
  - `quick`: one profiled run for screening. No correctness check.
  - Exit codes: 0 ok, 1 incorrect, 2 compile error, 3 benchmark error.
- `setup_workspaces.py` builds one git-initialized workspace per kernel with:
  - `kernel.py`, the baseline kernel and the only file an agent may edit;
  - `reference.py`, `nki_runner.py`, `baseline.json`;
  - `TASK.md`, the prompt, identical for every agent.
  - The theoretical lower bound comes from AccelOpt's peak model.
- `run_agent.sh` runs Codex or Claude Code on a fresh copy of a workspace under a wall-clock budget. It saves the transcript, the diff and the timing.
- `score.py` re-measures every final kernel in fresh processes:
  - The result is the median of 3 `bench` runs.
  - The baseline is re-measured on the same machine.
  - A tampered protected file, an incorrect kernel, or a result faster than the theoretical bound scores 1.0×.
  - It writes `results.csv` and a per-agent summary (geomean speedup, % of peak).

## Setup (trn1.2xlarge, Neuron Deep Learning AMI)

```bash
source /opt/aws_neuronx_venv_pytorch_2_7/bin/activate     # match your AMI's Neuron venv
git clone https://github.com/zhang677/AccelOpt ~/AccelOpt
(cd ~/AccelOpt && pip install pydantic && python setup.py install)

python setup_workspaces.py --accelopt ~/AccelOpt --out ~/nkib-ws --minutes 30

# Smoke test: the baseline kernel must pass and benchmark.
(cd ~/nkib-ws/silu_M4096_N7168 && python nki_runner.py check && python nki_runner.py bench)
```

## Run agents and score

```bash
export MINUTES=30 REASONING=high
for seed in 1 2 3; do
  for case in matmul_K5120_M4096_N12288 silu_M4096_N7168 transpose_matmul_K2048_M4096_N10944; do
    ./run_agent.sh codex       ~/nkib-ws/$case ~/nkib-runs $seed <openai-model>
    ./run_agent.sh claude-code ~/nkib-ws/$case ~/nkib-runs $seed <claude-model>
  done
done

NKI_PYTHON=$(which python) python score.py --workspaces ~/nkib-ws --runs ~/nkib-runs --repeats 3
```

`ls ~/nkib-ws` lists every kernel case. Run one agent at a time per instance.

## Notes

- Seeds are repeated runs of the same agent and task. Report the median across seeds and the spread, not a single run.
- Latency varies between runs and sessions. Run a few seeds of an unmodified kernel through `score.py` to see the noise floor before you compare agents.
- Check the CLI flags in `run_agent.sh` against your installed Codex and Claude Code versions (`--help`).

Kernels, references, and the peak model are from AccelOpt (Apache License 2.0); see NOTICE.
