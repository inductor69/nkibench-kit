#!/usr/bin/env bash
# Run one agent on one NKIBench workspace with a wall-clock budget.
#
#   run_agent.sh <codex|claude-code> <pristine-workspace> <runs-root> <seed> [model]
#
# Each run gets a fresh copy of the pristine workspace at <runs-root>/<agent>/<case>/seed<seed>/, so runs never share
# state. Run agents one at a time on the instance so measurements do not contend for the NeuronCore.
set -euo pipefail

AGENT=$1
SRC=$(cd "$2" && pwd)
ROOT=$3
SEED=$4
MODEL=${5:-}
MINUTES=${MINUTES:-30}
REASONING=${REASONING:-high}   # use the same effort for every agent
CASE=$(basename "$SRC")
RUN="$ROOT/$AGENT/$CASE/seed$SEED"

rm -rf "$RUN"
mkdir -p "$(dirname "$RUN")"
cp -a "$SRC" "$RUN"
cd "$RUN"
TASK=$(cat TASK.md)
START=$(date +%s)

case "$AGENT" in
	codex)
		# Codex's own sandbox can block /dev/neuron*; treat the benchmark instance as the sandbox. score.py catches tampering.
		timeout "${MINUTES}m" codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
			${MODEL:+-m "$MODEL"} -c model_reasoning_effort="$REASONING" --json "$TASK" >transcript.log 2>&1 || true
		;;
	claude-code)
		# Edit files and run the NKIBench runner only.
		timeout "${MINUTES}m" claude -p "$TASK" --permission-mode acceptEdits \
			--allowedTools "Read" "Edit" "Write" "Glob" "Grep" "Bash(python nki_runner.py:*)" \
			${MODEL:+--model "$MODEL"} --output-format stream-json --verbose >transcript.log 2>&1 || true
		;;
	*)
		echo "unknown agent: $AGENT (expected codex or claude-code)" >&2
		exit 2
		;;
esac

END=$(date +%s)
git diff --stat >diff.stat 2>/dev/null || true
git diff >final.diff 2>/dev/null || true
printf '{"agent":"%s","case":"%s","seed":%s,"model":"%s","reasoning":"%s","minutes_budget":%s,"wall_seconds":%s}\n' \
	"$AGENT" "$CASE" "$SEED" "$MODEL" "$REASONING" "$MINUTES" "$((END - START))" >run.json
echo "done: $RUN ($((END - START))s)"
