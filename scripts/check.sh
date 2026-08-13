#!/usr/bin/env bash
# One-command local verification for dreamerv3-latent-stability.
# Runs the 22 transcription tests, then the CPU smoke pipeline (gates + audit
# with the wall probe). Exit non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; fi

echo "== 1/2 transcription tests =="
"$PY" -m pytest tests/ -q

echo "== 2/2 CPU smoke (audit + wall probe) =="
KMP_DUPLICATE_LIB_OK=TRUE "$PY" experiments/d1_sampling_gap.py --quick --dump-conds /tmp/d1_check_conds.npy

echo
echo "OK: tests pass and the smoke pipeline runs end to end."
echo "Note: the smoke uses random weights; its numbers are never quoted."
