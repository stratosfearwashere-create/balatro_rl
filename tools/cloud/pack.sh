#!/bin/bash
# Run on YOUR PC (Git Bash), from C:\balatro_rl: packs the code and a checkpoint for the cloud machine.
#   bash tools/cloud/pack.sh [checkpoint]          (default: checkpoints/azc_v5.pt, the live run, at its last logged iteration)
# -> build/azc_cloud.tgz  (copy it to the machine: scp -i KEY.pem build/azc_cloud.tgz ubuntu@HOST:~ )
set -e
cd "$(dirname "$0")/../.."
CK="${1:-checkpoints/azc_v5.pt}"
[ -f "$CK" ] || { echo "no checkpoint found (pass one)"; exit 1; }
LOG="${CK%.pt}_log.jsonl"
if [ -f "$LOG" ]; then START_ITER=$(tail -1 "$LOG" | python -c "import json,sys; print(json.loads(sys.stdin.read())['iter'])")
else START_ITER=$(echo "$CK" | grep -o 'at[0-9]*' | tr -d 'at'); fi
mkdir -p build
cp "$CK" build/azc_start.pt
echo "${START_ITER:-0}" > build/start_iter.txt
tar czf build/azc_cloud.tgz --exclude='__pycache__' --exclude='*.pyd' --exclude='*.so' --exclude='*.c' \
    balatro_rl setup_cython.py requirements.txt tools/cloud tools/last_iters.py -C build azc_start.pt start_iter.txt
echo "build/azc_cloud.tgz ($(du -h build/azc_cloud.tgz | cut -f1)) from $CK, start iteration ${START_ITER:-0}"
