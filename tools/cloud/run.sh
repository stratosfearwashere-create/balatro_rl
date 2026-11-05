#!/bin/bash
# Run ON THE CLOUD MACHINE: trains azc with parallel self-play workers on the CPU (and the network on the GPU if there is one),
# and resumes by itself after a crash or a restart (from the newest checkpoint and the last logged iteration).
#   cd ~/balatro_rl && nohup bash tools/cloud/run.sh >> checkpoints/run.out 2>&1 &
#   progress:  python tools/last_iters.py checkpoints/azc_cloud_log.jsonl 5
# A new run beside the old one: RUN=azc_big START_CKPT=checkpoints/azc_big_start.pt START_ITER=135 HORIZON=3 SNAP_FRAC=0.25 PASSES=6 bash tools/cloud/run.sh
cd ~/balatro_rl
. ~/.azc_env
RUN=${RUN:-azc_cloud}                         # the run's name: checkpoints/$RUN.pt and checkpoints/${RUN}_log.jsonl
START_CKPT=${START_CKPT:-checkpoints/azc_start.pt}
CORES=${CORES:-$(nproc)}                      # in a container that sees the whole host but rents part of it, set CORES to the rented count
WORKERS=${WORKERS:-$(( CORES / 8 ))}          # ~8 cores per worker: 24 on a 192-vCPU c7a.48xlarge, 48 on a 384-thread machine
GAMES=${GAMES:-$(( WORKERS * 32 ))}           # ~32 games per worker per iteration (each worker's batch must be large enough to use its cores)
BUFFER=${BUFFER:-$(( GAMES * 40 * 15 ))}      # the window: ~15 iterations of samples (~40 per game)
PASSES=${PASSES:-3}                           # training passes over one iteration's samples
TRAIN_BATCHES=${TRAIN_BATCHES:-$(( GAMES * 40 * PASSES / 256 ))}
HORIZON=${HORIZON:-2}                         # blinds the search looks ahead
SNAP_FRAC=${SNAP_FRAC:-0}                     # share of the games that start from a saved ante-5+ state
BETA_FLOOR=${BETA_FLOOR:-0}                   # the exact value keeps at least this share in the search
LR=${LR:-3e-4}                                # the learning rate
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then DEVICE=cuda; else DEVICE=cpu; fi
while true; do
    if [ -f checkpoints/$RUN.pt ] && [ -s checkpoints/${RUN}_log.jsonl ]; then
        INIT=checkpoints/$RUN.pt
        START=$(tail -1 checkpoints/${RUN}_log.jsonl | python -c "import json,sys; print(json.loads(sys.stdin.read())['iter'])")
    else
        INIT=$START_CKPT
        START=${START_ITER:-$(cat checkpoints/start_iter.txt 2>/dev/null || echo 0)}
    fi
    echo "$(date) $RUN starting from $INIT at iteration $START: $CORES cores, $WORKERS workers, $GAMES games per iteration, window $BUFFER, $TRAIN_BATCHES batches, horizon $HORIZON, mid-run starts $SNAP_FRAC, beta floor $BETA_FLOOR, lr $LR, learner on $DEVICE"
    python -u -m balatro_rl.azc --device "$DEVICE" --worker-device cpu --cores "$CORES" \
        --init "$INIT" --start-iter "$START" --out checkpoints/$RUN.pt --iters 3000 \
        --games "$GAMES" --workers "$WORKERS" --rollout-node-cap 50 --beta-iters 150 --horizon "$HORIZON" --snap-frac "$SNAP_FRAC" --beta-floor "$BETA_FLOOR" --lr "$LR" \
        --buffer-cap "$BUFFER" --train-batches "$TRAIN_BATCHES" --eval-every 10 --eval-games 100
    echo "$(date) trainer exited ($?); restarting in 10 s"
    sleep 10
done
