# Training azc on a cloud machine (AWS c7a.48xlarge, or a Vast.ai machine with many CPU cores)

1. **Pack** (on your PC, Git Bash, in `C:\balatro_rl`):
   `bash tools/cloud/pack.sh` packs the code and the live run's checkpoint (`checkpoints/azc_v5.pt`, at its last logged iteration) into `build/azc_cloud.tgz`.
2. **Rent the machine**
   - **Vast.ai**: add credit, set an SSH key (Account → Keys), then Search: filter **CPU cores ≥ 128**, sort by price. On the offer click **Rent**, with the template / image
     `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, **disk 50 GB**, launch mode **SSH**. The instance card's **>_** button shows its SSH command (`ssh -p PORT root@HOST`).
   - **AWS**: an Ubuntu 24.04 instance, e.g. c7a.48xlarge, as Spot with the interruption behaviour **Stop** (keeps the disk), 50 GB disk, your key pair, SSH open to your IP.
3. **Copy and set up** (Vast: `-P PORT` for scp / `-p PORT` for ssh, user `root`; AWS: `-i KEY.pem`, user `ubuntu`):
   ```
   scp -P PORT build/azc_cloud.tgz root@HOST:~
   ssh -p PORT root@HOST
   mkdir -p ~/balatro_rl && tar xzf azc_cloud.tgz -C ~/balatro_rl
   bash ~/balatro_rl/tools/cloud/setup.sh
   cd ~/balatro_rl && nohup bash tools/cloud/run.sh >> checkpoints/run.out 2>&1 &
   ```
   With a GPU the network trains on it and the self-play workers stay on the CPU cores.
   If you rented only part of a Vast machine (the offer lists e.g. 256 of 512 cores), start with `CORES=256 nohup bash tools/cloud/run.sh ...`.
4. **Watch**: `python tools/last_iters.py checkpoints/azc_cloud_log.jsonl 5` (AWS: after `. ~/venv/bin/activate`). The first iterations show the real speed: check `play_s`.
   `tail checkpoints/run.out` shows the start line (cores, workers, games, device) and any error.
5. **Fetch results** to your PC: `scp -P PORT root@HOST:~/balatro_rl/checkpoints/azc_cloud* checkpoints/`
6. **Stop paying when done**: Vast: **destroy** the instance (a stopped one still bills its disk, and may not get its GPUs/cores back); AWS: stop or terminate it.

`run.sh` resumes from the last checkpoint after a crash. On AWS `setup.sh` registers it to start at boot; on Vast put the start line in the instance's On-start script.
Tune with environment variables: `WORKERS=24 GAMES=768 bash tools/cloud/run.sh`.
