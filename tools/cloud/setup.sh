#!/bin/bash
# Run ON THE CLOUD MACHINE after copying azc_cloud.tgz to the home directory:
#   mkdir -p ~/balatro_rl && tar xzf azc_cloud.tgz -C ~/balatro_rl
#   bash ~/balatro_rl/tools/cloud/setup.sh
# Works on a plain Ubuntu VM (AWS: ubuntu user, sudo) and in a Vast.ai container (root, a PyTorch image).
# Installs what is missing, builds the C++ engine, and registers run.sh to start at boot where cron exists.
set -e
cd ~/balatro_rl
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y build-essential
if python -c "import torch" 2>/dev/null; then
    # a PyTorch image (Vast.ai): use its Python as it is
    echo "# the image's own Python (it has PyTorch)" > ~/.azc_env
else
    $SUDO apt-get install -y python3-venv python3-dev
    python3 -m venv ~/venv
    . ~/venv/bin/activate
    pip install --upgrade pip
    if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then pip install torch
    else pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
    echo ". ~/venv/bin/activate" > ~/.azc_env
fi
. ~/.azc_env
pip install numpy pybind11 cython setuptools pyyaml
python setup_cython.py build --only balatro_rl/sim/cpp/engine/module.cpp
python -c "from balatro_rl.engine_env import CppVecEnv; v = CppVecEnv(4, 'RED', 'GOLD', seed0=1, threads=0); print('engine OK')"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (training on CPU)')"
mkdir -p checkpoints
[ -f checkpoints/azc_start.pt ] || mv azc_start.pt start_iter.txt checkpoints/ 2>/dev/null || true
if command -v crontab >/dev/null; then
    (crontab -l 2>/dev/null | grep -v azc_run; echo "@reboot cd ~/balatro_rl && nohup bash tools/cloud/run.sh >> checkpoints/run.out 2>&1 # azc_run") | crontab -
else
    echo "(no cron here: on Vast.ai, put this in the instance's On-start script to resume after a restart:"
    echo "   cd ~/balatro_rl && nohup bash tools/cloud/run.sh >> checkpoints/run.out 2>&1 & )"
fi
echo "setup done. Start now with:  cd ~/balatro_rl && nohup bash tools/cloud/run.sh >> checkpoints/run.out 2>&1 &"
