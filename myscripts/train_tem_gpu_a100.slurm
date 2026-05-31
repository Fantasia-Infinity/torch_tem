#!/bin/bash
#SBATCH --job-name=tem-tszh
#SBATCH --account=vusr117232
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./output/slurm-logs/%x-%j.out
#SBATCH --error=./output/slurm-logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shufan.z.t@gmail.com
#
# Tolman-Eichenbaum Machine (TEM) training on one A100 (Snellius).
#
# Submit:
#   mkdir -p output/slurm-logs
#   cd /home/szhang2/AlchemyRepos/torch_tem
#   sbatch myscripts/train_tem_gpu_a100.slurm
#
# Override train_it via env:
#   TRAIN_IT=50000 sbatch myscripts/train_tem_gpu_a100.slurm

set -euo pipefail

mkdir -p output/slurm-logs

module load 2023
module load CUDA/12.4.0
echo "nvcc: $(command -v nvcc)"
nvcc --version | head -n 1

source /home/szhang2/venvs/torch-tem-py311/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd /home/szhang2/AlchemyRepos/torch_tem

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA required for this job"
print("device", torch.cuda.get_device_name(0))
PY

TRAIN_IT="${TRAIN_IT:-20000}"

exec python -u myscripts/run_gpu.py --train-it "${TRAIN_IT}"
