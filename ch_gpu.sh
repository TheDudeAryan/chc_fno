#!/bin/bash
#SBATCH --job-name=ch_gpu
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=ch_gpu_%j.out
#SBATCH --error=ch_gpu_%j.err

# The training dataset to END, as in the group protocol: Work/data/test_t<END>.npz, in
# ch.py's format (train_X, train_y, val_X, val_y), the same trajectories and
# frames as ch.py.  ft.sh and ftp.sh train on the file of their END.
#
#   sbatch ch_gpu.sh                          END = 500 -> test_t0500.npz
#   sbatch --export=ALL,END=1000 ch_gpu.sh    -> test_t1000.npz
#   sbatch --export=ALL,END=300 ch_gpu.sh     -> test_t0300.npz
set -euo pipefail

END=${END:-500}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= DATASET (END=$END) ======================="
python ch_gpu.py --end "$END"

echo
echo "Job finished at $(date)"
