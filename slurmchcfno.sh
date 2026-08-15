#!/bin/bash
#SBATCH --job-name=fno_gpu_train
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=ch_fno_%j.out
#SBATCH --error=ch_fno_%j.err

echo "Job started on $(hostname) at$(date)"

model load python
source env/bin/activate

echo "Python Path: $(which python)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"

python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "Starting FNO Training..."
python fno_train.py

echo "Job finished at $(date)"
