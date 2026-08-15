#!/bin/bash
#SBATCH --job-name=fno_gpu_evo
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=fno_predict_%j.out
#SBATCH --error=fno_predict_%j.err

echo "Job started on $(hostname) at$(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"

python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "Starting Evolution..."
python fno_predict.py

echo "Job finished at $(date)"
