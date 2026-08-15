#!/bin/bash
#SBATCH --job-name=ch_dataset_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8          
#SBATCH --mem=32G                  
#SBATCH --time=04:00:00            
#SBATCH --output=ch_gen_%j.out
#SBATCH --error=ch_gen_%j.err

echo "Job started on $(hostname) at$(date)"

if [ -f "./env/bin/activate" ]; then
source ./env/bin/activate
elif [ -f "$HOME/env/bin/activate" ]; then
source "$HOME/env/bin/activate"
else
echo "Error: Virtual environment 'env' not found!"
exit 1
fi

echo "Python Path: $(which python)"
echo "Python Version: $(python --version)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting dataset generation with ch.py..."
python ch.py

echo "Job finished at $(date)"
