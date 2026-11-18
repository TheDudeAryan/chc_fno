#!/bin/bash
#SBATCH --job-name=fno_predict
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=07:15:00
#SBATCH --output=fp_%j.out
#SBATCH --error=fp_%j.err

# Predict and draw with an already trained model: the same END, VARIANT,
# CHANNELS, MODES and SEED as its ft.sh job pick Work/models/<label>/fno_ch.pt
# (defaults 500, lap, 38, 24, 0, so train_t0500_lap).
#
#   sbatch --export=ALL,END=500 fp.sh                predict to T_END = 10000
#   sbatch --export=ALL,END=300,VARIANT=base fp.sh
#   sbatch --export=ALL,END=500,CHANNELS=24,MODES=8 fp.sh
#   sbatch --export=ALL,END=500,SEED=1 fp.sh
#   sbatch --export=ALL,END=500,T_END=20000 fp.sh    predict to 20000
#   sbatch --export=ALL,END=1000 fp.sh               the model trained to 1000
#
# Results go to Results/<label>_pred_t<T_END>/, drawn by fno_predict.py; the
# last step only refreshes Results/README.md and the comparison images
# (make_submission.py --overview).  The first prediction of a reference set
# also solves the references (Work/gt_cache, or $FNO_GT_CACHE: about 860 GB
# in 800 files for the 15 cases, and about 8.5 h of solver time on gpu1, an
# estimate from the one-job timing of 2026-11-12, progress 4w): that can use
# up the time limit, and a resubmission goes on from the references,
# schedules and cases already done.
set -euo pipefail

END=${END:-500}
VARIANT=${VARIANT:-lap}
CHANNELS=${CHANNELS:-38}
MODES=${MODES:-24}
SEED=${SEED:-0}
T_END=${T_END:-10000}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

LABEL=$(python fno_train.py --t-train "$END" --variant "$VARIANT" --channels "$CHANNELS" --modes "$MODES" --seed "$SEED" --print-label)
echo "======================= PREDICTION ($LABEL, T_END=$T_END) ======================="
python fno_predict.py --model "Work/models/$LABEL/fno_ch.pt" --views "$T_END"

echo
echo "======================= RESULTS ======================="
python make_submission.py --overview

echo
echo "Job finished at $(date)"
