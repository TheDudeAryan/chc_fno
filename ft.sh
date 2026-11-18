#!/bin/bash
#SBATCH --job-name=fno_train
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=ft_%j.out
#SBATCH --error=ft_%j.err

# Train one model on Work/data/test_t<END>.npz (END defaults to 500; the
# file is made once per END with ch_gpu.sh).
#
#   sbatch --export=ALL,END=500 ft.sh                     Laplacian output (lap)
#   sbatch --export=ALL,END=500,VARIANT=base ft.sh        the plain head
#   sbatch --export=ALL,END=500,CHANNELS=24,MODES=8 ft.sh    a smaller model
#   sbatch --export=ALL,END=500,SEED=1 ft.sh              a seed replicate
#   sbatch --export=ALL,END=1000 ft.sh                   a longer horizon
#   sbatch --export=ALL,END=500,VARIANT=base,NOHARD=1 ft.sh                 the ablation, soft penalty 0.1
#   sbatch --export=ALL,END=500,VARIANT=base,NOHARD=1,MASS_WEIGHT=0,TAG=free ft.sh   no penalty: train_t0500_nohard_free
#   sbatch --export=ALL,END=500,NOMU=1,TAG=nomu ft.sh   no mu channel: train_t0500_lap_nomu
#
# VARIANT=lap (default) writes Work/models/train_t<END>_lap/; VARIANT=base
# writes Work/models/train_t<END>/.  CHANNELS (default 38) and MODES (default
# 24) set the model size (17.8M parameters); a non-default value adds a tag
# to the folder, e.g. Work/models/train_t0500_lap_c24m8/.  SEED (default 0)
# is the training seed; a non-zero value adds s<N>, e.g.
# Work/models/train_t0500_lap_s1/.  A job stopped by the time limit exits
# with code 75: resubmit the same command to continue from its checkpoint.
# NOHARD=1 adds --no-hard-constraints (no zero DC, no final mean removal; the
# folder gets _nohard), MASS_WEIGHT=W adds --mass-weight W (needs NOHARD=1;
# the checks are in fno_train.py), TAG=NAME adds --tag NAME and NOMU=1 adds
# --no-mu-channel (needs TAG; the folder's name does not say it).  Unset, the
# command is the one above.
#
# The model takes a step of any length (dt-conditioned) and is validated on
# the 10 validation seeds to t = 10000, whatever END is (the references are
# the shared cache of fno_predict.py: Work/gt_cache, or $FNO_GT_CACHE).
set -euo pipefail

END=${END:-500}
VARIANT=${VARIANT:-lap}
CHANNELS=${CHANNELS:-38}
MODES=${MODES:-24}
SEED=${SEED:-0}
# optional switches; unset adds nothing to the command
EXTRA=()
if [ "${NOHARD:-}" = "1" ]; then EXTRA+=(--no-hard-constraints); fi
if [ -n "${MASS_WEIGHT:-}" ]; then EXTRA+=(--mass-weight "$MASS_WEIGHT"); fi
if [ -n "${TAG:-}" ]; then EXTRA+=(--tag "$TAG"); fi
if [ "${NOMU:-}" = "1" ]; then EXTRA+=(--no-mu-channel); fi
# the dataset ch_gpu.sh writes for this END: test_t0500.npz
DATA=Work/data/test_t$(printf %04d "$END").npz

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= TRAINING (END=$END, VARIANT=$VARIANT, CHANNELS=$CHANNELS, MODES=$MODES, SEED=$SEED) ======================="
python fno_train.py --dataset "$DATA" --t-train "$END" --variant "$VARIANT" --channels "$CHANNELS" --modes "$MODES" --seed "$SEED" ${EXTRA[@]+"${EXTRA[@]}"}

echo
echo "Job finished at $(date)"
