#!/bin/bash
#SBATCH --job-name=fno_ch
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=08:00:00
#SBATCH --output=ftp_%j.out
#SBATCH --error=ftp_%j.err

# Train, predict to T_END (default 10000) and draw one model: ft.sh and
# fp.sh in one job, with the same variables, then refresh the Results index.
#
#   sbatch --export=ALL,END=500 ftp.sh                    Laplacian output (lap)
#   sbatch --export=ALL,END=500,VARIANT=base ftp.sh       the plain head
#   sbatch --export=ALL,END=500,CHANNELS=24,MODES=8 ftp.sh    a smaller model
#   sbatch --export=ALL,END=500,SEED=1 ftp.sh             a seed replicate
#   sbatch --export=ALL,END=1000 ftp.sh                  a longer horizon
#
# The model goes to Work/models/<label>/ and the results to
# Results/<label>_pred_t10000/ (label train_t0500_lap by default).  A
# training stopped by the time limit exits with code 75 and this job ends
# there, so nothing is predicted from a half-trained model.  Resubmit the
# same command to go on: a training that has finished trains nothing more,
# and a prediction cut short reuses what it had written (the references it
# solved, the schedules it tuned, the cases it finished).  The run's own
# Results/ folder is drawn by fno_predict.py; the last step only refreshes
# Results/README.md and the comparison images (make_submission.py --overview).
set -euo pipefail

END=${END:-500}
VARIANT=${VARIANT:-lap}
CHANNELS=${CHANNELS:-38}
MODES=${MODES:-24}
SEED=${SEED:-0}
T_END=${T_END:-10000}
# the dataset ch_gpu.sh writes for this END: test_t0500.npz
DATA=Work/data/test_t$(printf %04d "$END").npz

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= TRAINING (END=$END, VARIANT=$VARIANT, CHANNELS=$CHANNELS, MODES=$MODES, SEED=$SEED) ======================="
python fno_train.py --dataset "$DATA" --t-train "$END" --variant "$VARIANT" --channels "$CHANNELS" --modes "$MODES" --seed "$SEED"

LABEL=$(python fno_train.py --t-train "$END" --variant "$VARIANT" --channels "$CHANNELS" --modes "$MODES" --seed "$SEED" --print-label)
echo
echo "======================= PREDICTION ($LABEL, T_END=$T_END) ======================="
python fno_predict.py --model "Work/models/$LABEL/fno_ch.pt" --views "$T_END"

echo
echo "======================= RESULTS ======================="
python make_submission.py --overview

echo
echo "Job finished at $(date)"
