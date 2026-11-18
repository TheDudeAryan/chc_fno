#!/bin/bash
#SBATCH --job-name=hpc_storage
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=hs_%j.out
#SBATCH --error=hs_%j.err

# Used and available storage, and the quota, as a compute node sees them.
# It only reports: nothing is deleted or moved.  Submit it from the project
# root; arguments after the script name go straight to hpc_storage.py:
#
#   sbatch hpc_storage.sh                      standard folders and the quota
#   sbatch hpc_storage.sh /path/a /path/b      and these paths too
#   sbatch hpc_storage.sh --du                 and the size of Work, Results
#                                              and Archive
#   sbatch hpc_storage.sh --du Work/gt_cache   and the size of those paths
#
# `du` reads every file, which is slow on a shared filesystem and unwelcome
# on a login node, so ask for it here and not by hand.  /tmp and $TMPDIR are
# local to the node, so the job reports the compute node's, not the login
# node's.
#
# The partition is the one the group's job scripts use for CPU-only jobs
# (`sinfo` lists the partitions); change it if this account needs another.
# Output: hs_<jobid>.out, next to this script

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

module load python

echo "Python Path: $(which python3)"

python3 hpc_storage.py "$@"

echo
echo "Job finished at $(date)"
