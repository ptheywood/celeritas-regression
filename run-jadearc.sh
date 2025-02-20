#!/bin/bash -e
#SBATCH -A jade-beta
#SBATCH -t 3:59:59
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:1
#SBATCH -J celer-regression
#SBATCH -o jade-arc-%J.out
#SBATCH -e jade-arc-%J.err
#SBATCH -p medium

if [ -z "$SLURM_JOB_ID" ]; then
  set -x
  sbatch $0 && squeue -u $USER
  exit $?
fi

# Load modules + activate spack environment
source $HOME/celeritas-project/env/jadearc.sh 2> /dev/null

echo "Running on $HOSTNAME at $(date)"
module list 2>&1
python3 run-problems.py jadearc
echo "Completed at $(date)"
exit 0
