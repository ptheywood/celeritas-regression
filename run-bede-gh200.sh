#!/bin/bash -e
#SBATCH -A bdshe19
#SBATCH -t 03:59:59
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J celer-regression
#SBATCH -o bede-gh200-%J.out
#SBATCH -e bede-gh200-%J.err

if [ -z "$SLURM_JOB_ID" ]; then
  set -x
  sbatch $0 && squeue -u $USER
  exit $?
fi

# (un)load modules and activate spack environment 
source /nobackup/projects/bdshe19/aarch64/env/bede-gh200.sh 2> /dev/null

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py bede
echo "Completed at $(date)"
exit 0
