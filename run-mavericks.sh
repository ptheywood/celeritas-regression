#!/bin/bash -e

spack env activate celeritas-cu126
module load CUDA/12.6
module load gcc/11

module list 2>&1
nvidia-smi
lscpu

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py mavericks
echo "Completed at $(date)"
exit 0
