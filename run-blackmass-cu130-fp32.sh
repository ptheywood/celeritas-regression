#!/bin/bash -e

spack env activate celeritas-cu130
module load CUDA/13.0
module load gcc/11

module list 2>&1
nvidia-smi
lscpu

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py blackmass-cu130-fp32
echo "Completed at $(date)"
exit 0
