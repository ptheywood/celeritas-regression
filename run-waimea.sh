#!/bin/bash -e

module load CUDA/12.6
spack env activate celeritas-cu126

module list 2>&1
nvidia-smi
lscpu

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py waimea
echo "Completed at $(date)"
exit 0
