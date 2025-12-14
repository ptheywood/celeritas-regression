#!/bin/bash -e

spack env activate celeritas-dev-rocm

module list 2>&1
amd-smi list
lscpu

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py awe
echo "Completed at $(date)"
exit 0
