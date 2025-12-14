#!/bin/bash -ex

cd "$(dirname "$0")"

spack env activate celeritas-cu126
module load CUDA/12.6
module load gcc/11

cd ../celeritas-cu126/build-ndebug \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
cd ../celeritas-cu126/build-ndebug-novg \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
