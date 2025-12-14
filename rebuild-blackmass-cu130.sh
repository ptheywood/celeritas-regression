#!/bin/bash -ex

cd "$(dirname "$0")"

spack env activate celeritas-cu130
module load CUDA/13.0
module load gcc/11

cd ../celeritas/build-ndebug \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
cd ../celeritas/build-ndebug-novg \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
