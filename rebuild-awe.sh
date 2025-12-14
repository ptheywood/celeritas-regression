#!/bin/bash -ex

cd "$(dirname "$0")"

spack env activate celeritas-dev-rocm

cd ../celeritas/build-ndebug-novg \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
# cd ../celeritas/build-ndebug-novg-fp32 \
#   && cmake -UCeleritas_GIT_DESCRIBE . \
#   && ninja celer-sim celer-g4
