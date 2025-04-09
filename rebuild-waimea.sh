#!/bin/bash -ex

cd "$(dirname "$0")"

cd ../celeritas/build-reldeb \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
# cd ../celeritas/build-reldeb-vecgeom \
#   && cmake -UCeleritas_GIT_DESCRIBE . \
#   && ninja celer-sim celer-g4
