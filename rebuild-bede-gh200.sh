#!/bin/bash -ex

source /nobackup/projects/bdshe19/aarch64/env/bede-gh200.sh
cd /nobackup/projects/bdshe19/aarch64/celeritas-project/celeritas/build-reldeb \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
cd /nobackup/projects/bdshe19/aarch64/celeritas-project/celeritas/build-reldeb-novg \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
