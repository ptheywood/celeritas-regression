#!/bin/bash -ex

source $DATA/shareing-r1/env/jadearc.sh
cd $DATA/shareing-r1/celeritas/build-ndebug-novg \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4

