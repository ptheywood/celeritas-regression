#!/bin/bash -ex

source source $HOME/celeritas-project/env/jadearc.sh
cd $HOME/celeritas-project/celeritas/build-ndebug \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
