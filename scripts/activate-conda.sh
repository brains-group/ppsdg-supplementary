#!/bin/bash

# conda env should be specified in the .env of the project root
source .env
# run hook and activate
eval "$("{$CONDA_DIR}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
conda activate $CONDA_ENV
