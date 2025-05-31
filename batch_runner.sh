#!/bin/bash
# batch_runner.sh

source /data/koppikarps/conda/etc/profile.d/conda.sh
conda activate pitstop

python batch_main.py "$1"

