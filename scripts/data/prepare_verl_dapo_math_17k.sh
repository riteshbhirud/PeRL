#!/bin/bash
set -euox pipefail

python3 data/DAPO-Math-17k.py\
    --local_dataset_path "/mnt/llm-train/users/explore-train/qingyu/.cache/DAPO-Math-17k" \
    --local_dir "/mnt/llm-train/users/explore-train/qingyu/data/DAPO-Math-17k-verl"