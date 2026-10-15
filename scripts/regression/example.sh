#!/bin/sh

set -e

experiments=()
echo Training...
for fold in {1..5}; do
    experiments+=("reg_resnet18_224_finetune_none_$fold")
    python -m nako_eye.regression.train --backbone resnet18 --method finetune --imagesize 224 --batchsize 256 --epochs 25 --kfold "${fold}" --numworkers 8 --device cuda:0
done

python -m nako_eye.regression.validate --experiments "${experiments[@]}" --batchsize 256 --numworkers 8 --device cuda:0
python -m nako_eye.regression.eval --experiments "${experiments[@]}" --batchsize 256 --numworkers 8 --device cuda:0
