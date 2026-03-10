#!/bin/sh

set -e

experiments=()
echo Training...
for fold in {1..5}; do
    experiments+=("clf_resnet18_224_finetune_$fold")
    python -m classification.train --backbone resnet18 --method finetune --imagesize 224 --batchsize 256 --epochs 25 --kfold "${fold}" --numworkers 8 --device cuda:0
done

python -m classification.val --experiments "${experiments[@]}" --batchsize 256 --numworkers 8 --device cuda:0
python -m classification.eval --experiments "${experiments[@]}" --batchsize 256 --numworkers 8 --device cuda:0
