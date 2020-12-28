#!/usr/bin/env bash

# Hyperparameter sweep over optimizer parameters

conda activate mg
export CUDA_VISIBLE_DEVICES=0,2

for architecture in performer gat; do
for learning_rate in 1e-3 1e-4; do
for weight_decay in 1e-3 1e-5; do
python -m miniproject train 
  'conf/default.yaml' \
  architecture.model=${architecture} \
  optimizer.learning_rate=${learning_rate} \
  optimizer.weight_decay=${weight_decay} \
  logging.group='optimizer' \
  logging.tags="[${architecture}]" \
  other.gpus=2
done
done
done