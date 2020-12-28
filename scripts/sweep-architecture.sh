#!/usr/bin/env bash

# Hyperparameter sweep over architecture parameters

conda activate mg
export CUDA_VISIBLE_DEVICES=0,2

for architecture in performer gat; do
for num_layers in 2 4; do
for num_heads in 2 4; do
for hidden_features in 8 16; do
python -m miniproject train \
  'conf/default.yaml' \
  architecture.model=${architecture} \
  architecture.num_layers=${num_layers} \
  architecture.num_heads=${num_heads} \
  architecture.hidden_features=${hidden_features} \
  logging.group='architecture' \
  logging.tags="[${architecture}]" \
  other.gpus=2
done
done
done
done
