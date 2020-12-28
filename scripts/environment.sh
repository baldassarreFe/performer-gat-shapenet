#!/usr/bin/env bash

conda install -y -n base -c conda-forge mamba

mamba create -y -n mg -c conda-forge -c pytorch \
  python=3.8 \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  tqdm \
  ffmpeg \
  cudatoolkit=10.2 \
  pytorch \
  pytorch-lightning \
  nodejs \
  jupyterlab \
  ipywidgets \
  jinja2 \
  jupyter_console \
  jupyterlab_code_formatter \
  black \
  isort \
  pylint \
  rope \
  yaml

conda activate mg

pip install \
  wandb \
  coolname \
  omegaconf \
  git+https://github.com/idiap/fast-transformers

CUDA=cu102
TORCH=1.7.0
URL="https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html"
pip install -f "${URL}" torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install torch-geometric

jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager
jupyter labextension install --no-build @jupyterlab/toc
jupyter labextension install --no-build @ryantam626/jupyterlab_code_formatter
jupyter lab build
jupyter serverextension enable --py jupyterlab_code_formatter