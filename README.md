# Performer vs. Graph Attention Networks on ShapeNet

_Mini-Project on Graph Networks and Architectures for the Deep Learning Advanced course at KTH, Stockholm (DD2412)._

A performer-based architecture and a graph attention network are compared for point cloud classification on the ShapeNet dataset. 
GradCAM is used to investigate model differences by explaining the most important points for a prediction.

## Setup
Main dependencies:
- numpy, pandas, matplotlib, seaborn
- pytorch, torch_geometric
- pytorch lightning
- fast transformers
- weights and biases
- jupyter

Dependencies are managed using Conda and Mamba
```bash
bash scripts/environment.sh
pip install -e .
```

ShapeNet data is downloaded to `./data` on the first run.

## Code Structure
```
miniproject
├── __init__.py             
├── __main__.py              Main entrypoint for conf/train/test tasks
├── configuration.py         Configuration parsing and management
├── datamodules.py           ShapeNet dataset
├── metrics.py               Additional lighning metrics
├── models                   
│   ├── __init__.py          
│   ├── common.py            Lightning model wrapper
│   ├── performer.py         Performer model
│   └── graph_attention.py   Graph Attention model
├── train.py                 Training code
└── test.py                  Evaluation code
```
## Commands

### Configuration
Trainin configuration is managed using `yaml` files and command-line overrides.
To list the available options and verify a parsed configuration one can run:
```bash
python -m miniproject conf conf/quick.yaml logging.tags='[bbb,ccc]' trainer.max_epochs=2
```
```yaml
trainer:
  max_epochs: 2
# ...
logging:
  run_name: smart-flounder-of-honeydew
  tags:
  - bbb
  - ccc
# ...
```

### Training
A quick training run can be launched with:
```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m miniproject train conf/quick.yaml other.gpus=2
```

Hyperparameter sweeps can be launched using the scripts in `./scripts`, for example:
```bash
bash scripts/sweep-architecture.sh
```

### Evaluation
Given a model checkpoint, evaluation metrics on the test set can be computed with:
```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m miniproject test path/to/checkpoint
```

We provide the following checkpoints:
- Performer model: `runs/miniproject/myrtle-gorilla-of-culture/checkpoints/last.ckpt`
- Graph Attention model: `runs/miniproject/vivacious-honest-lion/checkpoints/last.ckpt`

## Notebooks
The following notebooks were developed alongside the code.
Each notebooks is accompanied by output figures/videos in the corresponding folder.
```
notebooks
├── Data/ShapenetDataset.ipynb              Dataset statistics
├── Hyperparameters/Hyperparameters.ipynb   Architecture hyperparameter sweep
├── Testing/Results.ipynb                   Test results and GradCAM explanations
└── PytorchHooks.ipynb                      How to make pytorch hooks work
```
