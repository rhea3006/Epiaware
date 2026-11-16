# EpiAware – Model Training Methodology

## 1. Objective

Establish a reproducible training pipeline for a deep ensemble of ResNet-18 classifiers on CIFAR-10. The trained ensemble will be used to investigate uncertainty estimation and its reliability under distribution shift.

## 2. Dataset

- Dataset: CIFAR-10
- Number of classes: 10
- Training samples: 45,000
- Validation samples: 5,000
- Official test samples: 10,000
- Training augmentation: Random Crop and Random Horizontal Flip
- Evaluation preprocessing: Tensor conversion and CIFAR-10 normalization

The official test set is reserved for final evaluation.

## 3. Model Architecture

- Architecture: ResNet-18
- Input: 32 × 32 RGB images
- Initial convolution: 3 × 3 kernel, stride 1
- Initial max pooling: Removed
- Output classes: 10
- Initialization: Independent random initialization for each ensemble member

## 4. Training Infrastructure

- Framework: PyTorch
- Hardware: Apple M2
- Acceleration: Metal Performance Shaders (MPS)
- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 0.0001
- Batch size: 128
- Loss function: Cross Entropy Loss
- Checkpoint selection: Best validation accuracy

## 5. Hyperparameter Experiments

Three training configurations were evaluated using seed 303.

| Experiment | Learning rate | Scheduler | Epochs | Best validation accuracy |
|---|---:|---|---:|---:|
| Baseline | 0.001 | None | 20 | 77.30% |
| Learning-rate comparison | 0.01 | None | 20 | 86.30% |
| Scheduler experiment | 0.01 | StepLR | 30 | 88.34% |

### Selected configuration

- Initial learning rate: 0.01
- Scheduler: StepLR
- Step size: 10
- Gamma: 0.1
- Total epochs: 30

Learning-rate schedule:

| Epochs | Learning rate |
|---|---:|
| 1–10 | 0.01 |
| 11–20 | 0.001 |
| 21–30 | 0.0001 |

The scheduler experiment achieved the highest validation accuracy of 88.34%.

Note: The scheduler experiment also used a longer training duration than the fixed-learning-rate comparison. Therefore, the observed improvement cannot be attributed exclusively to the scheduler.

## 6. Ensemble Training Plan

Three independently initialized ResNet-18 models will be trained sequentially.

Proposed seeds:

- Member 1: 404
- Member 2: 505
- Member 3: 606

Each member will use the selected training configuration and retain its best validation checkpoint.

## 7. Reproducibility

- Random seeds are explicitly configured.
- Dataset splitting is reproducible.
- Experiment configurations are stored as YAML files.
- Training histories are stored as JSON files.
- Model checkpoints are stored separately.
- The official test set remains excluded from model selection.

## 8. Current Status

Completed:
- CIFAR-10 data pipeline
- ResNet-18 implementation
- Training and validation engine
- Checkpoint saving and restoration
- Learning-rate comparison
- StepLR implementation and verification
- Training protocol selection

Next:
- Ensemble training orchestration
- Ensemble prediction generation
- Integration with uncertainty evaluation
