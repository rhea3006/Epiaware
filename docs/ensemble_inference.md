
# EpiAware — Ensemble Inference Documentation

## 1. Overview

This document describes the inference pipeline for the EpiAware
three-member Deep Ensemble.

The ensemble consists of three independently trained ResNet-18
models adapted for CIFAR-10 classification.

The inference module loads the trained checkpoints, generates
predictions from each ensemble member, and combines their raw
logits into a single tensor.

The individual predictions are preserved to support downstream
uncertainty estimation and analysis under distribution shift.

## 2. Ensemble Architecture

| Property | Configuration |
|---|---|
| Architecture | ResNet-18 |
| Dataset | CIFAR-10 |
| Number of classes | 10 |
| Ensemble size | 3 |
| Input channels | 3 (RGB) |
| Input resolution | 32 × 32 |
| Training device | Apple MPS |
| Inference device | MPS / CUDA / CPU |

### Ensemble Members

| Member | Random Seed | Best Epoch | Best Validation Accuracy |
|---|---:|---:|---:|
| Member 1 | 404 | 27 | 88.52% |
| Member 2 | 505 | 30 | 88.70% |
| Member 3 | 606 | 28 | 88.78% |

All three models use the same training and validation split
(split seed: 707), while maintaining independent random seeds.

## 3. Checkpoint Organization

The trained model checkpoints are stored in:

```text
checkpoints/
└── ensemble/
    ├── member_1_seed_404.pt
    ├── member_2_seed_505.pt
    └── member_3_seed_606.pt
```

Each checkpoint contains:

- `model_state_dict`: Trained model parameters.
- `best_epoch`: Epoch with the highest validation accuracy.
- `validation_accuracy`: Best validation accuracy.
- `validation_loss`: Validation loss at the best epoch.
- `seed`: Random seed used for training.
- `metadata`: Model and experiment information.

## 4. Inference Module

Implementation:

`src/epiaware/models/ensemble.py`

The module exposes two primary functions.

### 4.1 load_ensemble_models()

```python
load_ensemble_models(
    checkpoint_paths,
    device_name="auto"
)
```

Purpose:

- Loads trained ensemble checkpoints.
- Reconstructs the ResNet-18 architecture.
- Restores model parameters using `load_state_dict()`.
- Moves models to the selected device.
- Sets every model to evaluation mode.

Returns:

- A list containing the loaded models.
- The selected PyTorch device.

### 4.2 predict_ensemble_logits()

```python
predict_ensemble_logits(
    models,
    inputs,
    device
)
```

Purpose:

- Accepts a batch of CIFAR-10 images.
- Performs inference independently through each model.
- Preserves the raw logits from every ensemble member.
- Stacks individual outputs into one tensor.

Inference is performed using `torch.inference_mode()`.

## 5. Input and Output Interface

### Input

The inference function expects a four-dimensional image tensor:

```text
(N, 3, 32, 32)
```

Where:

- N = Number of images.
- 3 = RGB channels.
- 32 × 32 = Image resolution.

Images must be preprocessed using the project's CIFAR-10
evaluation transformations.

### Output

The function returns:

```text
(ensemble_size, N, num_classes)
```

For the current three-member ensemble:

```text
(3, N, 10)
```

Tensor dimensions:

| Dimension | Meaning |
|---|---|
| 0 | Ensemble member |
| 1 | Input image |
| 2 | CIFAR-10 class logits |

### Important

The inference module returns raw logits.

It does not apply:

- Softmax normalization.
- Ensemble averaging.
- Final class aggregation.

Individual model outputs are preserved because downstream
uncertainty calculations require access to the predictions
of each ensemble member.

## 6. Example Usage

```python
import torch

from epiaware.models.ensemble import (
    load_ensemble_models,
    predict_ensemble_logits,
)

checkpoint_paths = [
    "checkpoints/ensemble/member_1_seed_404.pt",
    "checkpoints/ensemble/member_2_seed_505.pt",
    "checkpoints/ensemble/member_3_seed_606.pt",
]

# Load ensemble
models, device = load_ensemble_models(checkpoint_paths)

# Assume images are a preprocessed CIFAR-10 batch
logits = predict_ensemble_logits(
    models=models,
    inputs=images,
    device=device,
)

print(logits.shape)

# Individual model predictions
predictions = logits.argmax(dim=2)
```

Expected output for a batch of 128 images:

```text
torch.Size([3, 128, 10])
```

Individual class predictions have shape:

```text
(3, 128)
```

## 7. Initial Ensemble Validation

The inference pipeline was tested using the trained checkpoints.

### Checkpoint Loading

All three models loaded successfully on Apple MPS.

Each model contains 11,173,962 parameters and was verified
to be in evaluation mode.

### Logit Shape Validation

A batch of eight validation images produced:

```text
Input shape:          (8, 3, 32, 32)
Labels shape:         (8,)
Ensemble logits:      (3, 8, 10)
```

All generated logits were finite.

### Initial Diversity Diagnostic

A separate batch of 128 validation images was used to compare
the outputs of the three ensemble members.

| Model Pair | Maximum Absolute Difference | Mean Absolute Difference |
|---|---:|---:|
| Member 1 vs 2 | 6.5664 | 1.2147 |
| Member 1 vs 3 | 6.2850 | 1.3076 |
| Member 2 vs 3 | 5.4741 | 1.2578 |

The fraction of images for which all three members predicted
the same class was 86.72%.

This is an initial batch-level diagnostic and should not be
interpreted as a dataset-wide diversity measurement.

## 8. Automated Testing

Inference tests are located in:

`tests/test_ensemble.py`

The current tests cover:

1. Correct ensemble logits dimensions.
2. Preservation of raw logits.
3. Rejection of an empty ensemble.
4. Rejection of invalid input dimensions.
5. Rejection of empty input batches.

Current project test status:

```text
28 passed
```

## 9. Integration Interface

The inference module provides the following interface to
the uncertainty evaluation component:

```text
Input:
    CIFAR-10 image batch

Input shape:
    (N, 3, 32, 32)

Output:
    Individual ensemble logits

Output shape:
    (3, N, 10)

Output type:
    Raw PyTorch logits
```

The output tensor should be passed to the uncertainty
evaluation component without averaging or applying softmax
inside the inference module.

This preserves individual member predictions for downstream
uncertainty analysis.

## 10. Current Status and Next Steps

Completed:

- ResNet-18 architecture implementation.
- Three-member ensemble training.
- Checkpoint saving and loading.
- Ensemble inference implementation.
- Raw-logit generation.
- Initial ensemble diversity diagnostic.
- Five automated ensemble inference tests.

Pending:

- Automated checkpoint-loading integration tests.
- Integration with the uncertainty evaluation component.
- Integration with the image corruption pipeline.
- Evaluation under controlled distribution shifts.
- Final evaluation using the held-out CIFAR-10 test set.

---

**Project:** EpiAware
**Component:** Model Architecture, Training and Ensemble Inference
**Ensemble Size:** 3
**Framework:** PyTorch
