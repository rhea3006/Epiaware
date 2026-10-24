# Contributing to EpiAware

## 1. Project Overview

EpiAware investigates when deep ensemble uncertainty stops reliably identifying individual classification errors under distribution shift.

Our initial experimental setup uses CIFAR-10, ResNet-18, deep ensembles, and controlled image corruptions at multiple severity levels.

### Research question

Under what types and severities of distribution shift does deep ensemble uncertainty stop predicting individual classification errors?

All contributions should support this research question and maintain reproducible, testable experiments.

## 2. Team Responsibilities

### Research Lead — Model Architecture and Ensemble Training

**Primary ownership:** `src/epiaware/models/`, `src/epiaware/training/`, and the core training pipeline.

Responsibilities:
- Implement ResNet-18 adapted for CIFAR-10.
- Develop training and validation loops.
- Implement checkpoint saving and loading.
- Coordinate independently initialized ensemble members.
- Maintain reproducible seeds, device selection, and training configuration.
- Document training decisions and experiment settings.
- Coordinate integration and resolve shared-interface decisions with the team.

Expected deliverables:
- A working single-model training pipeline.
- Reproducible training for three ensemble members.
- Saved checkpoints and training/validation metrics.
- Tests for model output shapes and essential training utilities.

### Pratham — Training Experiments and Uncertainty Evaluation

**Primary ownership:** `src/epiaware/evaluation/`, evaluation scripts, and experimental analysis.

Responsibilities:
- Implement predictive entropy and ensemble disagreement.
- Implement prediction-error identification and error-detection AUROC/AUPRC.
- Implement calibration metrics and risk–coverage evaluation.
- Build the single-model baseline experiment runner.
- Compare results across model seeds, corruption types, and severity levels.
- Generate result tables, plots, and analysis for the research paper.
- Coordinate prediction interfaces with the Research Lead.

Expected deliverables:
- Tested uncertainty and evaluation functions.
- A reproducible evaluation pipeline.
- Structured experiment outputs.
- Plots and analysis answering the research question.

### Stuti — Distribution Shifts and Data Validation

**Primary ownership:** `src/epiaware/data/corruptions.py`, corruption tests, and data-validation documentation.

Responsibilities:
- Implement Gaussian noise, blur, and rotation corruptions.
- Implement the agreed five severity levels for each corruption.
- Preserve labels and image dimensions.
- Validate image ranges, data types, and corruption reproducibility.
- Generate clean-versus-corrupted image grids.
- Document the corruption definitions and data-validation process.
- Raise issues whenever a specification or test expectation is unclear.

Expected deliverables:
- A tested corruption pipeline.
- Tests for shape, labels, valid values, reproducibility, and severity parameters.
- Visual examples of each corruption and severity.
- `docs/data_validation.md` describing the implemented pipeline and validation results.

Stuti's tasks should have explicit specifications and independently runnable tests. She is not expected to independently design the deep learning architecture or uncertainty methodology.

## 3. Branching Strategy

Use the following branches for the initial workstreams:

| Branch | Owner | Scope |
|---|---|---|
| `main` | Entire team via pull requests | Stable, integrated code |
| `feature/model-training` | Research Lead | Architecture and training |
| `feature/uncertainty-evaluation` | Pratham | Baselines, metrics, evaluation |
| `feature/data-shifts` | Stuti | Corruptions and data validation |

Create branches from the latest `main`. Keep changes focused on the assigned workstream.

Before starting work:

```bash
git switch main
git pull origin main
git switch -c feature/your-workstream
```

If your assigned branch already exists, switch to it instead of creating it again.

## 4. Development Workflow

1. Pull the latest changes from `main`.
2. Work on your assigned feature branch.
3. Implement the functionality and its tests together.
4. Run the relevant tests before committing.
5. Commit changes with a descriptive message.
6. Push the branch and open a pull request.
7. Request review from at least one teammate.
8. Address review feedback and merge only after the required checks pass.

Example commit messages:

```text
feat: add CIFAR-10 image corruptions
feat: implement ensemble uncertainty metrics
feat: add ResNet-18 training loop
test: validate corruption severity parameters
docs: document data validation procedure
```

Do not commit downloaded datasets, model checkpoints, API keys, virtual environments, or generated experiment outputs unless the team explicitly agrees that a particular artifact belongs in version control.

## 5. Testing Requirements

Run the test suite before opening a pull request:

```bash
pytest -v
```

New functionality should include relevant tests. Tests should be deterministic wherever practical and should not require a full training run or dataset download unless explicitly designated as an integration test.

At minimum:
- Data utilities must validate expected shapes, labels, and values.
- Model utilities must validate output dimensions and checkpoint behavior.
- Evaluation functions must be checked against known or synthetic examples.
- Reproducibility-sensitive code must use explicit seeds where applicable.

The existing baseline is 17 passing tests. Preserve existing behavior unless a change is intentional, justified, and covered by tests.

## 6. Documentation Responsibilities

Each contributor must document their own component.

- **Research Lead:** training configuration, architecture choices, seeds, checkpoints, and training procedure.
- **Pratham:** metric definitions, baseline methodology, evaluation procedure, and interpretation of results.
- **Stuti:** corruption parameters, severity definitions, validation checks, and example images.

Keep the following project documents updated as applicable:
- `README.md`: project overview, setup, and instructions to run the pipeline.
- `CONTRIBUTING.md`: team workflow and contribution rules.
- `docs/data_validation.md`: data and corruption validation.
- `docs/experiments.md`: experiment protocol, configuration, and reproducibility notes.
- `docs/results.md`: key findings and references to generated figures and tables.

Document actual implementation and measured results. Do not report planned experiments as completed or make claims unsupported by the results.

## 7. Shared Interfaces

Before parallel implementation, the team must agree on:

1. **Image representation:** input type, dimensions, and pixel-value range.
2. **Corruption contract:** supported corruption names and exact parameters for severities 1–5.
3. **Model output contract:** expected tensor shapes, class ordering, and checkpoint interface.
4. **Evaluation contract:** ensemble predictions, labels, corruption type, and severity metadata.
5. **Experiment output format:** consistent fields for metrics and result files.

The agreed prediction interface should support ensemble logits shaped:

`(ensemble_size, number_of_images, number_of_classes)`

Ground-truth labels and corruption metadata must remain aligned with the corresponding images.

## 8. Mandatory Team Checkpoints

### Checkpoint A — Before parallel implementation

All three members agree on the corruption definitions, image format, model prediction interface, and branch ownership.

### Checkpoint B — Before full ensemble training

The single-model pipeline works, data corruption tests pass, and the evaluation functions pass synthetic-input tests.

### Checkpoint C — Before full evaluation

The trained models load successfully, clean test predictions have the agreed format, and the corruption pipeline produces validated evaluation inputs.

### Checkpoint D — Before final results and paper writing

The team reviews the experiment protocol, checks that all planned conditions were evaluated, verifies the results, and agrees on the interpretation.

Do not silently change a shared interface or experimental protocol. Discuss it with the team and document the decision.

## 9. Pull Request Checklist

Before requesting review, confirm:

- [ ] The change belongs to the stated workstream.
- [ ] Relevant tests have been added or updated.
- [ ] `pytest -v` passes, or any failure is clearly explained.
- [ ] No datasets, checkpoints, credentials, or unintended generated files are included.
- [ ] Documentation is updated where necessary.
- [ ] Shared interfaces and experimental assumptions remain consistent.
- [ ] The pull request explains what changed, why, and how it was tested.

## 10. Research Integrity

All members are responsible for reproducibility and accurate reporting. Keep experiment configurations and seeds, distinguish preliminary results from final results, and record meaningful deviations from the agreed protocol.

The purpose of collaboration is not only to produce working code, but to produce evidence that can be independently checked and used to answer EpiAware's research question.
