# Data validation and distribution shifts

## Image contract

The corruption functions operate on **one** CIFAR-10 image at a time.  An input
must be a NumPy RGB array of shape `(32, 32, 3)`, dtype `uint8`, with values in
the inclusive range 0--255.  Each function returns a newly allocated image in
the same format.  The dataset pipeline, rather than an individual corruption
function, keeps the image associated with its original label.

## Gaussian noise

`gaussian_noise(image, severity, seed=...)` adds independent, zero-mean
Gaussian noise directly in uint8 pixel scale, rounds to the nearest integer,
then clips to `[0, 255]`.  It uses a local `numpy.random.Generator`; identical
input, severity, and seed therefore produce an identical result without
altering global random state.

The five values are the approved CIFAR-10-C Gaussian-noise configuration from
Hendrycks and Dietterich, *Benchmarking Neural Network Robustness to Common
Corruptions and Perturbations*.

| Severity | Normalized standard deviation | uint8-scale standard deviation |
| --- | ---: | ---: |
| 1 | 0.08 | 20.4 |
| 2 | 0.12 | 30.6 |
| 3 | 0.18 | 45.9 |
| 4 | 0.26 | 66.3 |
| 5 | 0.38 | 96.9 |

Severity is ordered within Gaussian noise only.  It is not assumed to be
directly comparable with a severity number for a future blur or rotation
corruption.

## Validation checks

`tests/test_corruptions.py` uses synthetic images to verify:

- shape and `uint8` dtype are preserved;
- output values remain within `[0, 255]`, including when clipping is needed;
- identical seeds reproduce identical output;
- the input array is unchanged; and
- the five severity levels map to the documented values, while invalid levels
  are rejected.

## Current scope and limitations

Only Gaussian noise is implemented in this milestone.  Blur and rotation will
be added after their parameters and, for rotation, interpolation and boundary
fill behavior are agreed by the team.  The synthetic tests do not download
CIFAR-10; an integration check on sampled CIFAR-10 images remains a later
milestone.
