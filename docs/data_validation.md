# Data validation and distribution shifts

## Image contract

Each corruption operates on **one** CIFAR-10 image: a NumPy RGB array of shape
`(32, 32, 3)`, dtype `uint8`, with values in the inclusive range 0--255. Each
function returns a newly allocated image in the same format and never changes
the input. The dataset pipeline—not an individual corruption function—keeps
each image paired with its original label.

Severity is ordered only within one corruption family. Severity 3 for noise,
blur, and rotation must not be interpreted as equivalent shifts.

## Parameters and benchmark status

| Corruption | Status | Severity 1--5 parameters |
| --- | --- | --- |
| Gaussian noise | CIFAR-10-C-parameterized project implementation | normalized sigma `[0.04, 0.06, 0.08, 0.09, 0.10]`; uint8 sigma `[10.2, 15.3, 20.4, 22.95, 25.5]` |
| Gaussian blur | CIFAR-10-C-extra-parameterized project implementation | Gaussian-filter sigma `[0.4, 0.6, 0.7, 0.8, 1.0]` |
| Rotation | Custom extension | counter-clockwise angle `[5, 10, 15, 20, 25]` degrees |

The noise and blur **parameters** come from the CIFAR-specific generator in
Hendrycks and Dietterich's corruption benchmark. This project intentionally
uses its own documented processing choices, including uint8-scale noise,
rounding, a seeded local NumPy generator, and SciPy filtering. Therefore, the
functions are CIFAR-10-C-parameterized implementations; they do not claim to
be a bit-for-bit reproduction of the official generator. Gaussian blur is one
of the four extra CIFAR-10-C corruptions, rather than one of its 15 primary
corruptions. Rotation is **not** an official CIFAR-10-C corruption and is
reported separately as a project-specific controlled shift.

## Corruption definitions

### Gaussian noise

`gaussian_noise(image, severity, seed=...)` adds independent zero-mean noise
in uint8 pixel scale, rounds to the nearest integer, then clips to `[0, 255]`.
It uses a local NumPy random generator, so identical input, severity, and seed
give identical output without changing global random state.

### Gaussian blur

`gaussian_blur(image, severity)` applies a Gaussian filter independently to
the RGB channels with the documented sigma. It uses nearest-edge boundary
handling, is deterministic, and does not need a seed.

### Rotation

`rotation(image, severity)` rotates around the image centre. Positive angles
are counter-clockwise. It uses bilinear interpolation; uncovered corner pixels
use fixed black fill (value 0). It is deterministic and does not need a seed.

## Validation checks

`tests/test_corruptions.py` uses synthetic images to check shape, `uint8`
dtype, valid pixel range, input immutability, severity-to-parameter mapping,
and invalid-severity rejection for all three corruptions. It additionally
checks seeded noise reproducibility, noise clipping, Gaussian blur on a
uniform image, and black corner fill after rotation.

Run the focused suite with:

```bash
python -m pytest tests/test_corruptions.py -v
```

Focused corruption-suite result: **42 passed**.

Run the complete repository suite with:

```bash
python -m pytest -v
```

Complete repository-suite result: **46 passed**. This includes the 42
corruption tests above and 4 pre-existing reproducibility tests.

## Real CIFAR-10 validation and visual examples

### Validation Status: Pending

The synthetic-image tests have passed, but real CIFAR-10 visualization is not
yet complete: the first dataset download attempt ended when the remote server
closed the connection. Before full distribution-shift experiments begin, the
team must:

1. successfully load the CIFAR-10 test dataset;
2. run the visualization script;
3. generate grids for Gaussian noise, Gaussian blur, and custom rotation at
   all five severity levels; and
4. visually inspect those grids to confirm that the transformations look
   sensible.

The script downloads the CIFAR-10 test split if needed, checks the first
selected real image against the image contract, then saves a grid with one
clean image and all five severity levels for each corruption:

```bash
python scripts/visualize_corruptions.py --download
```

The default output is `outputs/corruption_grid.png`. Downloaded datasets and
generated figures are intentionally excluded from version control.
