# CIFAR-100 Image Generation

Welcome to the Elema team's codebase. This codebase is focused on CIFAR-100 image generation.

## Overview

The **CIFAR-100** dataset contains 60,000 32x32 color images categorized into 100 fine-grained classes, with 600 images per class. The dataset is divided into:
- **Training set**: 50,000 images

Additionally, CIFAR-100 classes are grouped into:
- **Superclasses**: 20 broader categories
- **Fine classes**: 100 specific categories

## Training Configuration

To ensure reproducibility of our results, we set the random seed to **42** during the training process.

## Best Training Results

- **FID Score**: 15.8565
- **Inception Score**: 6.0625 ± 0.7847
- **Intra-FID**: 51.2626

*All the above results were obtained with the random seed set to 42.*

## Reproducing the Results

To reproduce the training results, make sure to set the random seed to **42** in your environment. This can typically be done by setting the seed in your training script as follows:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
