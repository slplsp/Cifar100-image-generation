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

![image](https://github.com/user-attachments/assets/6a254e90-1c6d-403d-87b2-72902f4d1b00)

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
```

## Running the Code

The codebase includes several Jupyter notebooks to help you reproduce our results and generate images. Below are the steps to run each notebook:

### 1. Reproduce Model Files

**Notebook:** `run.ipynb`

This notebook trains the model and generates the necessary model files.

**Steps:**
1. Open `run.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Run all the cells sequentially.
3. Upon completion, the trained model files will be saved in the designated directory.

### 2. Evaluate the Model

**Notebook:** `Three indicator tests.ipynb`

This notebook calculates the **FID**, **Intra-FID**, and **Inception Score (IS)** values to reproduce our evaluation metrics.

**Steps:**
1. Ensure that the trained model files from `run.ipynb` are available.
2. Open `Three indicator tests.ipynb` in Jupyter Notebook or Jupyter Lab.
3. Run all the cells sequentially.
4. The **FID**, **Intra-FID**, and **IS** values will be displayed as output.

### 3. Generate Images

**Notebook:** `Image generation.ipynb`

This notebook generates new images using the trained model.

**Steps:**
1. Ensure that the trained model files from `run.ipynb` are available.
2. Open `Image generation.ipynb` in Jupyter Notebook or Jupyter Lab.
3. Run all the cells sequentially.
4. Generated images will be saved in the specified output directory.

## Dependencies

Make sure to install the required dependencies before running the notebooks. You can install them using:

```bash
pip install -r requirements.txt
```

## Contact

If you have any questions, feel free to reach out via email:  
- suilinpeng15@gmail.com
- sl695969@outlook.com
```

