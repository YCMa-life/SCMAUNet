# SCMAU-Net: Sparsified Complex-valued Multi-scale Attention U-Net for Accelerated MRI Reconstruction

Code Base:Adapted from [Zhaoyang Jin's SCUNET](https://github.com/ZhyJin/SCUNET)

1. Requirements

### Python Environment

Create a virtual environment (i.e., with pip or conda) and install all the required packages.
Here are a few packages that I use.
- Python = 2.9
- PyTorch = 1.9.5 (with CUDA support recommended)
- NumPy
- SciPy
- CUDA

### MATLAB Environment (for evaluation)

- MATLAB
- Image Processing Toolbox

---

2. Data Preparation

The model expects brain MRI data stored as `.mat` files.
Implement undersampling and sparsification processing in the `../MATLAB/producePMRIdata.m`, and save the data as a `.mat` file.
Each file contains a variable named `'imags'` — a 3D array of shape `(H, W, 2×C)`,
where `C` is the number of coils and the factor of 2 accounts for concatenated real and imaginary parts (first `C` channels = real, remaining `C` channels = imaginary).

### Expected directory layout:

```
data/
├── ref_test/               # Fully-sampled reference images (test set)
│   └── *.mat               # Each file: variable 'imags' of shape (320, 320, 32)
├── sref_train/             # Fully-sampled reference images (training set)
│   └── *.mat
└── C=40 N=4/
    ├── strain/             # Undersampled images (training input)
    │   └── *.mat
    └── stest/              # Undersampled images (test input)
        └── *.mat
```
The central 40 lines (C=40) of k-space were fully sampled, and the peripheral region was undersampled with a line interval N=4.

---

3. How to use

### Training

Run the training script at the `SCMAU-Net/train_val.py`.

### Testing

Run the `test.py` to generate the reconstructed `.mat` file on the test set.

In MATLAB, navigate to the `MATLAB/` directory and run `Recon_test.m`,
it replaces the inverse filtered data with fully sampled lines around the center of k-space,
and performs data consistency replacement on the inverse filtered data.

After generating reconstructions, evaluate quantitative metrics.


