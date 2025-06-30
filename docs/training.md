# 🏋️ Training

We provide preprocessed datasets and training scripts to reproduce the results in the paper.

---

## 📦 Dataset Preparation

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1pM2SU0p6nNdW_c4z2bgtnKsnyCsI2gKm?usp=drive_link) and place it under the following directory structure:
```angular2html
{project_root}/
├── src/
├── data/
│ └── abc_processed/
│ ├── v1_1_grid8.h5
│ ├── deepcad_30_train.txt
│ ├── deepcad_30_val.txt
│ ├── deepcad_30_test.txt
│ ├── deepcad_30_pkl_absence.txt
│ ├── abc_50_train.txt
│ ├── abc_50_val.txt
│ ├── abc_50_test.txt
│ └── abc_50_pkl_absence.txt
└── ...
```

- `deepcad_30_*` contains the train/val/test splits for the DeepCAD dataset.
- `abc_50_*` contains the splits for the ABC 50-face subset.
- The `*_pkl_absence.txt` files list missing samples relative to BrepGen. For the DeepCAD dataset, this should be empty,  so do not worry about it.

---

## 🚀 Training

We recommend the following hardware configurations:

### 🔹 DeepCAD dataset (2 × 24GB GPUs)
```bash
python scripts/run.py logs/brepdiff/deepcad --config-path configs/deepcad.yaml
```
Expected training time: ~4 days with 2×NVIDIA 3090 (24GB) GPUs.

### 🔹 ABC 50-face dataset (4 × 48GB GPUs)
```bash
python scripts/run.py logs/brepdiff/abc --config-path configs/abc.yaml
```
Expected training time: ~7 days with 4×NVIDIA A6000 (48GB) GPUs.


Add the `--debug` flag to quickly verify that training runs correctly.
By default, Weights & Biases (W&B) logging is enabled.
Use the `--wandb-offline` flag to disable W&B logging.

---

## 🧯 Troubleshooting
### ❌ ValueError: current limit exceeds maximum limit
You may see an error like this:
```
Traceback (most recent call last):
  File "scripts/run.py", line 18, in <module>
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
ValueError: current limit exceeds maximum limit
```

This means your system’s file descriptor soft limit is too high.
To resolve it, lower the value in `scripts/run.py`:
```
# Before:
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

# After:
resource.setrlimit(resource.RLIMIT_NOFILE, (1024, rlimit[1]))
```


