# 🔧 Overfitting BrepDiff on Custom STEP Data

This guide explains how to adapt BrepDiff so that you can overfit the model on your own CAD solids stored as `.step` files. The process has three parts:

1. **Convert STEP solids to UV-grids** that follow the structure expected by BrepDiff.
2. **Register a dataset split** that points to your processed sample(s).
3. **Enable overfitting in the training configuration**.

---

## 1. Convert STEP Files into the Expected Format

BrepDiff trains on UV-grids saved inside an HDF5 file. Each entry must provide face coordinates, normals, a visibility mask, and primitive labels. You can start from STEP files by reusing the conversion utilities that ship with the repository.

### 1.1 Generate UV-grids

Use the helper pipeline in `scripts/visualization/vis_step.py` to convert STEP geometry into UV-grid tensors:

```bash
python scripts/visualization/vis_step.py path/to/part.step --output-dir data/custom_uvgrid
```

This script reads the STEP solid with OpenCascade, discretises faces and edges, normalises the shape to a unit box, and saves the result as an `.npz` file containing vertices, faces and the sampled UV-grid (`*_uvgrid.npz`).【F:scripts/visualization/vis_step.py†L79-L179】【F:scripts/visualization/vis_step.py†L200-L255】

### 1.2 Write an HDF5 sample

Training expects an HDF5 file where each sample is stored under `data/<uid>` with the following datasets: `coords`, `normals`, `masks`, and `types`. The ABC/DeepCAD loader shows the required layout when it reads the file.【F:src/brepdiff/datasets/abc_dataset.py†L63-L114】 The values come directly from the UV-grid tensors; `types` can be set to a constant (e.g., all ones) when labels are unavailable.

Run the helper script to package all NPZ files into the correct HDF5 structure and emit a matching split list:

```bash
python -m scripts.postprocessing.npz_to_h5 \
    data/custom_uvgrid/npz_for_vis \
    data/custom_uvgrid/custom.h5 \
    --list-path data/custom_uvgrid/custom_train.txt
```

The command strips the `_uvgrid` suffix from each file name to build the dataset ids, writes the tensors into `custom.h5`, and records the ids inside `custom_train.txt`.【F:scripts/postprocessing/npz_to_h5.py†L104-L166】

If you prefer to script the conversion yourself, the snippet below illustrates the file structure expected by the loader:

```python
import h5py
import numpy as np
from brepdiff.primitives.uvgrid import UvGrid

uvgrid = UvGrid.load_from_npz_data(dict(np.load("data/custom_uvgrid/my_part_uvgrid.npz")))
uid = "my_part"

with h5py.File("data/custom_uvgrid/custom.h5", "a") as f:
    grp = f.require_group("data")
    if uid in grp:
        del grp[uid]
    entry = grp.create_group(uid)
    entry.create_dataset("coords", data=uvgrid.coord)
    entry.create_dataset("normals", data=uvgrid.normal)
    entry.create_dataset("masks", data=uvgrid.grid_mask)
    entry.create_dataset("types", data=np.ones((uvgrid.coord.shape[0],), dtype=np.int32))
```

Create companion text files that list the sample ids (one per line) so the dataset loader can find them. For example:

```
data/custom_uvgrid/custom_train.txt
└── my_part
```

Store empty lists for validation/test if you only want to overfit training.

---

## 2. Register a Dataset Class

Add a lightweight dataset wrapper that points to your new file. You can inherit from the existing ABC dataset so that tokenisation, augmentations, and batching continue to work. Create `src/brepdiff/datasets/custom_dataset.py` with:

```python
from brepdiff.datasets.abc_dataset import ABCDataset

class CustomStepDataset(ABCDataset):
    name = "custom_step"

    def get_data_list_path(self) -> str:
        return "./data/custom_uvgrid/custom_train.txt"

    def get_invalid_step_list_path(self) -> str:
        return "./data/custom_uvgrid/custom_invalid.txt"
```

Then register it so the factory recognises the new key by updating `src/brepdiff/datasets/__init__.py`:

```python
from brepdiff.datasets.custom_dataset import CustomStepDataset

DATASETS = {
    ...
    CustomStepDataset.name: CustomStepDataset,
}
```

The base ABC implementation handles batching, augmentations, and UV-grid stacking for you.【F:src/brepdiff/datasets/abc_dataset.py†L35-L170】

---

## 3. Enable Overfitting in the Config

Duplicate one of the existing configs and point it to your dataset:

```yaml
# configs/custom_overfit.yaml
dataset: custom_step
h5_path: "./data/custom_uvgrid/custom.h5"
max_n_prims: 30
random_augmentations: "none"
overfit: True
overfit_data_size: 1
overfit_data_repetition: 2048
num_epochs: 200
batch_size: 16
```

The `overfit` switches tell the dataset to keep only the first `overfit_data_size` items and repeat them enough times to fill each batch.【F:src/brepdiff/datasets/abc_dataset.py†L128-L150】【F:src/brepdiff/config.py†L218-L229】 You can override additional hyperparameters at launch time:

```bash
python scripts/run.py logs/custom_overfit --config-path configs/custom_overfit.yaml --debug
```

During debugging you can also force Lightning to train on a single batch by adding `--trainer.fast_dev_run=1` if necessary.

---

## ✅ Checklist

1. Convert each STEP file to UV-grids and pack them into an HDF5 file.
2. Register a dataset class that points to your processed splits.
3. Turn on `overfit` in the config and launch `scripts/run.py` with your new configuration.

Following these steps will let you quickly verify that BrepDiff can memorise your custom CAD sample before scaling to larger datasets.
