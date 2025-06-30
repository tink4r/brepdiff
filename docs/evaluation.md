# 🎲 Evaluation: Unconditional Generation
We provide scripts, preprocessed datasets, and pretrained models to reproduce the results reported in the paper, specifically the generation quality (Tables 1 and 4) and generation speed (Table 2).

🔗 Pretrained models: Download [here](https://drive.google.com/drive/folders/1HtmkWrssQhna5Ouyqq80pgSJNyT81aRS?usp=drive_link),

📂 Preprocessed datasets: Download [here](https://drive.google.com/drive/folders/1pM2SU0p6nNdW_c4z2bgtnKsnyCsI2gKm?usp=drive_link).

Please organize the downloaded files as follows:

```
{repo_root}/
├── logs/pretrained/
│   ├── deepcad/
│   │   ├── config.yaml
│   │   └── ckpts/
│   │       └── epoch_5999-*.ckpt
│   └── abc/
│       ├── config.yaml
│       └── ckpts/
│           └── epoch_5999-*.ckpt
└── data/
    └── abc_processed/
        ├── v1_1_grid8.h5

        # DeepCAD split files
        ├── deepcad_30_train.txt
        ├── deepcad_30_val.txt
        ├── deepcad_30_test.txt
        ├── deepcad_30_pkl_absence.txt

        # ABC split files
        ├── abc_50_train.txt
        ├── abc_50_val.txt
        ├── abc_50_test.txt
        ├── abc_50_pkl_absence.txt

        # Test point clouds for Table 1 (1-NNA, COV, MMD)
        ├── deepcad_30_test_brep_points.npz
        ├── abc_50_test_brep_points.npz

        # FID cache for Table 1
        └── fid/
            ├── fid_cache_vanilla_deepcad_30_4views_test.npz
            └── fid_cache_vanilla_abc_50_4views_test.npz

        # Mesh-sampled test point clouds for Table 4 (from BrepGen)
        ├── deepcad_30_test_pcd/
        │   └── *.ply
        └── abc_50_test_pcd/
            └── *.ply
```

## ✨ Generation Quality
To evaluate generation quality, the script samples 10,000 UV grids using the diffusion model, then postprocesses them into B-reps.
We validate using the first 3,000 valid (watertight) B-reps, as not all UV grids yield valid outputs.

You can modify:
* test_batch_size to fit your GPU memory
* test_num_sample to adjust the number of sampled grids

### Step 1. Generate uvgrids using diffusion model
DeepCAD
```
python scripts/run.py --test --wandb-offline logs/pretrained/deepcad \
    --ckpt-path logs/pretrained/deepcad/ckpts/epoch_5999-step_486000-val_loss_0.0133.ckpt \
    --override "test_batch_size=2500|test_num_sample=10000"
```

ABC
```
python scripts/run.py --test --wandb-offline logs/pretrained/abc \
    --ckpt-path logs/pretrained/abc/ckpts/epoch_5999-step_756000-val_loss_0.0141.ckpt \
    --override "test_batch_size=2500|test_num_sample=10000"
```

### Step 2. Postprocess and Evaluate
Given the masked uvgrids generated from step 2, we postprocess them into B-reps and evaluate the quality of generated B-reps.

DeepCAD
```
 python -m scripts.postprocessing.npz_pp save-breps logs/pretrained/deepcad/vis/test/step-000486000/deepcad_30/cfg_1.00 --coarse-to-fine
```

ABC
```
 python -m scripts.postprocessing.npz_pp save-breps logs/pretrained/abc/vis/test/step-000756000/abc_50/cfg_1.00 --coarse-to-fine --dataset-name abc --max-n-prims 50
```



## ⚡ Generation Speed
To evaluate generation speed (as shown in Table 2), run the following:

DeepCAD
```
python -m scripts.metrics.test_speed main \
    logs/pretrained/deepcad/ckpts/epoch_5999-step_486000-val_loss_0.0133.ckpt \
    --coarse-to-fine --n-generate 100
```

ABC
```
python -m scripts.metrics.test_speed main \
    logs/pretrained/abc/ckpts/epoch_5999-step_756000-val_loss_0.0141.ckpt \
    --coarse-to-fine --n-generate 100
```

