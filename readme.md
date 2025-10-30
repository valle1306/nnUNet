# Welcome to the new nnU-Net!

Click [here](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) if you were looking for the old one instead.

Coming from V1? Check out the [TLDR Migration Guide](documentation/tldr_migration_guide_from_v1.md). Reading the rest of the documentation is still strongly recommended ;-)

## **2024-04-18 UPDATE: New residual encoder UNet presets available!**
Residual encoder UNet presets substantially improve segmentation performance.
They ship for a variety of GPU memory targets. It's all awesome stuff, promised! 
Read more :point_right: [here](documentation/resenc_presets.md) :point_left:

Also check out our [new paper](https://arxiv.org/pdf/2404.09556.pdf) on systematically benchmarking recent developments in medical image segmentation. You might be surprised!

# What is nnU-Net?
Image datasets are enormously diverse: image dimensionality (2D, 3D), modalities/input channels (RGB image, CT, MRI, microscopy, ...), 
image sizes, voxel sizes, class ratio, target structure properties and more change substantially between datasets. 
Traditionally, given a new problem, a tailored solution needs to be manually designed and optimized  - a process that 
is prone to errors, not scalable and where success is overwhelmingly determined by the skill of the experimenter. Even 
for experts, this process is anything but simple: there are not only many design choices and data properties that need to 
be considered, but they are also tightly interconnected, rendering reliable manual pipeline optimization all but impossible! 

![nnU-Net overview](documentation/assets/nnU-Net_overview.png)

**nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided 
training cases and automatically configure a matching U-Net-based segmentation pipeline. No expertise required on your 
end! You can simply train the models and use them for your application**.

Upon release, nnU-Net was evaluated on 23 datasets belonging to competitions from the biomedical domain. Despite competing 
with handcrafted solutions for each respective dataset, nnU-Net's fully automated pipeline scored several first places on 
open leaderboards! Since then nnU-Net has stood the test of time: it continues to be used as a baseline and method 
development framework ([9 out of 10 challenge winners at MICCAI 2020](https://arxiv.org/abs/2101.00232) and 5 out of 7 
# MC-Dropout nnU‑Net — Reproduction & Usage

This repository is an nnU‑Net v2 codebase extended to support Monte Carlo dropout (MC‑Dropout) inference, evaluation, and visualization used in our experiments. The document below explains the minimal, reproducible steps to go from raw data to uncertainty visualizations and metrics. It lists the key files that were edited/added and provides exact commands you can run on a cluster or a workstation.

## Quick summary
- Purpose: add MC‑Dropout inference and uncertainty export on top of nnU‑Net v2.
- Key changes: edited full‑res plans to add dropout (≈0.2), added MC‑Dropout inference predictor, evaluation and visualization scripts (visualization uses rotation fix (2,1,0)).
- Canonical files to use in this repo: `nnunetv2/inference/predict_with_mc_dropout_edited.py`, `run_mc_dropout_inference.py`, `nnunetv2/visualization/visualize_seg_and_uncertainty.py`.

If you only want a one‑line recommendation: follow the steps in "Reproducible steps" below in order.

## Prerequisites
1. Clone repository and install in editable mode:

```bash
git clone <this-repo-url>
cd <repo>
pip install -e .
```

2. Set nnU‑Net environment variables (example):

```bash
export nnUNet_raw='/path/to/nnUNet_raw'
export nnUNet_preprocessed='/path/to/nnUNet_preprocessed'
export nnUNet_results='/path/to/nnUNet_results'
```

3. Create and activate a Python environment with required packages (example):

```bash
conda create -n nnunetv2 python=3.10
conda activate nnunetv2
pip install -r requirements.txt  # if present, otherwise ensure nibabel, numpy, matplotlib, scipy
pip install -e .
```

## Reproducible steps

1) Preprocess dataset (nnU‑Net standard)

Follow the repository documentation: `documentation/how_to_use_nnunet.md` and `documentation/dataset_format.md` to convert your dataset into nnU‑Net format and run preprocessing. After preprocessing, locate the generated plans JSON file for the full‑resolution configuration (look in the preprocessed output folder under `plans*.json`).

2) Edit the full‑resolution plans JSON (manual change you performed)

- Open the full‑res plans JSON (example path under `nnUNet_preprocessed/.../plans_3d_fullres.json`).
- Locate encoder blocks where dropout probability is defined (may appear under `residual_blocks` / `dropout` / similar keys depending on the preset). Set encoder dropout probability to `0.2` (or `0.2` where appropriate). Save the file.

Example (pseudo‑JSON snippet):

```json
"encoder": {
  "dropout_prob": 0.2
}
```

3) Train the model

Use your standard training command, ensuring the trainer loads the edited plans file. Example (adapt to your trainer name):

```bash
python -m nnunetv2.run.training --dataset <TASK_ID> --plans /path/to/plans_3d_fullres.json --trainer YourTrainer
```

4) Confirm dropout layers exist

Run the provided helper to confirm dropout layers are present in the trained model:

```bash
python check_dropout_layers.py --model-folder /path/to/trainer/output/fold_0
```

5) Run MC‑Dropout inference

Edit top of `run_mc_dropout_inference.py` to set input, output, model path, folds and checkpoint names. Then run on the cluster or locally.

Local example:

```bash
python run_mc_dropout_inference.py
```

Example sbatch job for a SLURM cluster (Amarel style):

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
module load anaconda
conda activate nnunetv2
cd /path/to/repo
python run_mc_dropout_inference.py
```

Notes on inference:
- The canonical predictor used here is `nnunetv2/inference/predict_with_mc_dropout_edited.py`. This script enables dropout at inference time and samples multiple stochastic forward passes to compute mean prediction and uncertainty maps.
- Outputs: segmentation predictions and uncertainty maps (written to `nnUNet_results` by default). Check the script headers for exact file naming.

6) Evaluation

Use included evaluation utilities to compute metrics on predictions. Example:

```python
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
# compute_metrics_on_folder(gt_folder, pred_folder, output_csv)
```

Or use any wrapper script you prefer. The repository contains helpers under `nnunetv2/evaluation`.

7) Visualization

- The visualization script that includes segmentation contours and applies the orientation fix (transpose `(2,1,0)`) is `nnunetv2/visualization/visualize_seg_and_uncertainty.py`.
- The uncertainty‑only overlay script is `nnunetv2/visualization/create_proper_uncertainty_overlay.py`.
- The provided Windows runner is `nnunetv2/visualization/run_visualization.ps1`. On Linux you can call the Python scripts directly:

```bash
python nnunetv2/visualization/visualize_seg_and_uncertainty.py
python nnunetv2/visualization/create_proper_uncertainty_overlay.py --original /path/to/case_0000.nii.gz --uncertainty /path/to/case_uncertainty.nii.gz -o /path/to/out
```

Important: `visualize_seg_and_uncertainty.py` contains the explicit transpose step used to correct orientation for our uncertainty volumes:

```python
unc_data = np.transpose(unc_data, (2,1,0))
```

This rotation was determined by inspection (the rotation test is `simple_rotation_test.py`). If your uncertainty maps are already in `(x,y,z)` order you do not need the transpose.

## Key files and purpose
- `nnunetv2/inference/predict_with_mc_dropout_edited.py` — MC‑Dropout predictor used for inference.
- `run_mc_dropout_inference.py` — top‑level runner that calls the predictor (edit constants at top to set paths and folds).
- `check_dropout_layers.py` — helper that checks model files for dropout layers.
- `nnunetv2/visualization/visualize_seg_and_uncertainty.py` — canonical visualizer (applies orientation fix `(2,1,0)` and draws contours).
- `nnunetv2/visualization/create_proper_uncertainty_overlay.py` — uncertainty‑only, attractive overlays (3‑view, multislice, best slice).
- `nnunetv2/visualization/run_visualization.ps1` — Windows PowerShell helper to run either visualizer.
- `nnunetv2/evaluation/` — evaluation helpers and scripts used to compute metrics.

## Backups and history
- Backup folders created during cleanup: `inference_backup_YYYYMMDD_HHMMSS` and `visualization_backup_YYYYMMDD_HHMMSS` at the repository root (look for folders with these prefixes). Example commit SHAs relevant to recent cleanup operations: `403b9a8`, `194a320`.

## Minimal checklist to reproduce
1. Set env vars and install dependencies.
2. Preprocess dataset using nnU‑Net standard steps.
3. Edit full‑res plans JSON and set encoder dropout to `0.2`.
4. Train model with edited plans.
5. Confirm dropout exists with `python check_dropout_layers.py`.
6. Configure `run_mc_dropout_inference.py` and run it on the cluster.
7. Compute metrics and visualize using the visualization scripts (use transpose `(2,1,0)` if needed).

## Contact / notes
If you have questions about a specific change or need the exact commands and job scripts used in a particular run, I can add a precise audit log (timestamps and exact `sbatch` calls) to the repo. Ask me to append an `AUDIT.md` and I will add it.

---
Version: MC‑Dropout adaptation (documented). See `nnunetv2/inference/predict_with_mc_dropout_edited.py` for the main implementation.
