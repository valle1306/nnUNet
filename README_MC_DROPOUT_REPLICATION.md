# MC Dropout Integration: Training, Inference, and Visualization

This guide provides complete instructions to replicate the MC Dropout workflow on nnU-Net, including plans modification, training with dropout regularization, custom inference with uncertainty quantification, and visualization of predictions with uncertainty heatmaps.

## Overview

The MC Dropout workflow consists of four phases:

1. **Phase 1: Plans Modification** - Enable Dropout3d in the training configuration
2. **Phase 2: Training** - Train the network with Dropout3d(p=0.2) for regularization
3. **Phase 3: Inference** - Use the trained model with Dropout3d(p=0.05) for MC Dropout uncertainty estimation
4. **Phase 4: Visualization** - Create combined segmentation and uncertainty overlay images

---

## Phase 1: Plans Modification

### Step 1.1: Locate the Plans File

After running nnU-Net preprocessing, locate the plans file for the desired configuration:

```bash
# Typical path (adjust dataset ID and trainer as needed)
PLANS_FILE="nnUNet_preprocessed/Dataset777_BraTS2024/nnUNetPlans_3d_fullres.json"
```

### Step 1.2: Edit the Plans JSON

Open the plans JSON file and locate the top-level `configurations` section. Find your target configuration (e.g., `3d_fullres`) and add the dropout settings.

**Required modifications:**

```json
{
  "configurations": {
    "3d_fullres": {
      ...existing configuration...
      "dropout_op": "torch.nn.modules.dropout.Dropout3d",
      "dropout_op_kwargs": {
        "p": 0.2,
        "inplace": false
      }
    }
  }
}
```

**Key points:**

- The `"p": 0.2` value regularizes training by randomly zeroing 20% of activations during each epoch
- This is placed at the configuration level, not in the network architecture kwargs
- The architecture builder will not include dropout (arch_kwargs.dropout_op remains null), but the configuration wrapper adds it after initialization
- This two-stage approach allows flexible use of the architecture with or without dropout

### Step 1.3: Verify the Configuration

```python
import json
with open('PLANS_FILE', 'r') as f:
    plans = json.load(f)

config = plans['configurations']['3d_fullres']
assert 'dropout_op_kwargs' in config, "dropout_op_kwargs not found"
assert config['dropout_op_kwargs']['p'] == 0.2, "dropout probability should be 0.2"
print(f"Plans verified: dropout_op_kwargs = {config['dropout_op_kwargs']}")
```

---

## Phase 2: Training

Once the plans are modified, training proceeds using the standard nnU-Net training command:

```bash
nnUNetv2_train Dataset777_BraTS2024 3d_fullres 0 -tr nnUNetTrainer
```

**During training:**

- The network builds without dropout in the architecture (arch_kwargs.dropout_op: null)
- The configuration wrapper adds Dropout3d(p=0.2) layers after initialization
- Dropout is active during all 250-500 epochs, providing regularization
- Validation runs without dropout active (standard practice)

**Verification after training:**

1. Check the generated `plans.json` in the results directory contains your dropout_op_kwargs
2. Check debug.json in fold_0 for the dropout configuration in the configuration_manager output
3. Check the trained checkpoint file contains Dropout3d modules with p=0.2

---

## Phase 3: Custom Inference with MC Dropout

The custom inference script (`nnunetv2/inference/predict_with_mc_dropout_edited.py`) enables MC Dropout uncertainty estimation through these key components:

### Step 3.1: Enable MC Dropout Function

```python
def enable_mc_dropout(model):
    """
    Enable dropout during inference and set all Dropout layers to p=0.05.
    
    During inference:
    - MC Dropout uses lower probability (p=0.05) than training (p=0.2)
    - This provides uncertainty calibration without excessive stochasticity
    - Multiple forward passes (typically 20-50) sample different dropout masks
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = 0.05        # Reduce from training p=0.2 to inference p=0.05
            m.train()          # Keep dropout active during inference
```

**Rationale for p=0.05:**

- Training uses p=0.2 to prevent overfitting during supervised learning
- Inference uses p=0.05 (lower) because we only need stochasticity for uncertainty, not regularization
- This is supported by Concrete Dropout (Gal et al., 2017), which shows optimal uncertainty dropout rates differ from training rates
- Lower inference dropout produces more stable yet diverse predictions

### Step 3.2: Handling Spatial Cropping in Inference

The custom inference script includes a critical modification for handling cropped regions during prediction:

#### Problem

Standard nnU-Net preprocessing may crop images to desired sizes. During inference:
1. The input image is cropped to match training spatial dimensions
2. The network produces predictions for the cropped region only
3. These predictions must be mapped back to the original (uncropped) image space
4. Uncertainty maps must follow the same spatial transformation

#### Solution: `insert_crop_into_image` Mapping

The custom inference script implements proper spatial mapping through four steps:

**Step 3.2.1: Resampling**

```python
# Resample from network output space (typically isotropic spacing)
# to original volume spacing
uncertainty_resampled = resample_uncertainty_to_original_spacing(
    uncertainty_from_network,
    source_spacing=configuration_manager.spacing,
    target_spacing=original_volume_spacing
)
```

**Step 3.2.2: Uncropping**

```python
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

# Create full-size array initialized to zero (for uncropped regions)
uncertainty_uncropped = np.zeros(properties['shape_before_cropping'], dtype=np.float32)

# Insert the resampled predictions back into the original bounding box
uncertainty_uncropped = insert_crop_into_image(
    uncertainty_uncropped,
    uncertainty_resampled,
    properties['crop_bbox']  # [x_min:x_max, y_min:y_max, z_min:z_max]
)
```

The `properties['crop_bbox']` comes from preprocessing and defines exactly where the cropped region sits in the original volume.

**Step 3.2.3: Spatial Transpose**

```python
# Transpose from network coordinate system back to original image coordinate system
uncertainty_final = uncertainty_uncropped.transpose(plans_manager.transpose_backward)
```

**Step 3.2.4: Save with Correct Affine**

```python
# Apply original image affine matrix for correct spatial registration
nifti_img = nib.Nifti1Image(uncertainty_final, affine=original_affine_matrix)
nib.save(nifti_img, output_path)
```

### Step 3.3: Running Inference

Use the provided inference script:

```bash
python nnunetv2/inference/predict_with_mc_dropout_edited.py \
    --dataset_folder "/path/to/preprocessed/dataset" \
    --output_folder "/path/to/predictions" \
    --model_folder "/path/to/trained/model" \
    --num_samples 20
```

**Parameters:**

- `--num_samples`: Number of MC Dropout forward passes (typically 20-50)
  - More samples = better uncertainty estimates but slower inference
  - 20 provides good balance between quality and speed

**Output:**

- `prediction_<case_id>.nii.gz` - Mean prediction (averaged over samples)
- `uncertainty_<case_id>.nii.gz` - Uncertainty map (variance across samples)

---

## Phase 4: Visualization

Combine segmentation predictions with uncertainty heatmaps for visual inspection.

### Step 4.1: Prepare Input Files

```bash
# Required files for each case:
original_image.nii.gz        # Original anatomical image
segmentation.nii.gz          # Segmentation mask (from mean prediction)
uncertainty.nii.gz           # Uncertainty map (from MC Dropout variance)
```

### Step 4.2: Run Visualization

```bash
python nnunetv2/visualization/visualize_seg_and_uncertainty.py
```

### Step 4.3: Visualization Output

The script produces PNG images showing:

1. **Anatomical Background** - Gray-scale original image
2. **Segmentation Contours** - Color-coded boundaries of predicted tumor regions
3. **Uncertainty Overlay** - Hot colormap (red/yellow) indicating high uncertainty regions
4. **Combined View** - All three elements overlaid on axial slices

**Important Note on Image Orientation:**

The visualization script uses `origin='upper'` in matplotlib to display images in standard radiological orientation:
- Superior structures (brain top, eyes, nose) appear at the top
- Inferior structures appear at the bottom
- This matches clinical imaging standards

---

## Complete Workflow Example

### Example: Training on Dataset777_BraTS2024

```bash
# Step 1: Preprocess dataset (standard nnU-Net)
nnUNetv2_plan_and_preprocess -d 777 -c 3d_fullres

# Step 2: Modify plans.json to add dropout
# Edit: nnUNet_preprocessed/Dataset777_BraTS2024/nnUNetPlans_3d_fullres.json
# Add: "dropout_op_kwargs": {"p": 0.2, "inplace": false}

# Step 3: Train with dropout
nnUNetv2_train Dataset777_BraTS2024 3d_fullres 0 -tr nnUNetTrainer

# Step 4: Run inference with MC Dropout
python nnunetv2/inference/predict_with_mc_dropout_edited.py \
    --dataset_folder "nnUNet_preprocessed/Dataset777_BraTS2024/imagesTr" \
    --output_folder "./predictions" \
    --model_folder "nnUNet_results/Dataset777_BraTS2024/nnUNetTrainer__nnUNetPlans__3d_fullres" \
    --num_samples 20

# Step 5: Visualize predictions
python nnunetv2/visualization/visualize_seg_and_uncertainty.py
```

---

## Key Implementation Details

### Dropout Configuration Levels

nnU-Net uses a two-stage dropout configuration:

1. **Architecture Level** (`arch_kwargs.dropout_op`)
   - Location: `plans.json` → `configurations` → `<config_name>` → `architecture` → `arch_kwargs`
   - Default: `null` (no dropout in base architecture)
   - Purpose: Tells the architecture builder not to include dropout layers

2. **Configuration Level** (`dropout_op_kwargs`)
   - Location: `plans.json` → `configurations` → `<config_name>` → top-level
   - Value: `{"p": 0.2, "inplace": false}`
   - Purpose: Configuration wrapper adds Dropout3d after network initialization
   - This is the ACTUAL dropout used during training and inference

### Why Lower Dropout for Inference?

- **Training (p=0.2)**: Regularizes by forcing robustness to missing activations
- **Inference (p=0.05)**: Minimal stochasticity to sample uncertainty without instability
- **Theoretical Support**: Concrete Dropout (Gal et al., 2017) shows these can differ
- **Practical Benefit**: p=0.05 uncertainty is better calibrated for decision-making

### Spatial Mapping in Custom Inference

The custom inference script is critical because:

1. **Cropping Handling**: Preprocessing may crop images to desired dimensions
2. **Proper Mapping**: The `insert_crop_into_image` function places cropped predictions back in original space
3. **Uncertainty Preservation**: Same spatial transformations applied to uncertainty maps
4. **Affine Preservation**: Original image affine matrix ensures correct registration

Without proper mapping, predictions would be spatially offset from the original image.

---

## Troubleshooting

### Issue: Dropout layers not found during inference

**Cause**: Dropout not properly added during configuration initialization

**Solution**: Verify plans.json has `dropout_op_kwargs` at configuration level, not in arch_kwargs

### Issue: Uncertainty maps show all zeros

**Cause**: Dropout not active during inference or only single forward pass

**Solution**: Ensure `enable_mc_dropout()` is called and sets `m.train()`. Verify `num_samples > 1`

### Issue: Uncertainty maps misaligned with segmentation

**Cause**: Cropping offset not properly handled

**Solution**: Verify `properties['crop_bbox']` is correctly extracted and used in `insert_crop_into_image()`

### Issue: Images appear upside-down in visualization

**Cause**: Matplotlib origin parameter set incorrectly

**Solution**: Use `origin='upper'` in imshow() calls for radiological orientation

---

## References

**Theoretical Foundation:**

- Gal & Ghahramani (2015): Dropout as a Bayesian Approximation
- Gal et al. (2017): Concrete Dropout
- Ovadia et al. (2019): Uncertainty Calibration Under Distribution Shift

**Framework Reference:**

- Isensee et al. (2021): nnU-Net v2 architecture and training strategies

---

## Summary

This workflow enables uncertainty quantification in medical image segmentation by:

1. Modifying nnU-Net plans to enable Dropout3d(p=0.2) during training
2. Training the network with dropout regularization
3. Adapting dropout to p=0.05 during inference for uncertainty sampling
4. Properly handling spatial cropping through inverse transformations
5. Visualizing predictions with uncertainty overlays

The custom inference script handles the critical spatial mapping to ensure predictions align with original images despite preprocessing crop operations.
