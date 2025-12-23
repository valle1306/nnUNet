#!/bin/bash
#SBATCH --job-name=conseg_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=conseg_%j.log
#SBATCH --error=conseg_%j.err

echo "========================================"
echo "CONSeg Final Version - Simplified"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"

source ~/.bashrc
conda activate nnunetv2

export nnUNet_raw="/scratch/hpl14/nnunet_v2_valerie/raw"
export nnUNet_preprocessed="/scratch/hpl14/nnunet_v2_valerie/preprocessed"
export nnUNet_results="/scratch/hpl14/nnunet_v2_valerie/results"

cd ~/nnUNet

DATASET="Dataset777_BraTSPED2024"
MODEL_FOLDER="${nnUNet_results}/${DATASET}/nnUNetTrainer__nnUNetPlans__3d_fullres"
TEST_INPUT="${nnUNet_raw}/${DATASET}/imagesTs"
OUTPUT_BASE="${nnUNet_results}/${DATASET}/conseg_output"

mkdir -p ${OUTPUT_BASE}/segmentations
mkdir -p ${OUTPUT_BASE}/probabilities
mkdir -p ${OUTPUT_BASE}/uncertainty_maps
mkdir -p ${OUTPUT_BASE}/stats

echo "Model: ${MODEL_FOLDER}"
echo "Test images: ${TEST_INPUT}"
echo "Output: ${OUTPUT_BASE}"
echo ""

# Check model and dataset.json
if [ ! -f "${MODEL_FOLDER}/fold_0/checkpoint_final.pth" ]; then
    echo "ERROR: Model checkpoint not found"
    exit 1
fi

if [ ! -f "${MODEL_FOLDER}/dataset.json" ]; then
    echo "ERROR: dataset.json not found in model folder"
    exit 1
fi

echo "Found model checkpoint and dataset.json"
echo ""

# STEP 1: Run prediction WITH probabilities
echo "========================================" 
echo "Running Prediction with Probabilities"
echo "========================================"

nnUNetv2_predict \
    -i ${TEST_INPUT} \
    -o ${OUTPUT_BASE}/probabilities \
    -d 777 \
    -c 3d_fullres \
    -f 0 \
    --disable_tta \
    --save_probabilities \
    -device cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Prediction failed"
    exit 1
fi

echo ""
echo "Prediction complete!"
echo ""

# STEP 2: Process probabilities to create uncertainty maps
echo "========================================"
echo "Computing Uncertainty Maps from Probabilities"
echo "========================================"

python << 'PYTHON_EOF'
import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import json

print("Processing predicted probabilities...")

prob_folder = os.environ['nnUNet_results'] + "/Dataset777_BraTSPED2024/conseg_output/probabilities"
seg_folder = os.environ['nnUNet_results'] + "/Dataset777_BraTSPED2024/conseg_output/segmentations"
unc_folder = os.environ['nnUNet_results'] + "/Dataset777_BraTSPED2024/conseg_output/uncertainty_maps"
stats_folder = os.environ['nnUNet_results'] + "/Dataset777_BraTSPED2024/conseg_output/stats"

# Get all npz files
npz_files = sorted([f for f in os.listdir(prob_folder) if f.endswith('.npz')])
print(f"Found {len(npz_files)} probability files")

# Conformal threshold (from CONSeg paper - 90% coverage)
conformal_threshold = 0.10

all_urs = []
processed = 0

for npz_file in tqdm(npz_files, desc="Processing"):
    try:
        case_id = npz_file.replace('.npz', '')
        
        # Load probabilities
        data = np.load(os.path.join(prob_folder, npz_file))
        probs = data['probabilities']  # Shape: (num_classes, D, H, W) or (num_classes, H, W)
        
        # Compute nonconformity scores (1 - max probability)
        max_probs = np.max(probs, axis=0)
        nonconformity = 1.0 - max_probs
        
        # Uncertainty ratio: fraction of voxels exceeding threshold
        uncertain_mask = nonconformity > conformal_threshold
        ur = float(np.mean(uncertain_mask))
        all_urs.append(ur)
        
        # Find corresponding original image for spacing/orientation
        test_folder = os.environ['nnUNet_raw'] + "/Dataset777_BraTSPED2024/imagesTs"
        ref_file = os.path.join(test_folder, f"{case_id}_0000.nii.gz")
        
        if os.path.exists(ref_file):
            ref_img = sitk.ReadImage(ref_file)
            unc_img = sitk.GetImageFromArray(nonconformity.astype(np.float32))
            unc_img.CopyInformation(ref_img)
            sitk.WriteImage(unc_img, os.path.join(unc_folder, f"{case_id}_uncertainty.nii.gz"))
        else:
            # Save without metadata
            unc_img = sitk.GetImageFromArray(nonconformity.astype(np.float32))
            sitk.WriteImage(unc_img, os.path.join(unc_folder, f"{case_id}_uncertainty.nii.gz"))
        
        # Move segmentation to proper folder
        seg_file_src = os.path.join(prob_folder, f"{case_id}.nii.gz")
        seg_file_dst = os.path.join(seg_folder, f"{case_id}.nii.gz")
        if os.path.exists(seg_file_src):
            os.rename(seg_file_src, seg_file_dst)
        
        # Save statistics
        stats = {
            'case_id': case_id,
            'uncertainty_ratio': ur,
            'mean_nonconformity': float(np.mean(nonconformity)),
            'median_nonconformity': float(np.median(nonconformity)),
            'std_nonconformity': float(np.std(nonconformity)),
            'max_nonconformity': float(np.max(nonconformity)),
            'min_nonconformity': float(np.min(nonconformity)),
            'threshold': conformal_threshold,
            'num_uncertain_voxels': int(np.sum(uncertain_mask)),
            'total_voxels': int(nonconformity.size)
        }
        
        with open(os.path.join(stats_folder, f"{case_id}.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        processed += 1
        
    except Exception as e:
        print(f"ERROR processing {npz_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"\nSuccessfully processed {processed}/{len(npz_files)} cases")

# Save summary
if all_urs:
    summary = {
        'method': 'CONSeg - Conformal Prediction for Segmentation',
        'paper': 'arXiv:2502.21158',
        'conformal_threshold': conformal_threshold,
        'coverage_level': 0.90,
        'num_cases_processed': processed,
        'num_cases_total': len(npz_files),
        'uncertainty_ratio_statistics': {
            'mean': float(np.mean(all_urs)),
            'median': float(np.median(all_urs)),
            'std': float(np.std(all_urs)),
            'min': float(np.min(all_urs)),
            'max': float(np.max(all_urs)),
            'q25': float(np.percentile(all_urs, 25)),
            'q75': float(np.percentile(all_urs, 75))
        }
    }
    
    summary_file = os.path.join(os.environ['nnUNet_results'], "Dataset777_BraTSPED2024/conseg_output/summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("CONSeg Uncertainty Quantification Complete!")
    print("="*60)
    print(f"Conformal threshold: {conformal_threshold}")
    print(f"Mean Uncertainty Ratio: {summary['uncertainty_ratio_statistics']['mean']:.4f}")
    print(f"Median Uncertainty Ratio: {summary['uncertainty_ratio_statistics']['median']:.4f}")
    print(f"\nOutputs:")
    print(f"  Segmentations: {seg_folder}")
    print(f"  Uncertainty maps: {unc_folder}")
    print(f"  Statistics: {stats_folder}")
    print(f"  Summary: {summary_file}")
else:
    print("\nWARNING: No cases were successfully processed!")

PYTHON_EOF

echo ""
echo "========================================"
echo "CONSeg Pipeline Complete!"
echo "Finished: $(date)"
echo "========================================"
