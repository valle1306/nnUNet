#!/usr/bin/env python3
"""
CONSeg Evaluation Script
Computes Dice scores and analyzes uncertainty-error correlation
"""

import os
import sys
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from scipy import stats

def compute_dice(pred, gt, label):
    """Compute Dice score for a specific label"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 1.0  # Both empty
    return 2.0 * intersection / (pred_mask.sum() + gt_mask.sum())

def evaluate_case(pred_file, gt_file, unc_file, case_id):
    """Evaluate a single case"""
    try:
        # Load files
        pred_img = sitk.ReadImage(pred_file)
        gt_img = sitk.ReadImage(gt_file)
        unc_img = sitk.ReadImage(unc_file)
        
        pred = sitk.GetArrayFromImage(pred_img)
        gt = sitk.GetArrayFromImage(gt_img)
        unc = sitk.GetArrayFromImage(unc_img)
        
        # Compute Dice scores per class
        dice_scores = {}
        labels = {
            'enhancing_core': 1,
            'non_enhancing_core': 2,
            'cyst': 3,
            'edema': 4
        }
        
        for name, label in labels.items():
            dice_scores[name] = compute_dice(pred, gt, label)
        
        # Overall Dice (mean of all classes)
        dice_scores['mean'] = np.mean(list(dice_scores.values()))
        
        # Analyze uncertainty-error correlation
        errors = (pred != gt)
        
        # Mean uncertainty in error regions vs correct regions
        unc_in_errors = unc[errors].mean() if errors.sum() > 0 else 0
        unc_in_correct = unc[~errors].mean() if (~errors).sum() > 0 else 0
        
        # Correlation between uncertainty and errors
        unc_flat = unc.flatten()
        error_flat = errors.flatten().astype(float)
        
        if len(np.unique(unc_flat)) > 1 and len(np.unique(error_flat)) > 1:
            correlation, p_value = stats.pearsonr(unc_flat, error_flat)
        else:
            correlation, p_value = 0.0, 1.0
        
        results = {
            'case_id': case_id,
            'dice_scores': dice_scores,
            'uncertainty_analysis': {
                'mean_unc_in_errors': float(unc_in_errors),
                'mean_unc_in_correct': float(unc_in_correct),
                'unc_error_correlation': float(correlation),
                'correlation_p_value': float(p_value),
                'error_rate': float(errors.sum() / errors.size),
                'num_error_voxels': int(errors.sum()),
                'total_voxels': int(errors.size)
            }
        }
        
        return results
        
    except Exception as e:
        print(f"ERROR evaluating {case_id}: {e}")
        return None

def main():
    # Paths
    pred_folder = os.environ['nnUNet_results'] + "/Dataset777_BraTSPED2024/conseg_output/segmentations"
    gt_folder = os.environ['nnUNet_raw'] + "/Dataset777_BraTSPED2024/labelsTs"
    unc_folder = os.environ['nnUNet_results'] + "/Dataset777_BraTSPED2024/conseg_output/uncertainty_maps"
    output_folder = os.environ['nnUNet_results'] + "/Dataset777_BraTSPED2024/conseg_output/evaluation"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all cases
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.nii.gz')])
    print(f"Evaluating {len(pred_files)} cases...")
    
    all_results = []
    dice_scores_list = []
    
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        case_id = pred_file.replace('.nii.gz', '')
        
        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, pred_file)
        unc_path = os.path.join(unc_folder, f"{case_id}_uncertainty.nii.gz")
        
        if not os.path.exists(gt_path):
            print(f"WARNING: No ground truth for {case_id}")
            continue
        
        if not os.path.exists(unc_path):
            print(f"WARNING: No uncertainty map for {case_id}")
            continue
        
        result = evaluate_case(pred_path, gt_path, unc_path, case_id)
        
        if result:
            all_results.append(result)
            dice_scores_list.append(result['dice_scores']['mean'])
            
            # Save per-case result
            with open(os.path.join(output_folder, f"{case_id}_eval.json"), 'w') as f:
                json.dump(result, f, indent=2)
    
    if not all_results:
        print("ERROR: No cases were evaluated!")
        return
    
    # Aggregate results
    print(f"\nSuccessfully evaluated {len(all_results)} cases")
    
    # Compute summary statistics
    all_dice = {
        'enhancing_core': [],
        'non_enhancing_core': [],
        'cyst': [],
        'edema': [],
        'mean': []
    }
    
    all_unc_in_errors = []
    all_unc_in_correct = []
    all_correlations = []
    all_error_rates = []
    
    for result in all_results:
        for key in all_dice.keys():
            all_dice[key].append(result['dice_scores'][key])
        
        all_unc_in_errors.append(result['uncertainty_analysis']['mean_unc_in_errors'])
        all_unc_in_correct.append(result['uncertainty_analysis']['mean_unc_in_correct'])
        all_correlations.append(result['uncertainty_analysis']['unc_error_correlation'])
        all_error_rates.append(result['uncertainty_analysis']['error_rate'])
    
    # Summary
    summary = {
        'num_cases': len(all_results),
        'dice_scores': {
            key: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            for key, values in all_dice.items()
        },
        'uncertainty_error_analysis': {
            'mean_unc_in_errors': {
                'mean': float(np.mean(all_unc_in_errors)),
                'std': float(np.std(all_unc_in_errors))
            },
            'mean_unc_in_correct': {
                'mean': float(np.mean(all_unc_in_correct)),
                'std': float(np.std(all_unc_in_correct))
            },
            'unc_error_correlation': {
                'mean': float(np.mean(all_correlations)),
                'std': float(np.std(all_correlations))
            },
            'error_rate': {
                'mean': float(np.mean(all_error_rates)),
                'std': float(np.std(all_error_rates))
            }
        }
    }
    
    # Save summary
    summary_file = os.path.join(output_folder, 'evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nDice Scores (Mean ± Std):")
    for key, values in summary['dice_scores'].items():
        print(f"  {key:20s}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    print(f"\nUncertainty-Error Analysis:")
    print(f"  Mean unc in errors:  {summary['uncertainty_error_analysis']['mean_unc_in_errors']['mean']:.6f}")
    print(f"  Mean unc in correct: {summary['uncertainty_error_analysis']['mean_unc_in_correct']['mean']:.6f}")
    print(f"  Correlation (unc-error): {summary['uncertainty_error_analysis']['unc_error_correlation']['mean']:.4f}")
    print(f"  Mean error rate: {summary['uncertainty_error_analysis']['error_rate']['mean']*100:.2f}%")
    
    print(f"\nResults saved to: {output_folder}")
    print("="*60)

if __name__ == '__main__':
    main()
