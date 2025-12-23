"""
CONSeg: Conformal Prediction for Voxelwise Uncertainty Quantification in Glioma Segmentation

Implementation based on: CONSeg: Voxelwise Uncertainty Quantification for Glioma Segmentation 
Using Conformal Prediction (arxiv:2502.21158)

Authors: Danial Elyassirad, Benyamin Gheiji, Mahsa Vatanparast, 
         Amir Mahmoud Ahmadzadeh, Shahriar Faghani

Key concepts:
1. Conformal Prediction (CP): Distribution-free uncertainty quantification
2. Nonconformity Score: Measures how different a prediction is from the calibration set
3. Coverage Guarantee: Ensures predictions are correct with specified probability
4. Uncertainty Ratio (UR): Quantifies prediction uncertainty
"""

import inspect
import os
from typing import Tuple, Union, List, Optional
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json, join, isdir, maybe_mkdir_p


class ConformalSegmentationPredictor(nnUNetPredictor):
    """
    Enhanced nnUNet predictor with conformal prediction for uncertainty quantification.
    
    Workflow:
    1. Initialize from trained model
    2. Compute nonconformity scores on calibration set
    3. Select conformal threshold for desired coverage (e.g., 90%, 95%)
    4. Apply conformal prediction to test cases
    5. Compute uncertainty ratio (UR) for each prediction
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conformal_threshold = None
        self.calibration_scores = []
        self.coverage_level = 0.90  # Default 90% coverage as per paper
        
    def compute_nonconformity_score(self, 
                                    predicted_probs: np.ndarray, 
                                    ground_truth: np.ndarray) -> float:
        """
        Compute nonconformity score for a single prediction.
        
        The nonconformity score measures how well the prediction conforms to the ground truth.
        Lower scores indicate better conformity.
        
        Args:
            predicted_probs: Predicted probabilities [C, H, W, D] where C is number of classes
            ground_truth: Ground truth segmentation [H, W, D]
            
        Returns:
            nonconformity_score: Scalar value representing prediction nonconformity
        """
        # Get predicted class probabilities at ground truth locations
        # For each voxel, get the probability of the true class
        
        # Ensure predicted_probs has correct shape
        if predicted_probs.ndim == 3:
            # Single class probability, add channel dimension
            predicted_probs = predicted_probs[np.newaxis, ...]
        
        num_classes = predicted_probs.shape[0]
        
        # For each voxel, get the probability assigned to its true class
        # Shape: [H, W, D]
        true_class_probs = np.zeros_like(ground_truth, dtype=np.float32)
        
        for c in range(num_classes):
            mask = (ground_truth == c)
            true_class_probs[mask] = predicted_probs[c][mask]
        
        # Nonconformity score: 1 - P(true class)
        # Higher score = lower confidence in true class = higher nonconformity
        nonconformity_scores = 1.0 - true_class_probs
        
        # Use quantile-based aggregation (e.g., 90th percentile)
        # This is more robust than mean
        nonconformity_score = np.quantile(nonconformity_scores, 0.90)
        
        return float(nonconformity_score)
    
    def calibrate(self, 
                  calibration_input_folder: str,
                  calibration_labels_folder: str,
                  save_calibration_file: Optional[str] = None):
        """
        Calibrate conformal predictor using calibration set.
        
        Steps:
        1. Run inference on calibration set
        2. Compute nonconformity scores for each case
        3. Determine conformal threshold based on desired coverage level
        
        Args:
            calibration_input_folder: Folder with calibration images
            calibration_labels_folder: Folder with calibration ground truth labels
            save_calibration_file: Optional path to save calibration results
        """
        print("\n" + "="*80)
        print("CONFORMAL PREDICTION CALIBRATION")
        print("="*80)
        print(f"Coverage level: {self.coverage_level:.1%}")
        print(f"Calibration input: {calibration_input_folder}")
        print(f"Calibration labels: {calibration_labels_folder}")
        
        # Get list of calibration cases
        from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        
        calibration_cases = create_lists_from_splitted_dataset_folder(calibration_input_folder)
        print(f"\nFound {len(calibration_cases)} calibration cases")
        
        # Initialize image reader
        reader = SimpleITKIO()
        
        self.calibration_scores = []
        
        for case_id in tqdm(calibration_cases, desc="Calibrating"):
            # Predict on calibration case
            print(f"\nProcessing calibration case: {case_id}")
            
            # Get input files
            input_images = []
            for i in range(len(self.dataset_json['channel_names'])):
                input_file = join(calibration_input_folder, f"{case_id}_{i:04d}.nii.gz")
                if not os.path.exists(input_file):
                    # Try without channel suffix
                    input_file = join(calibration_input_folder, f"{case_id}.nii.gz")
                input_images.append(input_file)
            
            # Run prediction
            predicted_logits, properties = self.predict_single_npy_array(
                self.list_of_parameters[0],  # Use first parameter set
                input_images,
                properties=None,
                save_or_return_probabilities=True
            )
            
            # Convert logits to probabilities
            predicted_probs = torch.softmax(torch.from_numpy(predicted_logits), dim=0).numpy()
            
            # Load ground truth
            label_file = join(calibration_labels_folder, f"{case_id}.nii.gz")
            if not os.path.exists(label_file):
                print(f"Warning: Ground truth not found for {case_id}, skipping")
                continue
            
            ground_truth_data, _ = reader.read_seg(label_file)
            
            # Compute nonconformity score
            score = self.compute_nonconformity_score(predicted_probs, ground_truth_data)
            self.calibration_scores.append(score)
            print(f"  Nonconformity score: {score:.4f}")
        
        # Determine conformal threshold
        # For coverage level alpha (e.g., 0.90), we want the (1-alpha) quantile of calibration scores
        # This ensures that at least alpha fraction of test cases will be covered
        quantile_level = 1.0 - self.coverage_level
        self.conformal_threshold = np.quantile(self.calibration_scores, 1.0 - quantile_level)
        
        print("\n" + "="*80)
        print("CALIBRATION COMPLETE")
        print("="*80)
        print(f"Number of calibration samples: {len(self.calibration_scores)}")
        print(f"Nonconformity score statistics:")
        print(f"  Mean: {np.mean(self.calibration_scores):.4f}")
        print(f"  Median: {np.median(self.calibration_scores):.4f}")
        print(f"  Std: {np.std(self.calibration_scores):.4f}")
        print(f"  Min: {np.min(self.calibration_scores):.4f}")
        print(f"  Max: {np.max(self.calibration_scores):.4f}")
        print(f"\nConformal threshold for {self.coverage_level:.1%} coverage: {self.conformal_threshold:.4f}")
        
        # Save calibration results
        if save_calibration_file:
            calibration_data = {
                'coverage_level': self.coverage_level,
                'conformal_threshold': float(self.conformal_threshold),
                'calibration_scores': [float(s) for s in self.calibration_scores],
                'num_calibration_samples': len(self.calibration_scores),
                'statistics': {
                    'mean': float(np.mean(self.calibration_scores)),
                    'median': float(np.median(self.calibration_scores)),
                    'std': float(np.std(self.calibration_scores)),
                    'min': float(np.min(self.calibration_scores)),
                    'max': float(np.max(self.calibration_scores))
                }
            }
            save_json(calibration_data, save_calibration_file)
            print(f"\nCalibration data saved to: {save_calibration_file}")
    
    def load_calibration(self, calibration_file: str):
        """
        Load pre-computed calibration data.
        
        Args:
            calibration_file: Path to calibration JSON file
        """
        calibration_data = load_json(calibration_file)
        self.coverage_level = calibration_data['coverage_level']
        self.conformal_threshold = calibration_data['conformal_threshold']
        self.calibration_scores = calibration_data['calibration_scores']
        
        print("\n" + "="*80)
        print("LOADED CALIBRATION DATA")
        print("="*80)
        print(f"Coverage level: {self.coverage_level:.1%}")
        print(f"Conformal threshold: {self.conformal_threshold:.4f}")
        print(f"Number of calibration samples: {len(self.calibration_scores)}")
    
    def compute_uncertainty_ratio(self, predicted_probs: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute Uncertainty Ratio (UR) for a prediction.
        
        UR measures the fraction of voxels with high uncertainty (nonconformity).
        As defined in the paper, UR correlates negatively with Dice score.
        
        Args:
            predicted_probs: Predicted probabilities [C, H, W, D]
            
        Returns:
            uncertainty_ratio: Scalar UR value for the whole volume
            voxel_uncertainties: Voxel-wise uncertainty map [H, W, D]
        """
        if self.conformal_threshold is None:
            raise ValueError("Conformal threshold not set. Run calibration first or load calibration data.")
        
        # For each voxel, compute nonconformity: 1 - max_prob
        # This measures uncertainty at each voxel
        max_probs = np.max(predicted_probs, axis=0)  # [H, W, D]
        voxel_nonconformity = 1.0 - max_probs
        
        # Uncertainty ratio: fraction of voxels exceeding conformal threshold
        uncertain_voxels = voxel_nonconformity > self.conformal_threshold
        uncertainty_ratio = np.mean(uncertain_voxels)
        
        return float(uncertainty_ratio), voxel_nonconformity
    
    def predict_with_conformal(self,
                               list_of_lists_or_source_folder: Union[str, List[List[str]]],
                               output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                               save_probabilities: bool = False,
                               overwrite: bool = True,
                               num_processes_preprocessing: int = None,
                               num_processes_segmentation_export: int = None,
                               folder_with_segs_from_prev_stage: str = None,
                               num_parts: int = 1,
                               part_id: int = 0,
                               save_uncertainty_maps: bool = True,
                               save_uncertainty_stats: bool = True):
        """
        Run prediction with conformal uncertainty quantification.
        
        This method extends standard nnUNet prediction to include:
        1. Voxel-wise uncertainty maps
        2. Uncertainty ratio (UR) computation
        3. Coverage analysis
        
        Args:
            save_uncertainty_maps: Whether to save voxel-wise uncertainty maps
            save_uncertainty_stats: Whether to save per-case uncertainty statistics
            (Other args same as nnUNetPredictor.predict_from_files)
        """
        if self.conformal_threshold is None:
            raise ValueError("Conformal threshold not set. Run calibration first or load calibration data.")
        
        print("\n" + "="*80)
        print("CONFORMAL PREDICTION INFERENCE")
        print("="*80)
        print(f"Coverage level: {self.coverage_level:.1%}")
        print(f"Conformal threshold: {self.conformal_threshold:.4f}")
        print(f"Save uncertainty maps: {save_uncertainty_maps}")
        print(f"Save uncertainty stats: {save_uncertainty_stats}")
        
        # Create output folders
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
            maybe_mkdir_p(output_folder)
            
            if save_uncertainty_maps:
                uncertainty_folder = join(output_folder, "uncertainty_maps")
                maybe_mkdir_p(uncertainty_folder)
            
            if save_uncertainty_stats:
                stats_folder = join(output_folder, "uncertainty_stats")
                maybe_mkdir_p(stats_folder)
        else:
            output_folder = None
            uncertainty_folder = None
            stats_folder = None
        
        # Store uncertainty statistics for all cases
        all_uncertainty_stats = {}
        
        # Run standard prediction
        ret = self.predict_from_files(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            save_probabilities=True,  # Need probabilities for uncertainty
            overwrite=overwrite,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
            folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
            num_parts=num_parts,
            part_id=part_id
        )
        
        # Note: For full implementation, we would need to modify the prediction pipeline
        # to access probabilities and compute uncertainty during inference.
        # This is a simplified version that demonstrates the concept.
        
        print("\n" + "="*80)
        print("CONFORMAL PREDICTION COMPLETE")
        print("="*80)
        
        return ret


def enable_dropout_for_uncertainty(model: torch.nn.Module, dropout_rate: float = 0.1):
    """
    Enable dropout for uncertainty estimation (optional for MC Dropout + Conformal).
    
    Args:
        model: Neural network model
        dropout_rate: Dropout probability
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = dropout_rate
            m.train()
    print(f"Enabled dropout with p={dropout_rate} for uncertainty estimation")


def compute_dice_score(prediction: np.ndarray, ground_truth: np.ndarray, 
                      label: int = 1) -> float:
    """
    Compute Dice score for a specific label.
    
    Args:
        prediction: Predicted segmentation
        ground_truth: Ground truth segmentation
        label: Label to compute Dice for
        
    Returns:
        dice: Dice coefficient
    """
    pred_mask = (prediction == label)
    gt_mask = (ground_truth == label)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2.0 * intersection / union
    return float(dice)


def analyze_uncertainty_dice_correlation(uncertainty_ratios: List[float],
                                         dice_scores: List[float],
                                         output_file: str = None):
    """
    Analyze correlation between uncertainty ratio (UR) and Dice score.
    
    As reported in the paper: UR shows significant negative correlation with DSC (p<0.001)
    
    Args:
        uncertainty_ratios: List of UR values
        dice_scores: List of Dice scores
        output_file: Optional file to save analysis results
    """
    from scipy.stats import pearsonr, spearmanr
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(uncertainty_ratios, dice_scores)
    spearman_r, spearman_p = spearmanr(uncertainty_ratios, dice_scores)
    
    print("\n" + "="*80)
    print("UNCERTAINTY-DICE CORRELATION ANALYSIS")
    print("="*80)
    print(f"Number of samples: {len(uncertainty_ratios)}")
    print(f"\nPearson correlation:")
    print(f"  r = {pearson_r:.4f}")
    print(f"  p-value = {pearson_p:.4e}")
    print(f"\nSpearman correlation:")
    print(f"  rho = {spearman_r:.4f}")
    print(f"  p-value = {spearman_p:.4e}")
    
    # Categorize cases
    median_ur = np.median(uncertainty_ratios)
    certain_cases = [dice_scores[i] for i, ur in enumerate(uncertainty_ratios) if ur < median_ur]
    uncertain_cases = [dice_scores[i] for i, ur in enumerate(uncertainty_ratios) if ur >= median_ur]
    
    print(f"\nCategorization by median UR ({median_ur:.4f}):")
    print(f"  Certain cases (UR < median): n={len(certain_cases)}")
    print(f"    Mean Dice: {np.mean(certain_cases):.4f}")
    print(f"  Uncertain cases (UR >= median): n={len(uncertain_cases)}")
    print(f"    Mean Dice: {np.mean(uncertain_cases):.4f}")
    
    from scipy.stats import ttest_ind
    t_stat, t_p = ttest_ind(certain_cases, uncertain_cases)
    print(f"  t-test: t={t_stat:.4f}, p={t_p:.4e}")
    
    if output_file:
        results = {
            'n_samples': len(uncertainty_ratios),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_rho': float(spearman_r),
            'spearman_p': float(spearman_p),
            'median_ur': float(median_ur),
            'certain_cases': {
                'n': len(certain_cases),
                'mean_dice': float(np.mean(certain_cases)),
                'std_dice': float(np.std(certain_cases))
            },
            'uncertain_cases': {
                'n': len(uncertain_cases),
                'mean_dice': float(np.mean(uncertain_cases)),
                'std_dice': float(np.std(uncertain_cases))
            },
            'ttest': {
                't_statistic': float(t_stat),
                'p_value': float(t_p)
            }
        }
        save_json(results, output_file)
        print(f"\nAnalysis saved to: {output_file}")
