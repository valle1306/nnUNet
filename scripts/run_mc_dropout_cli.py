#!/usr/bin/env python3
"""
CLI for MC Dropout inference (moved to scripts/)
"""
import argparse
import sys
import os
from pathlib import Path
import torch

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from nnunetv2.inference.predict_with_mc_dropout_ver4 import nnUNetPredictor, enable_mc_dropout

def main():
    parser = argparse.ArgumentParser(
        description='Run nnU-Net inference with MC Dropout uncertainty estimation'
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input folder containing images to predict')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output folder for predictions and uncertainty maps')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to trained model folder')
    parser.add_argument('-f', '--fold', type=str, default='all',
                        help='Fold to use (default: all, since you trained on all data)')
    parser.add_argument('-chk', '--checkpoint', type=str, default='checkpoint_final.pth',
                        help='Checkpoint name (default: checkpoint_final.pth)')
    parser.add_argument('--mc_passes', type=int, default=20,
                        help='Number of MC dropout passes (default: 20)')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Sliding window step size (default: 0.5)')
    parser.add_argument('--disable_tta', action='store_true',
                        help='Disable test time augmentation (mirroring)')
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Save class probabilities')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"Model folder: {args.model}")
    print(f"Using fold: {args.fold}")
    print(f"MC dropout passes: {args.mc_passes}")
    
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=True
    )
    
    # Load model
    print("Loading trained model...")
    use_folds = [args.fold] if isinstance(args.fold, str) else args.fold
    predictor.initialize_from_trained_model_folder(
        args.model,
        use_folds,
        args.checkpoint
    )
    
    # Enable MC dropout
    print("Enabling MC dropout...")
    enable_mc_dropout(predictor.network)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run inference
    print(f"Running MC dropout inference with {args.mc_passes} passes...")
    predictor.predict_from_files_sequential(
        args.input,
        args.output,
        save_probabilities=args.save_probabilities,
        overwrite=True,
        folder_with_segs_from_prev_stage=None
    )
    
    print("✅ MC dropout inference completed!")
    print(f"Results saved to: {args.output}")
    print("Check for both segmentation (.nii.gz) and uncertainty maps (uncertainty.nii.gz)")

if __name__ == "__main__":
    main()
