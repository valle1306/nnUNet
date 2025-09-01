#!/usr/bin/env python3
"""
Script to run MC Dropout inference with uncertainty estimation
"""
import sys
import os
from pathlib import Path

# Add the nnUNet directory to path
nnunet_dir = Path(__file__).parent
sys.path.insert(0, str(nnunet_dir))

# Import your custom predictor
from nnunetv2.inference.predict_with_mc_dropout_ver4 import nnUNetPredictor, enable_mc_dropout
import torch

def main():
    # Configuration
    INPUT_FOLDER = r"C:\path\to\your\input\images"  # Update this path
    OUTPUT_FOLDER = r"C:\path\to\your\output"       # Update this path
    MODEL_FOLDER = r"C:\path\to\your\trained\model" # Update this path
    
    # Since you trained on 'all' data, use fold 'all'
    USE_FOLDS = ['all']  # or just 'all' as string
    CHECKPOINT_NAME = 'checkpoint_final.pth'
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    
    # Initialize from your trained model
    print("Loading trained model...")
    predictor.initialize_from_trained_model_folder(
        MODEL_FOLDER,
        USE_FOLDS,
        CHECKPOINT_NAME
    )
    
    # Enable MC dropout
    print("Enabling MC dropout for uncertainty estimation...")
    enable_mc_dropout(predictor.network)
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Run prediction with MC dropout and uncertainty estimation
    print("Running MC dropout inference...")
    predictor.predict_from_files_sequential(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        save_probabilities=True,  # Save probabilities if needed
        overwrite=True,
        folder_with_segs_from_prev_stage=None
    )
    
    print("✅ MC dropout inference completed!")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    print("Both segmentation and uncertainty maps should be available.")

if __name__ == "__main__":
    main()
