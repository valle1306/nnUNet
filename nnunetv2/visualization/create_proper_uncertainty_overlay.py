#!/usr/bin/env python3
"""
Create proper colored uncertainty heatmap overlays like the BraTS-PED-00223 reference.
This script creates beautiful medical image visualizations with uncertainty shown as 
a colored heatmap (jet colormap) directly ON TOP of the anatomical grayscale image.

The visualization shows:
- Background: Grayscale anatomical brain image
- Overlay: Colored uncertainty heatmap (blue=low, red=high) with transparency
- Multiple slices: Axial, coronal, and sagittal views
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from pathlib import Path
import argparse

def create_beautiful_overlay(
    original_path: str,
    uncertainty_path: str,
    output_dir: str,
    case_name: str = None,
    uncertainty_threshold: float = 0.005,
    alpha: float = 0.7,
    percentile_norm: tuple = (1, 99)
):
    """
    Create beautiful colored uncertainty overlay visualization.
    
    Args:
        original_path: Path to original brain image (.nii.gz)
        uncertainty_path: Path to uncertainty map (.nii.gz)
        output_dir: Directory to save visualization
        case_name: Case identifier (extracted from filename if None)
        uncertainty_threshold: Minimum uncertainty to display (filter noise)
        alpha: Transparency of uncertainty overlay (0=transparent, 1=opaque)
        percentile_norm: Percentile range for normalizing uncertainty display
    """
    
    # Extract case name from filename if not provided
    if case_name is None:
        case_name = Path(original_path).name.split('_')[0]
    
    print(f"\n{'='*60}")
    print(f"Creating overlay for: {case_name}")
    print(f"{'='*60}")
    
    # Load images
    print("Loading images...")
    orig_nib = nib.load(original_path)
    unc_nib = nib.load(uncertainty_path)
    
    orig_data = orig_nib.get_fdata()
    unc_data = unc_nib.get_fdata()
    
    print(f"  Original shape: {orig_data.shape}")
    print(f"  Uncertainty shape: {unc_data.shape}")
    print(f"  Original affine:\n{orig_nib.affine}")
    print(f"  Uncertainty affine:\n{unc_nib.affine}")
    
    # Check if shapes match
    if orig_data.shape != unc_data.shape:
        print(f"  ⚠️  WARNING: Shape mismatch!")
        print(f"     This will cause spatial misalignment in the overlay.")
        print(f"     The uncertainty map needs to be fixed to match original shape.")
    else:
        print(f"  ✅ Shapes match!")
    
    # Check if affines match (approximately)
    affine_diff = np.abs(orig_nib.affine - unc_nib.affine).max()
    if affine_diff > 0.1:
        print(f"  ⚠️  WARNING: Affine mismatch (max diff: {affine_diff:.3f})")
    else:
        print(f"  ✅ Affines match (max diff: {affine_diff:.6f})")
    
    # Normalize original image for display (use percentile for better contrast)
    orig_min, orig_max = np.percentile(orig_data[orig_data > 0], [1, 99])
    orig_normalized = np.clip((orig_data - orig_min) / (orig_max - orig_min), 0, 1)
    
    # Normalize uncertainty for display
    unc_min, unc_max = np.percentile(unc_data[unc_data > 0], percentile_norm)
    unc_normalized = np.clip((unc_data - unc_min) / (unc_max - unc_min), 0, 1)
    
    # Create uncertainty mask (filter out very low uncertainty = noise)
    unc_mask = unc_data > uncertainty_threshold
    
    print(f"\nUncertainty statistics:")
    print(f"  Range: [{unc_data.min():.4f}, {unc_data.max():.4f}]")
    print(f"  Mean: {unc_data.mean():.4f}")
    print(f"  Std: {unc_data.std():.4f}")
    print(f"  Percentile ({percentile_norm[0]}%, {percentile_norm[1]}%): [{unc_min:.4f}, {unc_max:.4f}]")
    print(f"  Voxels above threshold ({uncertainty_threshold}): {unc_mask.sum()} / {unc_data.size} ({100*unc_mask.sum()/unc_data.size:.1f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== SINGLE SLICE VIEW (3 orientations) ====================
    print(f"\nCreating 3-view overlay...")
    
    # Choose middle slices
    axial_slice = orig_data.shape[2] // 2
    coronal_slice = orig_data.shape[1] // 2
    sagittal_slice = orig_data.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{case_name} - Uncertainty Overlay\n(Blue=Low Uncertainty, Red=High Uncertainty)', 
                 fontsize=16, fontweight='bold')
    
    # Axial view (z-axis)
    ax = axes[0]
    ax.imshow(orig_normalized[:, :, axial_slice].T, cmap='gray', origin='lower', aspect='auto')
    
    # Apply uncertainty overlay with transparency
    unc_overlay = ax.imshow(unc_normalized[:, :, axial_slice].T, 
                           cmap='jet', origin='lower', aspect='auto',
                           alpha=alpha * unc_mask[:, :, axial_slice].T.astype(float),
                           vmin=0, vmax=1)
    ax.set_title(f'Axial (z={axial_slice})', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Coronal view (y-axis)
    ax = axes[1]
    ax.imshow(orig_normalized[:, coronal_slice, :].T, cmap='gray', origin='lower', aspect='auto')
    ax.imshow(unc_normalized[:, coronal_slice, :].T, 
             cmap='jet', origin='lower', aspect='auto',
             alpha=alpha * unc_mask[:, coronal_slice, :].T.astype(float),
             vmin=0, vmax=1)
    ax.set_title(f'Coronal (y={coronal_slice})', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Sagittal view (x-axis)
    ax = axes[2]
    ax.imshow(orig_normalized[sagittal_slice, :, :].T, cmap='gray', origin='lower', aspect='auto')
    ax.imshow(unc_normalized[sagittal_slice, :, :].T, 
             cmap='jet', origin='lower', aspect='auto',
             alpha=alpha * unc_mask[sagittal_slice, :, :].T.astype(float),
             vmin=0, vmax=1)
    ax.set_title(f'Sagittal (x={sagittal_slice})', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(unc_overlay, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Normalized Uncertainty', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{case_name}_overlay_3views.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {output_path}")
    
    # ==================== MULTI-SLICE GRID (Axial slices) ====================
    print(f"Creating multi-slice grid...")
    
    # Select evenly spaced slices
    num_slices = 9
    slice_indices = np.linspace(20, orig_data.shape[2] - 20, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'{case_name} - Uncertainty Overlay (Multiple Axial Slices)\n' + 
                 f'Red/Yellow = High Uncertainty, Blue/Green = Low Uncertainty',
                 fontsize=16, fontweight='bold')
    
    for idx, slice_num in enumerate(slice_indices):
        ax = axes[idx // 3, idx % 3]
        
        # Show grayscale anatomy
        ax.imshow(orig_normalized[:, :, slice_num].T, cmap='gray', origin='lower', aspect='auto')
        
        # Overlay uncertainty with color
        unc_slice = unc_normalized[:, :, slice_num].T
        mask_slice = unc_mask[:, :, slice_num].T.astype(float)
        
        im = ax.imshow(unc_slice, cmap='jet', origin='lower', aspect='auto',
                      alpha=alpha * mask_slice, vmin=0, vmax=1)
        
        ax.set_title(f'Slice z={slice_num}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Add shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Normalized Uncertainty', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{case_name}_overlay_multislice.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {output_path}")
    
    # ==================== SINGLE BEAUTIFUL SLICE (like your reference) ====================
    print(f"Creating single beautiful slice...")
    
    # Find slice with highest uncertainty (most interesting)
    slice_uncertainties = [unc_data[:, :, i].max() for i in range(unc_data.shape[2])]
    best_slice = np.argmax(slice_uncertainties)
    
    # But make sure it's not too close to edges
    best_slice = max(30, min(best_slice, orig_data.shape[2] - 30))
    
    print(f"  Selected slice z={best_slice} (highest uncertainty: {slice_uncertainties[best_slice]:.4f})")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show anatomy in grayscale
    ax.imshow(orig_normalized[:, :, best_slice].T, cmap='gray', origin='lower', aspect='auto')
    
    # Overlay uncertainty
    unc_slice = unc_normalized[:, :, best_slice].T
    mask_slice = unc_mask[:, :, best_slice].T.astype(float)
    
    im = ax.imshow(unc_slice, cmap='jet', origin='lower', aspect='auto',
                   alpha=alpha * mask_slice, vmin=0, vmax=1)
    
    ax.set_title(f'{case_name} | z={best_slice}\nPredictive Uncertainty Heatmap',
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Uncertainty', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{case_name}_overlay_single_best.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {output_path}")
    
    print(f"\n✨ All visualizations created successfully!")
    return True


def process_all_cases(
    input_dir: str,
    output_dir: str,
    uncertainty_dir: str = None
):
    """
    Process all cases in a directory.
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save visualizations
        uncertainty_dir: Directory containing uncertainty maps (defaults to output subdir)
    """
    
    if uncertainty_dir is None:
        uncertainty_dir = input_dir
    
    # Find all original images
    input_path = Path(input_dir)
    uncertainty_path = Path(uncertainty_dir)
    
    # Look for files ending with .nii.gz but NOT ending with _uncertainty.nii.gz
    original_files = [f for f in input_path.glob('*.nii.gz') if '_uncertainty' not in f.name]
    
    print(f"\n{'='*70}")
    print(f"PROCESSING UNCERTAINTY OVERLAY VISUALIZATIONS")
    print(f"{'='*70}")
    print(f"Input directory: {input_dir}")
    print(f"Uncertainty directory: {uncertainty_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(original_files)} cases to process")
    print(f"{'='*70}\n")
    
    success_count = 0
    failed_cases = []
    
    for orig_file in sorted(original_files):
        # Extract case name (e.g., BraTS-PED-00215)
        # Remove .nii.gz to get the base name
        case_name = orig_file.name.replace('.nii.gz', '')
        
        # Find corresponding uncertainty file
        unc_file = uncertainty_path / f"{case_name}_uncertainty.nii.gz"
        
        if not unc_file.exists():
            print(f"⚠️  Skipping {case_name}: uncertainty file not found")
            print(f"   Looking for: {unc_file}")
            failed_cases.append(case_name)
            continue
        
        try:
            create_beautiful_overlay(
                original_path=str(orig_file),
                uncertainty_path=str(unc_file),
                output_dir=output_dir,
                case_name=case_name,
                uncertainty_threshold=0.005,
                alpha=0.7,
                percentile_norm=(1, 99)
            )
            success_count += 1
        except Exception as e:
            print(f"\n❌ ERROR processing {case_name}:")
            print(f"   {str(e)}")
            failed_cases.append(case_name)
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successfully processed: {success_count}/{len(original_files)} cases")
    if failed_cases:
        print(f"❌ Failed cases: {', '.join(failed_cases)}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create beautiful uncertainty overlay visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all cases in a directory
  python create_proper_uncertainty_overlay.py \\
      -i /path/to/input \\
      -u /path/to/uncertainty \\
      -o /path/to/output

  # Process single case
  python create_proper_uncertainty_overlay.py \\
      --original /path/to/BraTS-PED-00215.nii.gz \\
      --uncertainty /path/to/BraTS-PED-00215_uncertainty.nii.gz \\
      -o /path/to/output
        """
    )
    
    parser.add_argument('-i', '--input-dir', type=str,
                       help='Directory containing original images')
    parser.add_argument('-u', '--uncertainty-dir', type=str,
                       help='Directory containing uncertainty maps (defaults to input-dir)')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                       help='Directory to save visualizations')
    parser.add_argument('--original', type=str,
                       help='Single original image file')
    parser.add_argument('--uncertainty', type=str,
                       help='Single uncertainty map file')
    parser.add_argument('--threshold', type=float, default=0.005,
                       help='Uncertainty threshold for display (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Overlay transparency (0-1, default: 0.7)')
    
    args = parser.parse_args()
    
    # Single file mode
    if args.original and args.uncertainty:
        create_beautiful_overlay(
            original_path=args.original,
            uncertainty_path=args.uncertainty,
            output_dir=args.output_dir,
            uncertainty_threshold=args.threshold,
            alpha=args.alpha
        )
    
    # Batch mode
    elif args.input_dir:
        process_all_cases(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            uncertainty_dir=args.uncertainty_dir
        )
    
    else:
        parser.print_help()
        print("\n❌ Error: Must provide either --input-dir OR (--original AND --uncertainty)")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
