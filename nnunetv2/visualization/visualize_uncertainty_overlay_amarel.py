#!/usr/bin/env python3
"""
AMAREL UNCERTAINTY OVERLAY VISUALIZATION SCRIPT
Creates beautiful colored uncertainty overlays like BraTS-PED-00243 reference image.

This script:
1. Loads original anatomical images from input directory
2. Loads uncertainty maps from output directory  
3. Creates colored overlay visualizations (jet colormap: blue=low, red=high)
4. Saves PNG files to visualizations directory for download

Usage on Amarel:
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate nnunetv2
    python visualize_uncertainty_overlay_amarel.py

Then download the results:
    scp hpl14@amarel.rutgers.edu:/scratch/hpl14/mc_dropout_test_3cases/visualizations/*.png .

Requirements:
    - nnunetv2 conda environment must be activated
    - numpy, nibabel, matplotlib must be available
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def create_overlay_visualization(orig_file, unc_file, output_dir, case_name):
    """
    Create beautiful overlay visualization with uncertainty as colored heatmap on grayscale anatomy.
    
    Args:
        orig_file: Path to original anatomical image
        unc_file: Path to uncertainty map
        output_dir: Directory to save PNG visualizations
        case_name: Case identifier (e.g., BraTS-PED-00243)
    """
    print(f"\n{'='*70}")
    print(f"Processing: {case_name}")
    print(f"{'='*70}")
    
    # Load NIfTI files
    print("Loading images...")
    orig_nii = nib.load(orig_file)
    unc_nii = nib.load(unc_file)
    
    orig_data = orig_nii.get_fdata()
    unc_data = unc_nii.get_fdata()
    
    print(f"  Original shape: {orig_data.shape}")
    print(f"  Uncertainty shape: {unc_data.shape}")
    
    # Check alignment
    orig_affine = orig_nii.affine
    unc_affine = unc_nii.affine
    affine_match = np.allclose(orig_affine, unc_affine, rtol=1e-3, atol=1e-3)
    
    print(f"  Affines match: {affine_match}")
    if not affine_match:
        print("  ⚠️  WARNING: Affine matrices differ!")
        print(f"  Max affine difference: {np.abs(orig_affine - unc_affine).max():.6f}")
    
    # Check shape alignment
    shape_match = orig_data.shape == unc_data.shape
    print(f"  Shapes match: {shape_match}")
    if not shape_match:
        print("  ⚠️  WARNING: Shapes differ - overlay will be misaligned!")
    
    # Uncertainty statistics
    print(f"\nUncertainty statistics:")
    print(f"  Min:    {unc_data.min():.6f}")
    print(f"  Max:    {unc_data.max():.6f}")
    print(f"  Mean:   {unc_data.mean():.6f}")
    print(f"  Median: {np.median(unc_data):.6f}")
    print(f"  Std:    {unc_data.std():.6f}")
    print(f"  95th percentile: {np.percentile(unc_data, 95):.6f}")
    print(f"  99th percentile: {np.percentile(unc_data, 99):.6f}")
    
    # Normalize original image for display (using percentiles for better contrast)
    print("\nNormalizing images for display...")
    orig_display = orig_data.copy()
    # Use robust normalization (ignore extreme outliers)
    p1, p99 = np.percentile(orig_display[orig_display > 0], [1, 99])
    orig_display = np.clip((orig_display - p1) / (p99 - p1), 0, 1)
    
    # Normalize uncertainty for color mapping
    # Use percentile-based normalization to avoid outliers dominating the colormap
    unc_p1, unc_p99 = np.percentile(unc_data[unc_data > 0], [1, 99])
    unc_normalized = np.clip((unc_data - unc_p1) / (unc_p99 - unc_p1), 0, 1)
    
    # Create uncertainty mask (threshold out very low uncertainty = noise)
    uncertainty_threshold = 0.005
    unc_mask = unc_data > uncertainty_threshold
    print(f"  Voxels above threshold ({uncertainty_threshold}): {unc_mask.sum():,} / {unc_data.size:,} ({100*unc_mask.sum()/unc_data.size:.1f}%)")
    
    # ========================================================================
    # VISUALIZATION 1: Three orthogonal views (like medical imaging software)
    # ========================================================================
    print("\nCreating 3-view overlay...")
    
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f'{case_name} - Uncertainty Overlay\nBlue=Low Uncertainty, Red=High Uncertainty',
                 fontsize=16, fontweight='bold')
    
    # Select middle slices
    z_mid = orig_data.shape[2] // 2
    y_mid = orig_data.shape[1] // 2
    x_mid = orig_data.shape[0] // 2
    
    # Axial view (looking down, z-axis)
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(orig_display[:, :, z_mid].T, cmap='gray', origin='lower', aspect='auto')
    unc_overlay = np.ma.masked_where(~unc_mask[:, :, z_mid].T, unc_normalized[:, :, z_mid].T)
    im = ax1.imshow(unc_overlay, cmap='jet', origin='lower', aspect='auto',
                    alpha=0.7, vmin=0, vmax=1)
    ax1.set_title(f'Axial (z={z_mid})', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Coronal view (looking from front, y-axis)
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(orig_display[:, y_mid, :].T, cmap='gray', origin='lower', aspect='auto')
    unc_overlay = np.ma.masked_where(~unc_mask[:, y_mid, :].T, unc_normalized[:, y_mid, :].T)
    ax2.imshow(unc_overlay, cmap='jet', origin='lower', aspect='auto',
               alpha=0.7, vmin=0, vmax=1)
    ax2.set_title(f'Coronal (y={y_mid})', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Sagittal view (looking from side, x-axis)
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(orig_display[x_mid, :, :].T, cmap='gray', origin='lower', aspect='auto')
    unc_overlay = np.ma.masked_where(~unc_mask[x_mid, :, :].T, unc_normalized[x_mid, :, :].T)
    ax3.imshow(unc_overlay, cmap='jet', origin='lower', aspect='auto',
               alpha=0.7, vmin=0, vmax=1)
    ax3.set_title(f'Sagittal (x={x_mid})', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=[ax1, ax2, ax3], fraction=0.02, pad=0.04, orientation='horizontal')
    cbar.set_label('Normalized Uncertainty', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{case_name}_overlay_3views.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {output_path}")
    
    # ========================================================================
    # VISUALIZATION 2: Multi-slice grid (9 axial slices)
    # ========================================================================
    print("Creating multi-slice grid...")
    
    # Select evenly spaced slices (avoid empty edges)
    num_slices = 9
    slice_start = max(10, orig_data.shape[2] // 10)
    slice_end = min(orig_data.shape[2] - 10, 9 * orig_data.shape[2] // 10)
    slice_indices = np.linspace(slice_start, slice_end, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'{case_name} - Multi-Slice Uncertainty Overlay (Axial)\n' +
                 'Red/Yellow = High Uncertainty, Blue/Green = Low Uncertainty',
                 fontsize=16, fontweight='bold')
    
    for idx, z_slice in enumerate(slice_indices):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Show grayscale anatomy
        ax.imshow(orig_display[:, :, z_slice].T, cmap='gray', origin='lower', aspect='auto')
        
        # Overlay colored uncertainty (masked by threshold)
        unc_overlay = np.ma.masked_where(~unc_mask[:, :, z_slice].T, 
                                         unc_normalized[:, :, z_slice].T)
        im = ax.imshow(unc_overlay, cmap='jet', origin='lower', aspect='auto',
                       alpha=0.7, vmin=0, vmax=1)
        
        ax.set_title(f'Slice z={z_slice}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label('Normalized Uncertainty', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{case_name}_overlay_multislice.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {output_path}")
    
    # ========================================================================
    # VISUALIZATION 3: Single "best" slice (highest uncertainty = most interesting)
    # ========================================================================
    print("Creating single best slice visualization...")
    
    # Find slice with highest uncertainty
    slice_uncertainties = [unc_data[:, :, i].max() for i in range(unc_data.shape[2])]
    best_slice = np.argmax(slice_uncertainties)
    
    # Avoid edges
    best_slice = max(20, min(best_slice, orig_data.shape[2] - 20))
    
    print(f"  Selected slice z={best_slice} (max uncertainty: {slice_uncertainties[best_slice]:.6f})")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show anatomy
    ax.imshow(orig_display[:, :, best_slice].T, cmap='gray', origin='lower', aspect='auto')
    
    # Overlay uncertainty
    unc_overlay = np.ma.masked_where(~unc_mask[:, :, best_slice].T,
                                     unc_normalized[:, :, best_slice].T)
    im = ax.imshow(unc_overlay, cmap='jet', origin='lower', aspect='auto',
                   alpha=0.7, vmin=0, vmax=1)
    
    ax.set_title(f'{case_name} | z={best_slice}\nPredictive Uncertainty Heatmap',
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Uncertainty', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{case_name}_overlay_best_slice.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {output_path}")
    
    print(f"✨ Completed processing {case_name}")
    
    return affine_match, shape_match


def main():
    """Main function to process all cases."""
    
    # Amarel directory paths
    input_dir = '/scratch/hpl14/mc_dropout_test_3cases/input'
    output_dir_pred = '/scratch/hpl14/mc_dropout_test_3cases/output'
    output_dir = '/scratch/hpl14/mc_dropout_test_3cases/visualizations'
    
    print("="*80)
    print("UNCERTAINTY OVERLAY VISUALIZATION")
    print("="*80)
    print(f"Input directory:        {input_dir}")
    print(f"Uncertainty directory:  {output_dir_pred}")
    print(f"Output directory:       {output_dir}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all uncertainty files
    unc_pattern = os.path.join(output_dir_pred, '*_uncertainty.nii.gz')
    unc_files = sorted(glob.glob(unc_pattern))
    
    print(f"\nFound {len(unc_files)} uncertainty files to process:")
    for unc_file in unc_files:
        print(f"  - {os.path.basename(unc_file)}")
    
    if len(unc_files) == 0:
        print("\n❌ ERROR: No uncertainty files found!")
        print(f"   Looking for: {unc_pattern}")
        print("\n   Make sure MC dropout prediction has completed successfully.")
        return 1
    
    # Process each case
    results = []
    
    for unc_file in unc_files:
        # Extract case name (e.g., BraTS-PED-00243)
        case_name = os.path.basename(unc_file).replace('_uncertainty.nii.gz', '')
        
        # Find corresponding original file (use _0000 channel)
        orig_file = os.path.join(input_dir, f'{case_name}_0000.nii.gz')
        
        if not os.path.exists(orig_file):
            print(f"\n⚠️  WARNING: Original file not found for {case_name}")
            print(f"   Looking for: {orig_file}")
            results.append((case_name, False, False, "Original file not found"))
            continue
        
        try:
            affine_match, shape_match = create_overlay_visualization(
                orig_file, unc_file, output_dir, case_name
            )
            results.append((case_name, affine_match, shape_match, "Success"))
        except Exception as e:
            print(f"\n❌ ERROR processing {case_name}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((case_name, False, False, f"Error: {str(e)}"))
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Processed {len(results)} case(s)\n")
    
    for case_name, affine_match, shape_match, status in results:
        if status == "Success":
            print(f"✅ {case_name}")
            print(f"   Affine match: {affine_match}")
            print(f"   Shape match:  {shape_match}")
        else:
            print(f"❌ {case_name}: {status}")
    
    # Overall status
    success_count = sum(1 for _, _, _, status in results if status == "Success")
    all_affines_match = all(affine for _, affine, _, status in results if status == "Success")
    all_shapes_match = all(shape for _, _, shape, status in results if status == "Success")
    
    print(f"\n{'='*80}")
    print(f"✅ Successfully processed: {success_count}/{len(results)} cases")
    
    if success_count > 0:
        if all_affines_match and all_shapes_match:
            print("✅ All spatial alignments are CORRECT!")
        else:
            if not all_affines_match:
                print("⚠️  WARNING: Some affine matrices don't match")
            if not all_shapes_match:
                print("⚠️  WARNING: Some shapes don't match")
    
    print(f"\n📁 Visualization files saved to:")
    print(f"   {output_dir}")
    print(f"\n📥 Download with:")
    print(f"   scp hpl14@amarel.rutgers.edu:{output_dir}/*.png .")
    print("="*80)
    
    return 0 if success_count == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
