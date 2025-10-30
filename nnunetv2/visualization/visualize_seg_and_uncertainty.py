#!/usr/bin/env python3
"""
Create visualization with BOTH segmentation contours AND uncertainty overlay.
Like the BraTS-PED-00260 reference image:
- Purple/blue contours for tumor segmentation
- Red/yellow heatmap for uncertainty
- Gray anatomical background

Usage on Amarel:
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate nnunetv2
    python visualize_seg_and_uncertainty.py
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
import glob

def create_contour_from_seg(segmentation):
    """Create contour/boundary from segmentation mask"""
    # Find edges using morphological operations
    contours = np.zeros_like(segmentation, dtype=bool)
    
    # Get unique labels (excluding background)
    labels = [l for l in np.unique(segmentation) if l > 0]
    
    for label in labels:
        mask = (segmentation == label)
        # Erode the mask
        eroded = ndimage.binary_erosion(mask, iterations=1)
        # Boundary is original minus eroded
        boundary = mask & ~eroded
        contours |= boundary
    
    return contours


def create_combined_visualization(orig_file, seg_file, unc_file, output_dir, case_name):
    """
    Create visualization with segmentation contours AND uncertainty overlay.
    
    Args:
        orig_file: Original anatomical image
        seg_file: Segmentation mask file
        unc_file: Uncertainty map file
        output_dir: Output directory
        case_name: Case identifier
    """
    print(f"\n{'='*70}")
    print(f"Processing: {case_name}")
    print(f"{'='*70}")
    
    # Load all files
    print("Loading images...")
    orig_nii = nib.load(orig_file)
    seg_nii = nib.load(seg_file)
    unc_nii = nib.load(unc_file)
    
    orig_data = orig_nii.get_fdata()
    seg_data = seg_nii.get_fdata().astype(int)
    unc_data = unc_nii.get_fdata()
    
    print(f"  Original shape: {orig_data.shape}")
    print(f"  Segmentation shape: {seg_data.shape}")
    print(f"  Uncertainty shape: {unc_data.shape}")
    
    # Fix axis permutation issue - uncertainty is (155, 240, 240) but should be (240, 240, 155)
    if unc_data.shape != orig_data.shape:
        print(f"  🔧 Fixing axis permutation...")
        print(f"     Before: unc shape = {unc_data.shape}, orig shape = {orig_data.shape}")
        
        # Uncertainty is (155, 240, 240) and we need (240, 240, 155)
        # Based on visual diagnostic test, (2, 1, 0) provides best alignment
        unc_data = np.transpose(unc_data, (2, 1, 0))
        print(f"     After transpose (2, 1, 0): {unc_data.shape}")
        
        # Verify it matches now
        if unc_data.shape == orig_data.shape:
            print(f"     ✅ Shapes now match!")
        else:
            print(f"     ❌ Still don't match - may need different permutation")
    
    # Check alignment
    if orig_data.shape != seg_data.shape:
        print(f"  ⚠️  WARNING: Original and segmentation shapes don't match!")
    if orig_data.shape != unc_data.shape:
        print(f"  ⚠️  WARNING: Original and uncertainty shapes STILL don't match after transpose!")
    
    # Normalize original for display (percentile-based)
    p1, p99 = np.percentile(orig_data[orig_data > 0], [1, 99])
    orig_display = np.clip((orig_data - p1) / (p99 - p1), 0, 1)
    
    # Normalize uncertainty - MUCH more aggressive to make it VERY visible
    unc_nonzero = unc_data[unc_data > 0]
    if len(unc_nonzero) > 0:
        # Use mean + std for normalization to make uncertainty pop out more
        unc_mean = unc_nonzero.mean()
        unc_std = unc_nonzero.std()
        unc_max = unc_nonzero.max()
        
        print(f"     Unc stats: mean={unc_mean:.6f}, std={unc_std:.6f}, max={unc_max:.6f}")
        
        # Normalize to make even small uncertainties visible
        # Anything above mean shows up clearly
        unc_normalized = np.clip((unc_data - unc_mean * 0.5) / (unc_std * 3 + 1e-8), 0, 1)
    else:
        unc_normalized = unc_data
        print(f"     ⚠️  No non-zero uncertainty values!")
    
    # Create uncertainty mask (threshold) - use VERY low threshold to show everything
    uncertainty_threshold = 0.0001  # Very low threshold to show all uncertainty
    unc_mask = unc_data > uncertainty_threshold
    
    print(f"     Voxels in uncertainty mask: {unc_mask.sum():,} ({100*unc_mask.sum()/unc_data.size:.1f}%)")
    
    print(f"\nUncertainty statistics:")
    print(f"  Range: [{unc_data.min():.4f}, {unc_data.max():.4f}]")
    print(f"  Mean: {unc_data.mean():.4f}")
    print(f"  Voxels > threshold: {unc_mask.sum():,} ({100*unc_mask.sum()/unc_data.size:.1f}%)")
    
    print(f"\nSegmentation statistics:")
    unique_labels = np.unique(seg_data)
    print(f"  Unique labels: {unique_labels}")
    for label in unique_labels:
        if label > 0:
            count = (seg_data == label).sum()
            print(f"    Label {label}: {count:,} voxels ({100*count/seg_data.size:.1f}%)")
    
    # ========================================================================
    # Find slice with most segmentation for best visualization
    # ========================================================================
    seg_per_slice = [(seg_data[:, :, i] > 0).sum() for i in range(seg_data.shape[2])]
    best_slice = np.argmax(seg_per_slice)
    # Make sure it's not at the edge
    best_slice = max(20, min(best_slice, seg_data.shape[2] - 20))
    
    print(f"\n  Best slice (most segmentation): z={best_slice}")
    
    # ========================================================================
    # MAIN VISUALIZATION: Anatomy + Seg Contours + Uncertainty
    # ========================================================================
    print(f"\nCreating combined visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{case_name} - Segmentation (Purple Contours) + Uncertainty (Red/Yellow Heatmap)',
                 fontsize=14, fontweight='bold')
    
    # Select 3 slices: best slice and neighbors
    slice_offsets = [-15, 0, 15]
    
    for idx, offset in enumerate(slice_offsets):
        ax = axes[idx]
        z = max(10, min(best_slice + offset, orig_data.shape[2] - 10))
        
        # Extract slices BEFORE transpose to ensure alignment
        orig_slice = orig_display[:, :, z]
        unc_slice = unc_normalized[:, :, z]
        seg_slice = seg_data[:, :, z]
        
        # Apply same transpose to all slices for consistent alignment
        orig_slice_T = orig_slice.T
        unc_slice_T = unc_slice.T
        seg_slice_T = seg_slice.T
        
        # 1. Show grayscale anatomy
        ax.imshow(orig_slice_T, cmap='gray', origin='lower', aspect='auto',
                  vmin=0, vmax=1)
        
        # 2. Overlay uncertainty (red/yellow heatmap with HIGH visibility)
        # Use 'hot' colormap: black (0) -> red (0.33) -> orange (0.66) -> yellow (1.0)
        im = ax.imshow(unc_slice_T, cmap='hot', origin='lower', aspect='auto',
                       alpha=0.9, vmin=0, vmax=0.7)  # Lower vmax to make colors brighter
        
        # 3. Add segmentation contours (purple/blue outlines)
        contours = create_contour_from_seg(seg_slice_T)
        
        # Create colored contours
        # Label 1 (NCR/NET) = purple
        # Label 2 (ED) = blue
        # Label 3 (ET) = orange
        contour_colors = np.zeros((*contours.shape, 4))  # RGBA
        
        for label in [1, 2, 3]:
            label_mask = (seg_slice_T == label)
            label_contour = create_contour_from_seg(label_mask)
            
            if label == 1:  # NCR/NET - Purple
                color = [0.6, 0.4, 0.8, 1.0]  # Purple
            elif label == 2:  # ED - Blue  
                color = [0.4, 0.6, 1.0, 1.0]  # Light blue
            elif label == 3:  # ET - Orange
                color = [1.0, 0.6, 0.2, 1.0]  # Orange
            
            contour_colors[label_contour] = color
        
        # Draw contours
        if contours.any():
            ax.imshow(contour_colors, origin='lower', aspect='auto')
        
        ax.set_title(f'z={z}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Add colorbar for uncertainty
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04, orientation='horizontal')
    cbar.set_label('Normalized Uncertainty (Black=Low, Red/Yellow=High)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{case_name}_seg_unc_combined.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {output_path}")
    
    # Grid visualization removed per user request
    
    print(f"✨ Completed {case_name}")
    
    return True


def main():
    """Main function"""
    
    # Amarel paths
    input_dir = '/scratch/hpl14/mc_dropout_test_3cases/input'
    output_dir_pred = '/scratch/hpl14/mc_dropout_test_3cases/output'
    output_dir = '/scratch/hpl14/mc_dropout_test_3cases/visualizations'
    
    print("="*80)
    print("SEGMENTATION + UNCERTAINTY COMBINED VISUALIZATION")
    print("="*80)
    print(f"Input directory:        {input_dir}")
    print(f"Prediction directory:   {output_dir_pred}")
    print(f"Output directory:       {output_dir}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all uncertainty files
    unc_files = sorted(glob.glob(os.path.join(output_dir_pred, '*_uncertainty.nii.gz')))
    
    print(f"\nFound {len(unc_files)} uncertainty files")
    
    if len(unc_files) == 0:
        print("❌ ERROR: No uncertainty files found!")
        return 1
    
    success_count = 0
    
    for unc_file in unc_files:
        # Extract case name
        case_name = os.path.basename(unc_file).replace('_uncertainty.nii.gz', '')
        
        # Find corresponding files
        orig_file = os.path.join(input_dir, f'{case_name}_0000.nii.gz')
        seg_file = os.path.join(output_dir_pred, f'{case_name}.nii.gz')
        
        # Check files exist
        if not os.path.exists(orig_file):
            print(f"⚠️  Skipping {case_name}: original file not found")
            continue
        
        if not os.path.exists(seg_file):
            print(f"⚠️  Skipping {case_name}: segmentation file not found")
            continue
        
        try:
            create_combined_visualization(orig_file, seg_file, unc_file,
                                         output_dir, case_name)
            success_count += 1
        except Exception as e:
            print(f"❌ ERROR processing {case_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Successfully processed: {success_count}/{len(unc_files)} cases")
    print(f"\n📁 Files saved to: {output_dir}")
    print(f"\n📥 Download with:")
    print(f"   scp 'hpl14@amarel.rutgers.edu:{output_dir}/*_seg_unc_*.png' .")
    print("="*80)
    
    return 0 if success_count == len(unc_files) else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
