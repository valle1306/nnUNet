"""
Simple uncertainty rotation tester - try each rotation manually
This script shows you each possible rotation/flip so you can visually identify the correct one.
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def load_and_extract_slice():
    """Load data and extract a good slice for testing"""
    print("Loading data...")
    
    # Load images
    raw_img_nii = nib.load('BraTS-PED-00062-0000.nii.gz')
    gt_seg_nii = nib.load('BraTS-PED-00062-seg.nii.gz')
    pred_seg_nii = nib.load('BraTS-PED-00062.nii.gz')
    uncertainty_nii = nib.load('uncertainty.nii.gz')
    
    raw_img = raw_img_nii.get_fdata()
    gt_seg = gt_seg_nii.get_fdata()
    pred_seg = pred_seg_nii.get_fdata()
    uncertainty = uncertainty_nii.get_fdata()
    
    print(f"Raw image shape: {raw_img.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    
    # Find slice with tumor
    tumor_slices = np.where(np.any(gt_seg > 0, axis=(0, 1)))[0]
    if len(tumor_slices) == 0:
        slice_idx = raw_img.shape[2] // 2
    else:
        tumor_areas = [np.sum(gt_seg[:,:,s] > 0) for s in tumor_slices]
        slice_idx = tumor_slices[np.argmax(tumor_areas)]
    
    print(f"Using slice {slice_idx}")
    
    # Extract slices
    raw_slice = raw_img[:, :, slice_idx]
    gt_slice = gt_seg[:, :, slice_idx]
    pred_slice = pred_seg[:, :, slice_idx]
    
    # Handle uncertainty extraction based on dimensions
    if uncertainty.ndim == 4:
        if uncertainty.shape[1:] == pred_seg.shape:
            # Channel first
            uncert_slice = uncertainty[0, :, :, slice_idx]
        elif uncertainty.shape[:-1] == pred_seg.shape:
            # Channel last
            uncert_slice = uncertainty[:, :, slice_idx, 0]
        else:
            uncert_slice = uncertainty[0, :, :, slice_idx]
    else:
        uncert_slice = uncertainty[:, :, slice_idx]
    
    return raw_slice, gt_slice, pred_slice, uncert_slice

def test_basic_rotations(raw_slice, uncert_slice_original):
    """Test basic 90-degree rotations"""
    print("\n=== Testing Basic 90-Degree Rotations ===")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Basic Rotations Test', fontsize=16)
    
    rotations = [
        (uncert_slice_original, "Original"),
        (np.rot90(uncert_slice_original, 1), "90° clockwise"),
        (np.rot90(uncert_slice_original, 2), "180°"),
        (np.rot90(uncert_slice_original, 3), "270° (or 90° counter-clockwise)")
    ]
    
    for i, (uncert, name) in enumerate(rotations):
        # Top row: uncertainty alone
        axes[0, i].imshow(uncert, cmap='hot', vmin=0, vmax=np.percentile(uncert, 95))
        axes[0, i].set_title(f'{name}\nShape: {uncert.shape}')
        axes[0, i].axis('off')
        
        # Bottom row: overlay with raw image (if shapes match)
        if uncert.shape == raw_slice.shape:
            axes[1, i].imshow(raw_slice, cmap='gray')
            axes[1, i].imshow(uncert, cmap='hot', alpha=0.5, vmin=0, vmax=np.percentile(uncert, 95))
            axes[1, i].set_title('Overlay (✓)')
            axes[1, i].axis('off')
            # Add green border
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
        else:
            axes[1, i].text(0.5, 0.5, f'Shape mismatch\n{uncert.shape}', 
                           ha='center', va='center', transform=axes[1, i].transAxes,
                           color='red', fontsize=12)
            axes[1, i].set_title('Wrong shape (✗)')
            # Add red border
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.show()

def test_flips(raw_slice, uncert_slice_original):
    """Test different flip operations"""
    print("\n=== Testing Flip Operations ===")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Flip Operations Test', fontsize=16)
    
    flips = [
        (uncert_slice_original, "Original"),
        (np.flipud(uncert_slice_original), "Flip up-down"),
        (np.fliplr(uncert_slice_original), "Flip left-right"),
        (np.flipud(np.fliplr(uncert_slice_original)), "Flip both")
    ]
    
    for i, (uncert, name) in enumerate(flips):
        # Top row: uncertainty alone
        axes[0, i].imshow(uncert, cmap='hot', vmin=0, vmax=np.percentile(uncert, 95))
        axes[0, i].set_title(f'{name}\nShape: {uncert.shape}')
        axes[0, i].axis('off')
        
        # Bottom row: overlay with raw image
        if uncert.shape == raw_slice.shape:
            axes[1, i].imshow(raw_slice, cmap='gray')
            axes[1, i].imshow(uncert, cmap='hot', alpha=0.5, vmin=0, vmax=np.percentile(uncert, 95))
            axes[1, i].set_title('Overlay (✓)')
            axes[1, i].axis('off')
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
        else:
            axes[1, i].text(0.5, 0.5, f'Shape mismatch\n{uncert.shape}', 
                           ha='center', va='center', transform=axes[1, i].transAxes,
                           color='red', fontsize=12)
            axes[1, i].set_title('Wrong shape (✗)')
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.show()

def test_transpose_combinations(raw_slice, uncert_slice_original):
    """Test transpose combined with rotations"""
    print("\n=== Testing Transpose + Rotation Combinations ===")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Transpose + Rotation Combinations Test', fontsize=16)
    
    combinations = [
        (uncert_slice_original.T, "Transpose only"),
        (np.rot90(uncert_slice_original.T, 1), "Transpose + 90°"),
        (np.rot90(uncert_slice_original.T, 2), "Transpose + 180°"),
        (np.rot90(uncert_slice_original.T, 3), "Transpose + 270°")
    ]
    
    for i, (uncert, name) in enumerate(combinations):
        # Top row: uncertainty alone
        axes[0, i].imshow(uncert, cmap='hot', vmin=0, vmax=np.percentile(uncert, 95))
        axes[0, i].set_title(f'{name}\nShape: {uncert.shape}')
        axes[0, i].axis('off')
        
        # Bottom row: overlay with raw image
        if uncert.shape == raw_slice.shape:
            axes[1, i].imshow(raw_slice, cmap='gray')
            axes[1, i].imshow(uncert, cmap='hot', alpha=0.5, vmin=0, vmax=np.percentile(uncert, 95))
            axes[1, i].set_title('Overlay (✓)')
            axes[1, i].axis('off')
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
        else:
            axes[1, i].text(0.5, 0.5, f'Shape mismatch\n{uncert.shape}', 
                           ha='center', va='center', transform=axes[1, i].transAxes,
                           color='red', fontsize=12)
            axes[1, i].set_title('Wrong shape (✗)')
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.show()

def show_reference_images(raw_slice, gt_slice, pred_slice):
    """Show reference images for comparison"""
    print("\n=== Reference Images ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Reference Images for Comparison', fontsize=16)
    
    # Raw image
    axes[0].imshow(raw_slice, cmap='gray')
    axes[0].set_title(f'Raw Image\nShape: {raw_slice.shape}')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(raw_slice, cmap='gray')
    axes[1].imshow(gt_slice > 0, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth Overlay')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(raw_slice, cmap='gray')
    axes[2].imshow(pred_slice > 0, cmap='Blues', alpha=0.5)
    axes[2].set_title('Prediction Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def manual_test_specific_rotation(raw_slice, uncert_slice_original, rotation_code):
    """Test a specific rotation manually"""
    print(f"\n=== Testing specific rotation: {rotation_code} ===")
    
    # Apply the rotation
    try:
        uncert_transformed = eval(rotation_code.replace('uncert_slice_original', 'uncert_slice_original'))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Manual Test: {rotation_code}', fontsize=14)
        
        # Uncertainty alone
        axes[0].imshow(uncert_transformed, cmap='hot', vmin=0, vmax=np.percentile(uncert_transformed, 95))
        axes[0].set_title(f'Uncertainty\nShape: {uncert_transformed.shape}')
        axes[0].axis('off')
        
        # Raw image reference
        axes[1].imshow(raw_slice, cmap='gray')
        axes[1].set_title(f'Raw Image\nShape: {raw_slice.shape}')
        axes[1].axis('off')
        
        # Overlay
        if uncert_transformed.shape == raw_slice.shape:
            axes[2].imshow(raw_slice, cmap='gray')
            axes[2].imshow(uncert_transformed, cmap='hot', alpha=0.6, vmin=0, vmax=np.percentile(uncert_transformed, 95))
            axes[2].set_title('Overlay ✓')
            axes[2].axis('off')
            print("✅ Shapes match! This transformation works.")
        else:
            axes[2].text(0.5, 0.5, f'Shape mismatch\nUncertainty: {uncert_transformed.shape}\nRaw: {raw_slice.shape}', 
                        ha='center', va='center', transform=axes[2].transAxes, color='red')
            axes[2].set_title('Shape mismatch ✗')
            print(f"❌ Shape mismatch: uncertainty {uncert_transformed.shape} vs raw {raw_slice.shape}")
        
        plt.tight_layout()
        plt.show()
        
        return uncert_transformed
        
    except Exception as e:
        print(f"❌ Error applying transformation: {e}")
        return None

def print_transformation_codes():
    """Print all the transformation codes for easy copy-pasting"""
    print("\n" + "="*60)
    print("TRANSFORMATION CODES - Copy and paste to test manually:")
    print("="*60)
    
    codes = [
        "uncert_slice_original",                           # Original
        "uncert_slice_original.T",                         # Transpose
        "np.rot90(uncert_slice_original, 1)",             # 90° rotation
        "np.rot90(uncert_slice_original, 2)",             # 180° rotation
        "np.rot90(uncert_slice_original, 3)",             # 270° rotation
        "np.rot90(uncert_slice_original.T, 1)",           # Transpose + 90°
        "np.rot90(uncert_slice_original.T, 2)",           # Transpose + 180°
        "np.rot90(uncert_slice_original.T, 3)",           # Transpose + 270°
        "np.flipud(uncert_slice_original)",               # Flip up-down
        "np.fliplr(uncert_slice_original)",               # Flip left-right
        "np.flipud(np.fliplr(uncert_slice_original))",    # Flip both
    ]
    
    names = [
        "Original (no change)",
        "Transpose (swap x,y axes)",
        "90° clockwise rotation",
        "180° rotation",
        "270° clockwise (= 90° counter-clockwise)",
        "Transpose + 90° clockwise",
        "Transpose + 180°",
        "Transpose + 270° clockwise",
        "Flip up-down (vertical flip)",
        "Flip left-right (horizontal flip)",
        "Flip both directions",
    ]
    
    for i, (code, name) in enumerate(zip(codes, names)):
        print(f"{i+1:2d}. {name}")
        print(f"    {code}")
        print()

def main():
    """Main function"""
    print("🔄 Simple Uncertainty Rotation Tester")
    print("="*50)
    
    try:
        # Load data
        raw_slice, gt_slice, pred_slice, uncert_slice_original = load_and_extract_slice()
        
        print(f"Target shape (raw image): {raw_slice.shape}")
        print(f"Original uncertainty shape: {uncert_slice_original.shape}")
        
        # Show reference images first
        show_reference_images(raw_slice, gt_slice, pred_slice)
        
        # Test basic rotations
        test_basic_rotations(raw_slice, uncert_slice_original)
        
        # Test flips
        test_flips(raw_slice, uncert_slice_original)
        
        # Test transpose combinations
        test_transpose_combinations(raw_slice, uncert_slice_original)
        
        # Print transformation codes
        print_transformation_codes()
        
        # Interactive testing
        print("\n🎯 MANUAL TESTING:")
        print("Look at the plots above to identify which transformation looks best.")
        print("Then test specific transformations using the codes below.\n")
        
        # Example manual tests
        print("Example: Testing a few common transformations...")
        
        # Test 90-degree rotation
        manual_test_specific_rotation(raw_slice, uncert_slice_original, "np.rot90(uncert_slice_original, 1)")
        
        # Test transpose
        manual_test_specific_rotation(raw_slice, uncert_slice_original, "uncert_slice_original.T")
        
        print("\n✅ NEXT STEPS:")
        print("1. Look at all the plots above")
        print("2. Identify which transformation makes the uncertainty align with brain structures")
        print("3. Copy the corresponding transformation code")
        print("4. Use it in your main visualization script")
        print("\nExample usage in your script:")
        print("uncert_slice = np.rot90(uncert_slice, 1)  # Replace with your chosen transformation")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you're in the directory with your files:")
        print("- BraTS-PED-00062-0000.nii.gz")
        print("- BraTS-PED-00062-seg.nii.gz")
        print("- BraTS-PED-00062.nii.gz") 
        print("- uncertainty.nii.gz")

if __name__ == "__main__":
    main()
