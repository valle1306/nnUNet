import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


# Load images and print shapes/affines for debugging
raw_img_nii = nib.load('BraTS-PED-00062-0000.nii.gz')
gt_seg_nii = nib.load('BraTS-PED-00062-seg.nii.gz')
pred_seg_nii = nib.load('predictions.nii.gz')
uncertainty_nii = nib.load('uncertainty.nii.gz')

raw_img = raw_img_nii.get_fdata()
gt_seg = gt_seg_nii.get_fdata()
pred_seg = pred_seg_nii.get_fdata()
uncertainty = uncertainty_nii.get_fdata()

print("raw_img shape:", raw_img.shape)
print("gt_seg shape:", gt_seg.shape)
print("pred_seg shape:", pred_seg.shape)
print("uncertainty shape:", uncertainty.shape)
print("raw_img affine:\n", raw_img_nii.affine)
print("gt_seg affine:\n", gt_seg_nii.affine)
print("pred_seg affine:\n", pred_seg_nii.affine)
print("uncertainty affine:\n", uncertainty_nii.affine)

# Check for orientation differences in affine matrices
print("\n=== Orientation Analysis ===")
if not np.allclose(raw_img_nii.affine, uncertainty_nii.affine, atol=1e-3):
    print("⚠️  WARNING: Raw image and uncertainty have different affine matrices!")
    print("This may indicate different orientations or spacing.")
    print("Raw image affine:\n", raw_img_nii.affine)
    print("Uncertainty affine:\n", uncertainty_nii.affine)
    
    # Try to fix the uncertainty orientation by reorienting it to match the raw image
    print("Attempting to reorient uncertainty to match raw image...")
    try:
        import nibabel.orientations as nio
        
        # Get orientation of both images
        raw_ornt = nio.io_orientation(raw_img_nii.affine)
        uncert_ornt = nio.io_orientation(uncertainty_nii.affine)
        
        print(f"Raw image orientation: {raw_ornt}")
        print(f"Uncertainty orientation: {uncert_ornt}")
        
        # Calculate transformation to match orientations
        transform = nio.ornt_transform(uncert_ornt, raw_ornt)
        
        # Apply orientation transformation
        uncertainty_reoriented = nio.apply_orientation(uncertainty, transform)
        
        print(f"Original uncertainty shape: {uncertainty.shape}")
        print(f"Reoriented uncertainty shape: {uncertainty_reoriented.shape}")
        
        # Update uncertainty with reoriented version
        uncertainty = uncertainty_reoriented
        print("✅ Successfully reoriented uncertainty to match raw image")
        
    except Exception as e:
        print(f"❌ Could not automatically reorient: {e}")
        print("Will use manual transpose fix instead")
else:
    print("✅ Raw image and uncertainty have matching affine matrices.")

# Find a slice with tumor in ground truth
tumor_slices = np.where(np.any(gt_seg > 0, axis=(0, 1)))[0]
if len(tumor_slices) == 0:
    print("No tumor found in ground truth segmentation!")
    slice_idx = raw_img.shape[2] // 2
else:
    # Find slice with good tumor visibility (substantial tumor area)
    tumor_areas = []
    for s in tumor_slices:
        tumor_area = np.sum(gt_seg[:,:,s] > 0)
        tumor_areas.append(tumor_area)
    
    # Use slice with largest tumor area (or middle if you prefer)
    best_slice_idx = np.argmax(tumor_areas)
    slice_idx = tumor_slices[best_slice_idx]
    print(f"Found {len(tumor_slices)} tumor slices, using slice {slice_idx} with largest tumor area")



print(f"Visualizing slice {slice_idx} (contains tumor)")

# Extract slices - CRITICAL: Use the exact same indexing for prediction and uncertainty
raw_slice = raw_img[:,:,slice_idx]
gt_slice = gt_seg[:,:,slice_idx]
pred_slice = pred_seg[:,:,slice_idx]

# Highlight tumor region on FLAIR (raw) image with proper tumor classes
plt.figure(figsize=(10, 8))
plt.imshow(raw_slice, cmap='gray')

# Create a colored overlay for different tumor regions
# BraTS classes: 0=background, 1=necrotic, 2=edema, 3=enhancing
# Custom color mapping: edema (green, outside), enhancing/necrotic (red, inside)
tumor_overlay = np.zeros((*gt_slice.shape, 3))  # RGB image
tumor_overlay[gt_slice == 1] = [1, 0, 0]  # Necrotic core -> red
tumor_overlay[gt_slice == 2] = [0, 1, 0]  # Edema -> green  
tumor_overlay[gt_slice == 3] = [1, 0, 0]  # Enhancing tumor -> red
tumor_overlay[gt_slice == 4] = [1, 0, 0]  # Additional enhancing -> red (if exists)

# Only show overlay where there are tumors
tumor_mask = gt_slice > 0
if np.any(tumor_mask):
    plt.imshow(tumor_overlay, alpha=0.6)

plt.title(f'FLAIR with Tumor Classes Highlighted (Slice {slice_idx})')
plt.axis('off')

# Add a simple legend with correct color mapping
unique_classes = np.unique(gt_slice[gt_slice > 0])
if len(unique_classes) > 0:
    legend_text = []
    class_names = {1: 'Necrotic (red)', 2: 'Edema (green)', 3: 'Enhancing (red)', 4: 'Enhancing (red)'}
    for cls in unique_classes:
        if cls in class_names:
            legend_text.append(f'Class {int(cls)}: {class_names[cls]}')
    if legend_text:
        plt.text(0.02, 0.98, '\n'.join(legend_text), transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                color='white', fontsize=10, verticalalignment='top')

plt.show()

# Extract uncertainty slice using THE SAME logic as prediction slice
# Since pred_seg uses [:,:,slice_idx], uncertainty should use the same
print(f"Prediction shape: {pred_seg.shape}")
print(f"Uncertainty shape: {uncertainty.shape}")

if uncertainty.shape == pred_seg.shape:
    # Same shape, same indexing
    uncert_slice = uncertainty[:,:,slice_idx]
    print("✅ Using same indexing as prediction: [:,:,slice_idx]")
elif uncertainty.ndim == 4 and pred_seg.ndim == 3:
    # Uncertainty is 4D (has channel dimension), prediction is 3D
    if uncertainty.shape[1:] == pred_seg.shape:
        # Channel is first dimension
        uncert_slice = uncertainty[0,:,:,slice_idx]
        print("✅ Using 4D indexing: [0,:,:,slice_idx]")
    elif uncertainty.shape[:-1] == pred_seg.shape:
        # Channel is last dimension
        uncert_slice = uncertainty[:,:,slice_idx,0]
        print("✅ Using 4D indexing: [:,:,slice_idx,0]")
    else:
        print("❌ Warning: Uncertainty 4D shape doesn't match prediction 3D shape")
        uncert_slice = uncertainty[0,:,:,slice_idx]  # Best guess
else:
    # Default fallback
    uncert_slice = uncertainty[:,:,slice_idx]
    print("⚠️  Using default indexing [:,:,slice_idx]")

# Create masks for analysis
pred_mask = pred_slice > 0
background_mask = pred_slice == 0

# Check if uncertainty needs to be oriented to match image orientation
print(f"Raw image slice shape: {raw_slice.shape}")
print(f"Prediction slice shape: {pred_slice.shape}")
print(f"Uncertainty slice shape: {uncert_slice.shape}")

# Verify that prediction and raw image align (they should!)
if pred_slice.shape != raw_slice.shape:
    print("❌ WARNING: Prediction and raw image shapes don't match!")
    print(f"  Raw: {raw_slice.shape}, Pred: {pred_slice.shape}")
else:
    print("✅ Prediction and raw image shapes match")

def find_best_orientation_match(uncert_slice, reference_slice, pred_slice, method='correlation'):
    """
    Find the best orientation of uncertainty slice to match the reference image
    using correlation with the prediction as a guide.
    """
    if uncert_slice.shape != reference_slice.shape:
        print(f"Shape mismatch: uncertainty {uncert_slice.shape} vs reference {reference_slice.shape}")
        
        # Generate all possible orientations
        orientations = [
            uncert_slice,  # Original
            uncert_slice.T,  # Transpose
            np.rot90(uncert_slice, 1),  # 90° rotation
            np.rot90(uncert_slice, 2),  # 180° rotation  
            np.rot90(uncert_slice, 3),  # 270° rotation
            np.rot90(uncert_slice.T, 1),  # Transpose + 90°
            np.rot90(uncert_slice.T, 2),  # Transpose + 180°
            np.rot90(uncert_slice.T, 3),  # Transpose + 270°
        ]
        
        orientation_names = ['original', 'transpose', 'rot90', 'rot180', 'rot270', 'T+rot90', 'T+rot180', 'T+rot270']
        
        best_score = -1
        best_orientation = uncert_slice
        best_name = 'original'
        
        # Create a mask of predicted regions to guide the matching
        pred_mask = pred_slice > 0
        
        for i, orient in enumerate(orientations):
            if orient.shape == reference_slice.shape:
                if method == 'correlation' and np.any(pred_mask):
                    # Calculate correlation between uncertainty and prediction boundaries
                    # High uncertainty should correlate with prediction boundaries
                    pred_edges = np.gradient(pred_slice.astype(float))
                    pred_edge_magnitude = np.sqrt(pred_edges[0]**2 + pred_edges[1]**2)
                    
                    # Normalize both for correlation
                    if orient.std() > 0 and pred_edge_magnitude.std() > 0:
                        orient_norm = (orient - orient.mean()) / orient.std()
                        edges_norm = (pred_edge_magnitude - pred_edge_magnitude.mean()) / pred_edge_magnitude.std()
                        score = np.corrcoef(orient_norm.flatten(), edges_norm.flatten())[0,1]
                        if not np.isnan(score):
                            print(f"  {orientation_names[i]}: correlation = {score:.3f}")
                            if score > best_score:
                                best_score = score
                                best_orientation = orient
                                best_name = orientation_names[i]
                else:
                    # Fallback: just check if shapes match
                    best_orientation = orient
                    best_name = orientation_names[i]
                    print(f"Using {orientation_names[i]} - shapes match")
                    break
        
        if best_score > -1:
            print(f"✅ Best orientation: {best_name} (correlation: {best_score:.3f})")
        else:
            print(f"✅ Using orientation: {best_name}")
            
        return best_orientation, best_name
    else:
        print("✅ Shapes already match, no orientation change needed")
        return uncert_slice, 'original'

# Apply orientation matching
uncert_slice, orientation_applied = find_best_orientation_match(uncert_slice, raw_slice, pred_slice)

# Manual override option - Test different rotations of the uncertainty map
# Uncomment ONE of the following lines to test different orientations:

# OPTION 1: Original (no rotation)
# uncert_slice = uncert_slice.copy()
# print("Currently using: ORIGINAL orientation")

# OPTION 2: 90° clockwise rotation + vertical flip to match prediction location
uncert_slice = np.rot90(uncert_slice, 1)
uncert_slice = np.flipud(uncert_slice)  # Flip vertically to match prediction position
print("Currently using: 90° CLOCKWISE rotation + VERTICAL FLIP")

# OPTION 3: 180° rotation
# uncert_slice = np.rot90(uncert_slice, 2)
# print("Currently using: 180° rotation")

# OPTION 4: 270° clockwise (90° counter-clockwise) rotation
# uncert_slice = np.rot90(uncert_slice, 3)
# print("Currently using: 270° CLOCKWISE (90° counter-clockwise) rotation")

# OPTION 5: Transpose (swap x,y axes)
# uncert_slice = uncert_slice.T
# print("Currently using: TRANSPOSE")

# OPTION 6: Transpose + 90° clockwise
# uncert_slice = np.rot90(uncert_slice.T, 1)
# print("Currently using: TRANSPOSE + 90° clockwise")

# OPTION 7: Transpose + 180°
# uncert_slice = np.rot90(uncert_slice.T, 2)
# print("Currently using: TRANSPOSE + 180°")

# OPTION 8: Transpose + 270° clockwise
# uncert_slice = np.rot90(uncert_slice.T, 3)
# print("Currently using: TRANSPOSE + 270° clockwise")

# OPTION 9: Flip up-down
# uncert_slice = np.flipud(uncert_slice)
# print("Currently using: FLIP UP-DOWN")

# OPTION 10: Flip left-right
# uncert_slice = np.fliplr(uncert_slice)
# print("Currently using: FLIP LEFT-RIGHT")

# OPTION 11: Flip both directions
# uncert_slice = np.flipud(np.fliplr(uncert_slice))
# print("Currently using: FLIP BOTH DIRECTIONS")

print(f"Applied orientation: {orientation_applied}")
print(f"Final uncertainty slice shape: {uncert_slice.shape}")

# VALIDATION: Check if uncertainty and prediction are properly aligned
def validate_uncertainty_prediction_alignment(uncert_slice, pred_slice):
    """
    Validate that uncertainty and prediction are properly aligned
    by checking if high uncertainty correlates with prediction boundaries
    """
    if uncert_slice.shape != pred_slice.shape:
        print(f"❌ Shape mismatch after orientation fix: {uncert_slice.shape} vs {pred_slice.shape}")
        return False
    
    # Check if uncertainty is higher at prediction boundaries
    pred_mask = pred_slice > 0
    if not np.any(pred_mask):
        print("⚠️  No predictions found, cannot validate alignment")
        return True
    
    # Calculate prediction boundaries (edges)
    pred_edges = np.gradient(pred_slice.astype(float))
    pred_edge_magnitude = np.sqrt(pred_edges[0]**2 + pred_edges[1]**2)
    edge_mask = pred_edge_magnitude > np.percentile(pred_edge_magnitude, 75)  # Top 25% of edges
    
    if np.any(edge_mask):
        uncertainty_at_edges = uncert_slice[edge_mask].mean()
        uncertainty_overall = uncert_slice.mean()
        edge_ratio = uncertainty_at_edges / uncertainty_overall if uncertainty_overall > 0 else 1
        
        print(f"Uncertainty at prediction edges: {uncertainty_at_edges:.6f}")
        print(f"Overall uncertainty: {uncertainty_overall:.6f}")
        print(f"Edge-to-overall ratio: {edge_ratio:.2f}")
        
        if edge_ratio > 1.2:  # Uncertainty should be higher at edges
            print("✅ Good alignment: Higher uncertainty at prediction boundaries")
            return True
        else:
            print("⚠️  Possible misalignment: Uncertainty not elevated at boundaries")
            return False
    else:
        print("⚠️  No clear prediction boundaries found")
        return True

alignment_ok = validate_uncertainty_prediction_alignment(uncert_slice, pred_slice)

# DEBUG: Show all possible orientations to help identify the correct one
def show_orientation_debug(uncert_slice_orig, raw_slice):
    """Show all possible orientations to help identify the correct one visually"""
    
    # Generate all possible orientations
    orientations = [
        uncert_slice_orig,  # Original
        uncert_slice_orig.T,  # Transpose
        np.rot90(uncert_slice_orig, 1),  # 90° rotation
        np.rot90(uncert_slice_orig, 2),  # 180° rotation  
        np.rot90(uncert_slice_orig, 3),  # 270° rotation
        np.rot90(uncert_slice_orig.T, 1),  # Transpose + 90°
        np.rot90(uncert_slice_orig.T, 2),  # Transpose + 180°
        np.rot90(uncert_slice_orig.T, 3),  # Transpose + 270°
    ]
    
    orientation_names = ['Original', 'Transpose', '90° Rot', '180° Rot', '270° Rot', 'T+90°', 'T+180°', 'T+270°']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Uncertainty Orientation Options - Find the Best Match', fontsize=14)
    
    for i, (orient, name) in enumerate(zip(orientations, orientation_names)):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Show overlay if shapes match
        if orient.shape == raw_slice.shape:
            ax.imshow(raw_slice, cmap='gray')
            ax.imshow(orient, cmap='hot', alpha=0.5, vmin=0, vmax=np.percentile(orient, 95))
            title_color = 'green'
        else:
            # Just show the uncertainty if shapes don't match
            ax.imshow(orient, cmap='hot', vmin=0, vmax=np.percentile(orient, 95))
            title_color = 'red'
            
        ax.set_title(f'{name}\n{orient.shape}', color=title_color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Green titles = correct shape, Red titles = wrong shape")
    print("Look for the orientation where uncertainty aligns with brain structures")
    return orientations, orientation_names

# Uncomment the next line to see all orientation options:
# orientations, names = show_orientation_debug(uncert_slice if orientation_applied == 'original' else uncertainty[:,:,slice_idx], raw_slice)

print("Final uncertainty slice shape:", uncert_slice.shape)
print("Pred mask shape:", pred_mask.shape)
print("Uncertainty min:", uncert_slice.min(), "max:", uncert_slice.max(), "mean:", uncert_slice.mean())

# Visualize uncertainty alone with better scaling
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(raw_slice, cmap='gray')
plt.title('Raw Image (Reference)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(uncert_slice, cmap='hot', vmin=0, vmax=np.percentile(uncert_slice, 95))
plt.title('Uncertainty Map')
plt.colorbar(label='Uncertainty')
plt.axis('off')

plt.subplot(1, 3, 3)
# Overlay to check alignment
plt.imshow(raw_slice, cmap='gray')
plt.imshow(uncert_slice, cmap='hot', alpha=0.5, vmin=0, vmax=np.percentile(uncert_slice, 95))
plt.title('Overlay Check')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 6))

plt.subplot(1, 6, 1)
plt.imshow(raw_slice, cmap='gray')
plt.title('Raw Image')
plt.axis('off')

plt.subplot(1, 6, 2)
plt.imshow(raw_slice, cmap='gray')
# Show ground truth with proper tumor class colors: green outside (edema), red inside (enhancing/necrotic)
gt_colored = np.zeros((*gt_slice.shape, 3))  # RGB image
gt_colored[gt_slice == 1] = [1, 0, 0]  # Necrotic core -> red
gt_colored[gt_slice == 2] = [0, 1, 0]  # Edema -> green
gt_colored[gt_slice == 3] = [1, 0, 0]  # Enhancing tumor -> red
gt_colored[gt_slice == 4] = [1, 0, 0]  # Additional enhancing -> red (if exists)
plt.imshow(gt_colored, alpha=0.6)
plt.title('FLAIR + Ground Truth Overlay')
plt.axis('off')

plt.subplot(1, 6, 3)
# Show ground truth alone with custom red/green colors
gt_display = np.zeros((*gt_slice.shape, 3))  # RGB image
gt_display[gt_slice == 1] = [1, 0, 0]  # Necrotic core -> red
gt_display[gt_slice == 2] = [0, 1, 0]  # Edema -> green
gt_display[gt_slice == 3] = [1, 0, 0]  # Enhancing tumor -> red
gt_display[gt_slice == 4] = [1, 0, 0]  # Additional enhancing -> red (if exists)
plt.imshow(gt_display)
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(1, 6, 4)
# Show prediction with BraTS standard color scheme
pred_display = np.zeros((*pred_slice.shape, 3))  # RGB image
pred_display[pred_slice == 1] = [1, 0, 0]  # Enhancing core -> red
pred_display[pred_slice == 2] = [0, 1, 0]  # Non-enhancing core -> green
pred_display[pred_slice == 3] = [1, 1, 0]  # Cyst -> yellow
pred_display[pred_slice == 4] = [0, 1, 1]  # Edema -> cyan
plt.imshow(pred_display)
plt.title('Prediction')
plt.axis('off')

plt.subplot(1, 6, 5)
plt.imshow(uncert_slice, cmap='hot', vmin=0, vmax=np.percentile(uncert_slice, 95))
plt.title('Uncertainty Map')
plt.axis('off')

plt.subplot(1, 6, 6)
plt.imshow(raw_slice, cmap='gray')
# Create a masked uncertainty overlay that only shows uncertainty where there are predictions
masked_uncertainty = uncert_slice.copy()
if np.any(pred_mask):
    try:
        from scipy import ndimage
        dilated_pred = ndimage.binary_dilation(pred_mask, iterations=3)
        masked_uncertainty[~dilated_pred] *= 0.3
    except ImportError:
        masked_uncertainty[~pred_mask] *= 0.5
if masked_uncertainty.max() > 0:
    norm_uncert = masked_uncertainty / np.percentile(masked_uncertainty, 95)
    norm_uncert = np.clip(norm_uncert, 0, 1)
else:
    norm_uncert = masked_uncertainty
plt.imshow(norm_uncert, cmap='hot', alpha=0.6, vmin=0, vmax=1)
plt.title('Uncertainty Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()

# Additional analysis: Show uncertainty statistics
print("\n=== Uncertainty Analysis ===")
print(f"Overall uncertainty - Min: {uncert_slice.min():.4f}, Max: {uncert_slice.max():.4f}, Mean: {uncert_slice.mean():.4f}")

if np.any(pred_mask):
    tumor_uncertainty = uncert_slice[pred_mask]
    background_uncertainty = uncert_slice[background_mask]
    print(f"Tumor region uncertainty - Mean: {tumor_uncertainty.mean():.4f}, Std: {tumor_uncertainty.std():.4f}")
    print(f"Background uncertainty - Mean: {background_uncertainty.mean():.4f}, Std: {background_uncertainty.std():.4f}")
    
    # Show histogram of uncertainty values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(uncert_slice.flatten(), bins=50, alpha=0.7, label='All pixels')
    if len(tumor_uncertainty) > 0:
        plt.hist(tumor_uncertainty, bins=30, alpha=0.7, label='Tumor regions')
    if len(background_uncertainty) > 0:
        plt.hist(background_uncertainty, bins=30, alpha=0.7, label='Background')
    plt.xlabel('Uncertainty Value')
    plt.ylabel('Pixel Count')
    plt.title('Uncertainty Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Show where highest uncertainties are located
    high_uncertainty_mask = uncert_slice > np.percentile(uncert_slice, 90)
    plt.imshow(raw_slice, cmap='gray')
    plt.imshow(high_uncertainty_mask, cmap='Reds', alpha=0.5)
    plt.title('Top 10% Highest Uncertainty Regions')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()