"""
Test script to manually try different rotations and orientations for uncertainty visualization.
This script loads your data and lets you visually compare all possible orientations 
to find the correct one that aligns with the original image.
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import matplotlib.patches as patches

class UncertaintyOrientationTester:
    def __init__(self):
        self.current_orientation_idx = 0
        self.slice_idx = None
        self.raw_slice = None
        self.gt_slice = None
        self.pred_slice = None
        self.uncertainty_original = None
        self.orientations = []
        self.orientation_names = []
        self.fig = None
        self.axes = None
        
    def load_data(self):
        """Load all the image data"""
        print("Loading image data...")
        
        # Load images
        self.raw_img_nii = nib.load('BraTS-PED-00062-0000.nii.gz')
        self.gt_seg_nii = nib.load('BraTS-PED-00062-seg.nii.gz')
        self.pred_seg_nii = nib.load('BraTS-PED-00062.nii.gz')
        self.uncertainty_nii = nib.load('uncertainty.nii.gz')
        
        self.raw_img = self.raw_img_nii.get_fdata()
        self.gt_seg = self.gt_seg_nii.get_fdata()
        self.pred_seg = self.pred_seg_nii.get_fdata()
        self.uncertainty = self.uncertainty_nii.get_fdata()
        
        print(f"Raw image shape: {self.raw_img.shape}")
        print(f"GT segmentation shape: {self.gt_seg.shape}")
        print(f"Prediction shape: {self.pred_seg.shape}")
        print(f"Uncertainty shape: {self.uncertainty.shape}")
        
        # Find a good slice with tumor
        tumor_slices = np.where(np.any(self.gt_seg > 0, axis=(0, 1)))[0]
        if len(tumor_slices) == 0:
            print("No tumor found, using middle slice")
            self.slice_idx = self.raw_img.shape[2] // 2
        else:
            # Find slice with largest tumor area
            tumor_areas = [np.sum(self.gt_seg[:,:,s] > 0) for s in tumor_slices]
            best_slice_idx = np.argmax(tumor_areas)
            self.slice_idx = tumor_slices[best_slice_idx]
            print(f"Using slice {self.slice_idx} with largest tumor area")
        
        # Extract slices
        self.raw_slice = self.raw_img[:, :, self.slice_idx]
        self.gt_slice = self.gt_seg[:, :, self.slice_idx]
        self.pred_slice = self.pred_seg[:, :, self.slice_idx]
        
        # Extract uncertainty slice - handle different dimensions
        if self.uncertainty.ndim == 4:
            if self.uncertainty.shape[1:] == self.pred_seg.shape:
                # Channel first: (C, H, W, D)
                self.uncertainty_original = self.uncertainty[0, :, :, self.slice_idx]
                print("Using 4D uncertainty indexing: [0, :, :, slice_idx]")
            elif self.uncertainty.shape[:-1] == self.pred_seg.shape:
                # Channel last: (H, W, D, C)
                self.uncertainty_original = self.uncertainty[:, :, self.slice_idx, 0]
                print("Using 4D uncertainty indexing: [:, :, slice_idx, 0]")
            else:
                print("Warning: 4D uncertainty shape doesn't match prediction")
                self.uncertainty_original = self.uncertainty[0, :, :, self.slice_idx]
        else:
            # 3D uncertainty
            self.uncertainty_original = self.uncertainty[:, :, self.slice_idx]
            print("Using 3D uncertainty indexing: [:, :, slice_idx]")
        
        print(f"Extracted slice shapes:")
        print(f"  Raw: {self.raw_slice.shape}")
        print(f"  GT: {self.gt_slice.shape}")
        print(f"  Prediction: {self.pred_slice.shape}")
        print(f"  Uncertainty: {self.uncertainty_original.shape}")
        
    def generate_orientations(self):
        """Generate all possible orientations of the uncertainty map"""
        base = self.uncertainty_original
        
        self.orientations = [
            base,                           # 0: Original
            base.T,                         # 1: Transpose
            np.rot90(base, 1),             # 2: 90° rotation
            np.rot90(base, 2),             # 3: 180° rotation
            np.rot90(base, 3),             # 4: 270° rotation
            np.rot90(base.T, 1),           # 5: Transpose + 90°
            np.rot90(base.T, 2),           # 6: Transpose + 180°
            np.rot90(base.T, 3),           # 7: Transpose + 270°
            np.flipud(base),               # 8: Flip up-down
            np.fliplr(base),               # 9: Flip left-right
            np.flipud(np.fliplr(base)),    # 10: Flip both
            np.rot90(np.flipud(base), 1),  # 11: Flip up-down + 90°
            np.rot90(np.fliplr(base), 1),  # 12: Flip left-right + 90°
            np.rot90(np.flipud(base), 2),  # 13: Flip up-down + 180°
            np.rot90(np.fliplr(base), 2),  # 14: Flip left-right + 180°
            np.rot90(np.flipud(base), 3),  # 15: Flip up-down + 270°
        ]
        
        self.orientation_names = [
            "Original",
            "Transpose",
            "Rotate 90°",
            "Rotate 180°",
            "Rotate 270°",
            "Transpose + 90°",
            "Transpose + 180°",
            "Transpose + 270°",
            "Flip up-down",
            "Flip left-right", 
            "Flip both",
            "Flip up-down + 90°",
            "Flip left-right + 90°",
            "Flip up-down + 180°",
            "Flip left-right + 180°",
            "Flip up-down + 270°",
        ]
        
        print(f"Generated {len(self.orientations)} different orientations to test")
        
    def calculate_alignment_score(self, uncert_slice):
        """Calculate how well uncertainty aligns with prediction boundaries"""
        if uncert_slice.shape != self.pred_slice.shape:
            return -1, "Shape mismatch"
        
        # Calculate prediction boundaries
        pred_edges = np.gradient(self.pred_slice.astype(float))
        pred_edge_magnitude = np.sqrt(pred_edges[0]**2 + pred_edges[1]**2)
        
        # Normalize for correlation
        if uncert_slice.std() == 0 or pred_edge_magnitude.std() == 0:
            return 0, "No variation in data"
        
        uncert_norm = (uncert_slice - uncert_slice.mean()) / uncert_slice.std()
        edges_norm = (pred_edge_magnitude - pred_edge_magnitude.mean()) / pred_edge_magnitude.std()
        
        # Calculate correlation
        correlation = np.corrcoef(uncert_norm.flatten(), edges_norm.flatten())[0, 1]
        
        if np.isnan(correlation):
            return 0, "NaN correlation"
        
        return correlation, "OK"
    
    def show_all_orientations_grid(self):
        """Show all orientations in a grid for comparison"""
        n_orientations = len(self.orientations)
        n_cols = 4
        n_rows = (n_orientations + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        fig.suptitle('All Uncertainty Orientations - Find the Best Match with Brain Structure', fontsize=16)
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in (row if isinstance(row, np.ndarray) else [row])]
        
        scores = []
        for i, (orient, name) in enumerate(zip(self.orientations, self.orientation_names)):
            ax = axes_flat[i]
            
            # Calculate alignment score
            score, status = self.calculate_alignment_score(orient)
            scores.append(score)
            
            # Show overlay if shapes match
            if orient.shape == self.raw_slice.shape:
                ax.imshow(self.raw_slice, cmap='gray')
                ax.imshow(orient, cmap='hot', alpha=0.6, vmin=0, vmax=np.percentile(orient, 95))
                title_color = 'green' if score > 0.1 else 'orange'
                title = f'{i}: {name}\nScore: {score:.3f}'
            else:
                ax.imshow(orient, cmap='hot', vmin=0, vmax=np.percentile(orient, 95))
                title_color = 'red'
                title = f'{i}: {name}\nShape: {orient.shape}'
            
            ax.set_title(title, color=title_color, fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_orientations, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Print ranked results
        print("\n=== Orientation Ranking by Alignment Score ===")
        scored_orientations = [(i, score, name) for i, (score, name) in enumerate(zip(scores, self.orientation_names)) if score >= 0]
        scored_orientations.sort(key=lambda x: x[1], reverse=True)
        
        print("Top orientations (higher score = better alignment with prediction boundaries):")
        for i, (idx, score, name) in enumerate(scored_orientations[:8]):
            print(f"{i+1:2d}. Index {idx:2d}: {name:20s} (score: {score:.3f})")
        
        return scored_orientations
    
    def show_detailed_comparison(self, orientation_indices=None):
        """Show detailed comparison of specific orientations"""
        if orientation_indices is None:
            # Show top 4 by default
            scored = self.show_all_orientations_grid()
            orientation_indices = [x[0] for x in scored[:4]]
        
        fig, axes = plt.subplots(len(orientation_indices), 4, figsize=(16, 4*len(orientation_indices)))
        if len(orientation_indices) == 1:
            axes = [axes]
        
        for row, idx in enumerate(orientation_indices):
            orient = self.orientations[idx]
            name = self.orientation_names[idx]
            score, status = self.calculate_alignment_score(orient)
            
            # Raw image
            axes[row][0].imshow(self.raw_slice, cmap='gray')
            axes[row][0].set_title('Raw Image' if row == 0 else '')
            axes[row][0].axis('off')
            
            # Uncertainty alone
            if orient.shape == self.raw_slice.shape:
                axes[row][1].imshow(orient, cmap='hot', vmin=0, vmax=np.percentile(orient, 95))
                axes[row][1].set_title(f'Uncertainty\n{name}' if row == 0 else f'{name}')
            else:
                axes[row][1].imshow(orient, cmap='hot', vmin=0, vmax=np.percentile(orient, 95))
                axes[row][1].set_title(f'Wrong Shape\n{name}' if row == 0 else f'{name}')
            axes[row][1].axis('off')
            
            # Overlay
            if orient.shape == self.raw_slice.shape:
                axes[row][2].imshow(self.raw_slice, cmap='gray')
                axes[row][2].imshow(orient, cmap='hot', alpha=0.5, vmin=0, vmax=np.percentile(orient, 95))
                axes[row][2].set_title(f'Overlay\nScore: {score:.3f}' if row == 0 else f'Score: {score:.3f}')
            else:
                axes[row][2].text(0.5, 0.5, 'Shape\nMismatch', ha='center', va='center', transform=axes[row][2].transAxes)
                axes[row][2].set_title('Overlay' if row == 0 else '')
            axes[row][2].axis('off')
            
            # Prediction for reference
            axes[row][3].imshow(self.raw_slice, cmap='gray')
            axes[row][3].imshow(self.pred_slice > 0, cmap='Reds', alpha=0.5)
            axes[row][3].set_title('Prediction Reference' if row == 0 else '')
            axes[row][3].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def interactive_viewer(self):
        """Create an interactive viewer to cycle through orientations"""
        self.current_orientation_idx = 0
        
        self.fig, self.axes = plt.subplots(1, 4, figsize=(16, 4))
        self.fig.suptitle('Interactive Uncertainty Orientation Tester', fontsize=14)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.21, 0.02, 0.1, 0.04])
        ax_best = plt.axes([0.32, 0.02, 0.15, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_best = Button(ax_best, 'Show Best')
        
        self.btn_prev.on_clicked(self.prev_orientation)
        self.btn_next.on_clicked(self.next_orientation)
        self.btn_best.on_clicked(self.show_best_orientation)
        
        self.update_display()
        plt.show()
    
    def prev_orientation(self, event):
        """Go to previous orientation"""
        self.current_orientation_idx = (self.current_orientation_idx - 1) % len(self.orientations)
        self.update_display()
    
    def next_orientation(self, event):
        """Go to next orientation"""
        self.current_orientation_idx = (self.current_orientation_idx + 1) % len(self.orientations)
        self.update_display()
    
    def show_best_orientation(self, event):
        """Jump to the best scoring orientation"""
        scores = [self.calculate_alignment_score(orient)[0] for orient in self.orientations]
        valid_scores = [(i, score) for i, score in enumerate(scores) if score >= 0]
        if valid_scores:
            best_idx = max(valid_scores, key=lambda x: x[1])[0]
            self.current_orientation_idx = best_idx
            self.update_display()
    
    def update_display(self):
        """Update the interactive display"""
        if self.fig is None:
            return
        
        orient = self.orientations[self.current_orientation_idx]
        name = self.orientation_names[self.current_orientation_idx]
        score, status = self.calculate_alignment_score(orient)
        
        # Clear all axes
        for ax in self.axes:
            ax.clear()
        
        # Raw image
        self.axes[0].imshow(self.raw_slice, cmap='gray')
        self.axes[0].set_title('Raw Image')
        self.axes[0].axis('off')
        
        # Uncertainty
        if orient.shape == self.raw_slice.shape:
            self.axes[1].imshow(orient, cmap='hot', vmin=0, vmax=np.percentile(orient, 95))
            self.axes[1].set_title(f'Uncertainty\n{name}')
        else:
            self.axes[1].imshow(orient, cmap='hot', vmin=0, vmax=np.percentile(orient, 95))
            self.axes[1].set_title(f'Wrong Shape\n{name}')
        self.axes[1].axis('off')
        
        # Overlay
        if orient.shape == self.raw_slice.shape:
            self.axes[2].imshow(self.raw_slice, cmap='gray')
            self.axes[2].imshow(orient, cmap='hot', alpha=0.5, vmin=0, vmax=np.percentile(orient, 95))
            self.axes[2].set_title(f'Overlay\nScore: {score:.3f}')
            
            # Add border color based on score
            if score > 0.2:
                border_color = 'green'
            elif score > 0.0:
                border_color = 'orange'
            else:
                border_color = 'red'
            
            for spine in self.axes[2].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
        else:
            self.axes[2].text(0.5, 0.5, 'Shape\nMismatch', ha='center', va='center', 
                            transform=self.axes[2].transAxes, fontsize=14, color='red')
            self.axes[2].set_title('Overlay')
        self.axes[2].axis('off')
        
        # Prediction reference
        self.axes[3].imshow(self.raw_slice, cmap='gray')
        self.axes[3].imshow(self.pred_slice > 0, cmap='Reds', alpha=0.5)
        self.axes[3].set_title('Prediction Reference')
        self.axes[3].axis('off')
        
        # Update main title
        self.fig.suptitle(f'Orientation {self.current_orientation_idx}/{len(self.orientations)-1}: {name} (Score: {score:.3f})', 
                         fontsize=14)
        
        self.fig.canvas.draw()
        
        # Print info
        print(f"\nCurrent: {self.current_orientation_idx} - {name}")
        print(f"  Shape: {orient.shape} (target: {self.raw_slice.shape})")
        print(f"  Alignment score: {score:.3f} ({status})")
        if orient.shape == self.raw_slice.shape:
            print(f"  Uncertainty range: {orient.min():.4f} to {orient.max():.4f}")
        
    def get_transformation_code(self, orientation_idx):
        """Get the Python code to apply the selected transformation"""
        base_var = "uncertainty_slice"
        
        transformations = [
            f"{base_var}",                                    # 0: Original
            f"{base_var}.T",                                  # 1: Transpose
            f"np.rot90({base_var}, 1)",                      # 2: 90° rotation
            f"np.rot90({base_var}, 2)",                      # 3: 180° rotation
            f"np.rot90({base_var}, 3)",                      # 4: 270° rotation
            f"np.rot90({base_var}.T, 1)",                    # 5: Transpose + 90°
            f"np.rot90({base_var}.T, 2)",                    # 6: Transpose + 180°
            f"np.rot90({base_var}.T, 3)",                    # 7: Transpose + 270°
            f"np.flipud({base_var})",                        # 8: Flip up-down
            f"np.fliplr({base_var})",                        # 9: Flip left-right
            f"np.flipud(np.fliplr({base_var}))",            # 10: Flip both
            f"np.rot90(np.flipud({base_var}), 1)",          # 11: Flip up-down + 90°
            f"np.rot90(np.fliplr({base_var}), 1)",          # 12: Flip left-right + 90°
            f"np.rot90(np.flipud({base_var}), 2)",          # 13: Flip up-down + 180°
            f"np.rot90(np.fliplr({base_var}), 2)",          # 14: Flip left-right + 180°
            f"np.rot90(np.flipud({base_var}), 3)",          # 15: Flip up-down + 270°
        ]
        
        return transformations[orientation_idx]
    
    def print_summary(self):
        """Print a summary of all orientations with their scores and code"""
        print("\n" + "="*80)
        print("UNCERTAINTY ORIENTATION TEST SUMMARY")
        print("="*80)
        
        scores = []
        for i, (orient, name) in enumerate(zip(self.orientations, self.orientation_names)):
            score, status = self.calculate_alignment_score(orient)
            shape_match = orient.shape == self.raw_slice.shape
            scores.append((i, score, name, shape_match, status))
        
        # Sort by score (valid shapes first, then by score)
        scores.sort(key=lambda x: (x[3], x[1]), reverse=True)
        
        print(f"\nTarget shape: {self.raw_slice.shape}")
        print(f"Original uncertainty shape: {self.uncertainty_original.shape}")
        print("\nRanked orientations:")
        print("-" * 80)
        
        for rank, (idx, score, name, shape_match, status) in enumerate(scores):
            shape_str = "✓" if shape_match else "✗"
            score_str = f"{score:6.3f}" if score >= 0 else "  N/A "
            
            print(f"{rank+1:2d}. [{idx:2d}] {shape_str} {score_str} | {name:20s} | {self.orientations[idx].shape}")
            
            if rank < 5:  # Show code for top 5
                code = self.get_transformation_code(idx)
                print(f"     Code: {code}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        # Best valid orientation
        valid_orientations = [x for x in scores if x[3]]  # shape_match == True
        if valid_orientations:
            best = valid_orientations[0]
            best_idx, best_score, best_name, _, _ = best
            print(f"🎯 BEST MATCH: Index {best_idx} - {best_name}")
            print(f"   Alignment score: {best_score:.3f}")
            print(f"   Code to use: {self.get_transformation_code(best_idx)}")
            
            if best_score > 0.2:
                print("   ✅ Strong alignment - this is likely the correct orientation!")
            elif best_score > 0.0:
                print("   ⚠️  Moderate alignment - check visually to confirm")
            else:
                print("   ❌ Poor alignment - may need manual adjustment")
        else:
            print("❌ No orientations match the target shape!")
            print("   This suggests a more complex transformation is needed.")
        
        print("\n💡 USAGE:")
        print("   1. Look at the grid view to visually identify the best match")
        print("   2. Use the interactive viewer to compare top candidates")
        print("   3. Apply the transformation code in your visualization script")
        print("   4. Verify that uncertainty aligns with brain structures and prediction boundaries")


def main():
    """Main function to run the orientation tester"""
    print("🧠 Uncertainty Orientation Tester")
    print("=" * 50)
    
    tester = UncertaintyOrientationTester()
    
    try:
        # Load data
        tester.load_data()
        
        # Generate all orientations
        tester.generate_orientations()
        
        # Show grid overview
        print("\n🔍 Showing all orientations in grid view...")
        ranked_orientations = tester.show_all_orientations_grid()
        
        # Show detailed comparison of top candidates
        print("\n📊 Showing detailed comparison of top candidates...")
        top_indices = [x[0] for x in ranked_orientations[:4] if x[1] >= 0]
        if top_indices:
            tester.show_detailed_comparison(top_indices)
        
        # Print summary
        tester.print_summary()
        
        # Interactive viewer
        print("\n🎮 Launching interactive viewer...")
        print("   Use 'Previous' and 'Next' buttons to navigate")
        print("   Use 'Show Best' to jump to highest scoring orientation")
        tester.interactive_viewer()
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e}")
        print("   Make sure you're in the directory with your image files:")
        print("   - BraTS-PED-00062-0000.nii.gz")
        print("   - BraTS-PED-00062-seg.nii.gz") 
        print("   - BraTS-PED-00062.nii.gz")
        print("   - uncertainty.nii.gz")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
