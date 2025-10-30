#!/bin/bash
# Wrapper script to run visualization with proper conda environment

# Activate conda
source ~/.bashrc
conda activate nnunetv2

# Run the visualization
python create_proper_uncertainty_overlay.py \
    -i /scratch/hpl14/mc_dropout_test_3cases/input \
    -u /scratch/hpl14/mc_dropout_test_3cases/output \
    -o /scratch/hpl14/mc_dropout_test_3cases/beautiful_visualizations

echo "Visualization complete!"
ls -lh /scratch/hpl14/mc_dropout_test_3cases/beautiful_visualizations/
