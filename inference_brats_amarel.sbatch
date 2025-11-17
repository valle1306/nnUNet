#!/bin/bash
#SBATCH --job-name=nnunet_inference_mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=inference_%j.log
#SBATCH --error=inference_%j.err

# Load modules (adjust based on Amarel's module system)
module load cuda/11.8
module load python/3.9

# Activate conda environment
source ~/.bashrc
conda activate nnunetv2

# Set nnUNet environment variables
export nnUNet_raw="/scratch/hpl14/nnunet_v2_valerie/raw"
export nnUNet_preprocessed="/scratch/hpl14/nnunet_v2_valerie/preprocessed"
export nnUNet_results="/scratch/hpl14/nnunet_v2_valerie/results"

# Navigate to your nnUNet directory
cd ~/nnUNet

# Run inference with MC Dropout
python nnunetv2/inference/predict_with_mc_dropout_edited.py \
    -i /scratch/hpl14/nnunet_v2_valerie/raw/Dataset777_BraTSPED2024/imagesTs \
    -o /scratch/hpl14/nnunet_v2_valerie/inference_output_mc_dropout \
    -m /scratch/hpl14/nnunet_v2_valerie/results/Dataset777_BraTSPED2024/nnUNetTrainer__nnUNetPlans__3d_fullres \
    -f 0 1 2 3 4 \
    --verbose \
    --disable_progress_bar

echo "Inference completed!"
