# Quick script to check if your model has dropout layers
import torch
from nnunetv2.inference.predict_with_mc_dropout_ver4 import nnUNetPredictor

# Initialize predictor (adjust path as needed)
predictor = nnUNetPredictor()
predictor.initialize_from_trained_model_folder(
    'path/to/your/model',  # Update this
    use_folds=(0,),
    checkpoint_name='checkpoint_final.pth',
)

# Check for dropout layers
dropout_layers = []
for name, module in predictor.network.named_modules():
    if 'dropout' in module.__class__.__name__.lower():
        dropout_layers.append((name, module))

if dropout_layers:
    print(f"Found {len(dropout_layers)} dropout layers:")
    for name, module in dropout_layers:
        print(f"  {name}: {module}")
else:
    print("❌ No dropout layers found! MC dropout won't work.")
    print("Your model architecture may not include dropout layers.")
