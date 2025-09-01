#!/usr/bin/env python3
"""
Wrapper script to make MC dropout prediction work like nnUNetv2_predict
Save this as nnUNet_mc_predict.py and add to your PATH
"""

import sys
import os
from pathlib import Path

# Add nnUNet directory to path
nnunet_dir = Path(__file__).parent
sys.path.insert(0, str(nnunet_dir))

# Import and run the MC dropout prediction
from nnunetv2.inference.predict_with_mc_dropout_ver4 import predict_entry_point_modelfolder

if __name__ == "__main__":
    predict_entry_point_modelfolder()
