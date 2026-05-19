"""
classify_real_lenses.py
========================
Performs real-world inference by loading the pre-trained Physics-Informed LensPINN 
model and classifying the actual SDSS telescope lensing images we downloaded!

This connects your simulated models directly to observational astrophysics!
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models.lens_pinn import LensPINN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  Classifying Real Telescope Images with LensPINN")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # 1. Load LensPINN Checkpoint
    checkpoint_path = Path("results/checkpoints/lens_pinn_model_i_best.pt")
    if not checkpoint_path.exists():
        print(f"[Error] Pre-trained checkpoint not found at {checkpoint_path.resolve()}")
        print("Please ensure you have trained or saved your LensPINN checkpoint.")
        return

    print("-> Loading Physics-Informed LensPINN checkpoint...")
    try:
        model = LensPINN(num_classes=4).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("   [Success] Checkpoint loaded successfully!\n")
    except Exception as e:
        print(f"   [Error] Failed to load checkpoint: {e}")
        return

    # 2. Process real telescope assets
    processed_dir = Path("Data/Real_Hubble_Lenses/processed")
    if not processed_dir.exists():
        # fallback to parent directory check
        processed_dir = Path("../Data/Real_Hubble_Lenses/processed")
        
    if not processed_dir.exists():
        print(f"[Error] Preprocessed real images not found at {processed_dir.resolve()}")
        print("Please run `python my_work/download_real_lenses.py` first.")
        return

    classes = ['no_substructure', 'cdm_subhalos', 'axion', 'vortex']
    real_lenses = ["cosmic_horseshoe", "five_image_quasar", "cheshire_cat"]

    for name in real_lenses:
        npy_path = processed_dir / f"{name}_64.npy"
        if not npy_path.exists():
            print(f"   [Warning] Asset {npy_path.name} not found, skipping.")
            continue
            
        print(f"-> Processing lens: {name.replace('_', ' ').title()}")
        try:
            # Load the normalized numpy array
            arr = np.load(npy_path)
            
            # Format to tensor [Batch, Channel, Height, Width]
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(x)
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[-1]
                else:
                    logits = outputs
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
            pred_idx = np.argmax(probs)
            
            # Display probabilities
            for i, prob in enumerate(probs):
                print(f"   - {classes[i]:<16}: {prob*100:6.2f}%")
            print(f"   => Predicted Substructure: {classes[pred_idx].upper()}\n")
            
        except Exception as e:
            print(f"   [Error] Failed during inference of {name}: {e}")

    print(f"{'='*60}")
    print("[Success] Inference complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
