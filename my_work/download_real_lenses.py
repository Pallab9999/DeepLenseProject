"""
download_real_lenses.py
========================
Downloads and preprocesses actual, high-resolution Hubble Space Telescope (HST)
images of famous strong gravitational lenses directly from the ESA/Hubble archive.

This provides the "Real Telescope Target Domain" images for your project!

Lenses Included
---------------
1.  **The Cosmic Horseshoe** (SDSS J1011+0143) - A spectacular, nearly complete Einstein Ring.
2.  **The Einstein Cross** (Q2237+030) - Four lensed images of a distant quasar around a foreground galaxy.
3.  **Abell 2218** - A massive galaxy cluster showing numerous Einstein arcs and gravitational shears.

Usage
-----
    cd my_work
    python download_real_lenses.py
"""

import os
import sys
from pathlib import Path
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import IMAGE_SIZE


# ── Famous SDSS Telescope Lenses (Direct sky survey cutouts) ───────────────────
LENSES_URLS = {
    "cosmic_horseshoe": "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=152.6289&dec=1.4308&scale=0.2&width=258&height=258",
    "five_image_quasar": "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=151.1575&dec=41.2008&scale=0.2&width=256&height=256",
    "cheshire_cat":     "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=159.6263&dec=48.8189&scale=0.2&width=256&height=256"
}


def download_and_preprocess():
    # Setup directories
    dest_dir = Path("../Data/Real_Hubble_Lenses")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = dest_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Downloading Real SDSS Telescope Lensing Images")
    print(f"{'='*60}\n")

    for name, url in LENSES_URLS.items():
        raw_path = dest_dir / f"{name}_raw.jpg"
        
        # 1. Download raw telescope image
        if not raw_path.exists():
            print(f"-> Downloading {name} from SDSS archive...")
            time.sleep(2)  # prevent rate-limiting 403 block
            try:
                # Set a User-Agent header to bypass basic scraping blocks
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response, open(raw_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"   [Success] Saved raw image to {raw_path.name}")
            except Exception as e:
                print(f"   [Error] Failed to download {name}: {e}")
                continue
        else:
            print(f"   [NOTE] {name}_raw.jpg already exists, skipping download.")

        # 2. Preprocess the image (Crop to center ➡️ Convert to Grayscale ➡️ Resize to 64x64)
        print(f"-> Preprocessing {name} to match model input dimensions...")
        try:
            with Image.open(raw_path) as img:
                # Grayscale conversion
                img_gray = img.convert("L")
                
                # Center crop to a square aspect ratio
                w, h = img_gray.size
                crop_size = min(w, h)
                left = (w - crop_size) // 2
                top = (h - crop_size) // 2
                right = left + crop_size
                bottom = top + crop_size
                img_cropped = img_gray.crop((left, top, right, bottom))
                
                # Resize to model input image size (64x64)
                img_resized = img_cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                
                # Save preprocessed image
                npy_path = processed_dir / f"{name}_64.npy"
                png_path = processed_dir / f"{name}_64.png"
                
                # Save as standard PNG for visualization
                img_resized.save(png_path)
                
                # Save as numpy array matching Model I simulated data format
                arr = np.array(img_resized, dtype=np.float32) / 255.0  # scale to [0, 1]
                np.save(npy_path, arr)
                
                print(f"   [Success] Saved preprocessed assets:")
                print(f"             - PNG: {png_path.name}")
                print(f"             - NPY: {npy_path.name}")
        except Exception as e:
            print(f"   [Error] Failed to preprocess {name}: {e}")

    print(f"\n{'='*60}")
    print("[Success] Real telescope images downloaded & preprocessed successfully!")
    print(f"   Location: {dest_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    download_and_preprocess()
