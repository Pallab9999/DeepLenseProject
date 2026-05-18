# DeepLense GSoC 2026
## Physics-Guided ML on Real Gravitational Lensing Images
**Pallab Mondal** — MSc AI for Science and Technology, University of Milan-Bicocca

---

## Project Goal
Build the **first physics-informed model that works on real HSC telescope images** by combining:
- LensPINN physics encoder (ViT → Einstein radius → lensing inversion)
- ADDA unsupervised domain adaptation
- 4-class dark matter classification (no substructure, CDM subhalos, axion, **vortex**)

---

## Folder Structure
```
DeepLenseProject/
├── DeepLense/           ← ML4SCI repo (READ ONLY — reference code)
├── Data/
│   ├── Model_I/         ← simulated, 3 classes, 30k images each
│   ├── Model_II/        ← different redshift range
│   ├── Model_III/       ← different SNR
│   └── Model_IV/        ← real HSC hybrid (download separately)
├── my_work/             ← YOUR code
│   ├── config.py        ← ALL hyperparameters in one place
│   ├── train.py         ← Phase 1: supervised training
│   ├── evaluate.py      ← evaluation + GradCAM
│   ├── models/
│   │   ├── lens_pinn.py    ← LensPINN (ViT + physics + CNN, 4-class)
│   │   ├── heal_swin.py    ← HEALSwin (Swin + HEALPix + physics)
│   │   ├── adda.py         ← ADDA domain adaptation
│   │   └── classifier.py  ← generic backbone + head, ResNet baseline
│   ├── utils/
│   │   ├── physics.py      ← differentiable lensing equation (SIS)
│   │   ├── data_loader.py  ← dataset, augmentation, dataloaders
│   │   ├── preprocessing.py ← distortion map, normalisation
│   │   └── metrics.py      ← AUC, F1, GradCAM, confusion matrix
│   └── notebooks/
│       ├── 01_data_exploration.ipynb
│       ├── 03_4class_extension.ipynb
│       ├── 04_domain_adaptation.ipynb
│       └── 05_evaluation.ipynb
├── results/
│   ├── checkpoints/        ← saved model weights
│   ├── plots/              ← training curves, GradCAM, confusion matrices
│   └── logs/               ← training logs
├── requirements.txt
└── setup_data.ps1          ← downloads Model I/II/III
```

---

## Quick Start

### 1. Install dependencies
```powershell
# Install PyTorch with CUDA 11.8 (adjust if you have a different GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Download datasets
```powershell
.\setup_data.ps1
```

### 3. Train LensPINN
```powershell
cd my_work
python train.py --model lens_pinn --dataset model_i --epochs 100
```

### 4. Evaluate
```powershell
python evaluate.py --checkpoint ..\results\checkpoints\lens_pinn_model_i_best.pt \
                   --model lens_pinn --dataset model_i --confusion --gradcam
```

### 5. Open notebooks
Open Jupyter or VS Code and run the notebooks in order:
1. `01_data_exploration.ipynb` — understand the data
2. `03_4class_extension.ipynb` — train LensPINN, run GradCAM
3. `04_domain_adaptation.ipynb` — ADDA demo
4. `05_evaluation.ipynb` — ablation study table

---

## Model Training Commands

| Model | Command |
|-------|---------|
| ResNet-18 baseline | `python train.py --model resnet_baseline --dataset model_i` |
| ViT (no physics) | `python train.py --model vit_baseline --dataset model_i` |
| LensPINN | `python train.py --model lens_pinn --dataset model_i --lambda_phys 0.1` |
| HEALSwin | `python train.py --model heal_swin --dataset model_i` |

---

## Three Contributions

| # | Contribution | Status |
|---|-------------|--------|
| 1 | LensPINN → 4-class + GradCAM | ✅ Implemented in `models/lens_pinn.py` |
| 2 | HEALSwin + physics encoder | ✅ Implemented in `models/heal_swin.py` |
| 3 | ADDA domain adaptation | ✅ Implemented in `models/adda.py` |

---

## Key Physics

The SIS gravitational lensing equation (embedded into every model):

```
beta = theta - alpha(theta)          (lensing equation)
alpha = theta_E * theta / |theta|    (SIS deflection angle)
```

Where `theta_E` is the Einstein radius — predicted by the ViT encoder and used
to reconstruct the source-plane image via differentiable bilinear interpolation.

---

## References
| Paper | Link |
|-------|------|
| LensPINN (Ojha et al. 2024) | https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf |
| HEAL-PINN (Srivastava et al. 2025) | https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_252.pdf |
| Domain Adaptation (Alexander et al. 2023) | https://arxiv.org/abs/2112.12121 |
| ADDA (Tzeng et al. 2017) | https://arxiv.org/abs/1702.05464 |
