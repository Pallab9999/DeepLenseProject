# DeepLense — MSc Examination Project
## Physics-Informed ML and Domain Adaptation for Gravitational Lensing

**Pallab Mondal** | Student ID: 946804 | AI Models for Physics 2025/2026  
AI for Science and Technology — University of Milan-Bicocca, University of Milano, University of Pavia

---

## Exam deliverables (quick links)

| Item | File / location |
|------|-----------------|
| Proposal form | `examination_project_proposal_946804.pdf` |
| 2-page report (English) | `deep_lense_project_report.pdf` |
| Presentation (10 slides) | `deep_lense_presentation.pdf` |
| Code | `my_work/` + GitHub below |
| Submission guide | `EXAM_SUBMISSION.md` |

**Repository:** https://github.com/Pallab9999/DeepLenseProject

Rebuild report + slides (no LaTeX required):
```powershell
python scripts/build_exam_deliverables.py
```

---

## Project summary

Classify dark matter substructure in strong gravitational lensing images using:
- **ResNet-18** baseline (CNN)
- **LensPINN** (physics-informed: SIS lens inversion + residual map)
- **HEALSwin** (HEALPix + Swin Transformer)
- **ADDA** domain adaptation (simulated → real SDSS cutouts)

**Phase 1 (Model I):** LensPINN **52.35%** accuracy vs ResNet **43.41%**; CDM recall **28%** vs **2%**.

**Phase 2 (SDSS):** Real lenses classified `no_substructure` at **>99.9%** confidence.

---

## Folder structure

```
DeepLenseProject/
├── DeepLense/              ML4SCI reference repo (read-only)
├── Data/Model_I/           Simulated training data (~37k files)
├── my_work/                Your implementation
│   ├── train.py            Supervised training
│   ├── evaluate.py         Metrics + plots
│   ├── train_adda.py       Domain adaptation
│   ├── classify_real_lenses.py
│   ├── models/             lens_pinn, heal_swin, adda, classifier
│   └── utils/              physics, data_loader, metrics
├── results/
│   ├── checkpoints/        Saved weights
│   └── plots/              Figures for report/slides
├── scripts/build_exam_deliverables.py
└── setup_data.ps1          Dataset download helper
```

---

## Setup

### 1. Dependencies
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install fpdf2
```

### 2. Dataset (Model I)

Google Drive links in the DeepLense README may be unavailable. Options:

**A. Hugging Face (recommended if you have access):**
```powershell
pip install huggingface_hub
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ML4SCI/DeepLense', repo_type='dataset', local_dir='Data')"
```

**B. Script (Drive IDs — may fail if links expired):**
```powershell
.\setup_data.ps1
```

Model I is already present locally if `Data/Model_I` contains ~37k files.

### 3. Train and evaluate
```powershell
cd my_work
python train.py --model lens_pinn --dataset model_i --epochs 100
python evaluate.py --checkpoint ..\results\checkpoints\lens_pinn_model_i_best.pt --model lens_pinn --dataset model_i --confusion
```

### 4. Live demo commands (oral exam)
```powershell
python my_work\train_adda.py --limit_batches 50
python my_work\classify_real_lenses.py
```

---

## Key physics (LensPINN)

```
beta = theta - alpha(theta)           # lens equation
alpha = theta_E * theta / |theta|     # SIS deflection
I_residual = I_orig - I_source        # expose subhalo shears
```

---

## References

| Paper | Link |
|-------|------|
| LensPINN (Ojha et al. 2024) | https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf |
| HEAL-PINN (Srivastava et al. 2025) | https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_252.pdf |
| Domain Adaptation (Alexander et al. 2023) | https://arxiv.org/abs/2112.12121 |
| ADDA (Tzeng et al. 2017) | https://arxiv.org/abs/1702.05464 |
| DeepLense dataset (HF) | https://huggingface.co/datasets/ML4SCI/DeepLense |
