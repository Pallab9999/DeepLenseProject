# Exam submission checklist — AI Models for Physics 2025/2026

**Student:** Pallab Mondal | **ID:** 946804

Use this checklist before the oral examination. Deadlines are relative to your exam date.

---

## Requirement 1 — Project proposal (≥3 weeks before exam)

- [ ] Fill and send [examination_project_proposal_946804.pdf](examination_project_proposal_946804.pdf) via course/Moodle process
- [ ] Wait for **acceptance confirmation** (proposal must be accepted to be valid)
- [ ] Keep a copy of the acceptance email

**Status in repo:** Draft PDF and `.tex` source ready.

---

## Requirement 2 — Written summary (3 days before oral)

- [ ] PDF is **exactly 2 pages** (pages 3+ are ignored)
- [ ] English, with figures and results
- [ ] Attach: `deep_lense_project_report.pdf`
- [ ] Email to:
  - enrico.prati@unimi.it
  - paolo.zentilini@unimi.it

**Rebuild report:**
```powershell
python scripts/build_exam_deliverables.py
```
Open the PDF and confirm page count = 2.

**Email template:** see [EXAM_EMAIL_TEMPLATE.txt](EXAM_EMAIL_TEMPLATE.txt)

---

## Requirement 3 — Code (3 days before oral)

- [ ] Push latest code to GitHub: https://github.com/Pallab9999/DeepLenseProject
- [ ] README explains how to run training/evaluation
- [ ] Examiners can access the repo (public or invited)
- [ ] Include link in the same email as the report (or a separate email if preferred)

**What to highlight:**
- `my_work/` — all project code
- `results/plots/` — figures
- `results/checkpoints/` — trained weights (or note if too large for git)

---

## Requirement 4 — Oral part 1 (~10 min, max 10 slides + code discussion)

- [ ] Presentation PDF: `deep_lense_presentation.pdf` (10 slides)
- [ ] Rehearse to fit **~10 minutes**
- [ ] Be ready to open and explain:
  - `my_work/utils/physics.py`
  - `my_work/models/lens_pinn.py`
  - `my_work/train.py`, `evaluate.py`
  - `my_work/train_adda.py`, `classify_real_lenses.py`

**Rebuild slides:**
```powershell
python scripts/build_exam_deliverables.py
```

**Live demo (optional):**
```powershell
python my_work\train_adda.py --limit_batches 50
python my_work\classify_real_lenses.py
```

---

## Requirement 5 — Oral part 2 (whiteboard theory)

Prepare 1–2 topics from the course list. **Highest priority for this project:**

1. **Physics-informed neural networks** — lens equation, residual loss, why physics helps
2. **Convolutional neural networks** — conv layers, pooling, why ResNet baseline fails on CDM

Also review: Metropolis, Gibbs/Ising, RBM, RL/Bellman, random walks, GNN message passing, QML.

Study guide: [msc_exam_study_guide.md](msc_exam_study_guide.md)

---

## Pre-exam timeline (example)

| When | Action |
|------|--------|
| T − 3 weeks | Send proposal; get acceptance |
| T − 1 week | Rehearse slides; test live demos |
| T − 3 days | Email 2-page report + GitHub link |
| T − 1 day | Print/open report; verify repo; charge laptop |

---

## Current project status

| Component | Status |
|-----------|--------|
| Proposal PDF | Ready |
| 2-page report PDF | Ready (rebuild with script) |
| 10-slide presentation | Ready |
| Code (`my_work/`) | Complete |
| Model I data | Present locally |
| Model II/III | Not required if report focuses on Model I |
| GitHub repo | https://github.com/Pallab9999/DeepLenseProject |

---

## Key results to state in the oral

- LensPINN: **52.35%** accuracy, F1 **0.51** vs ResNet **43.41%**, F1 **0.36**
- CDM recall: **28%** (LensPINN) vs **2%** (ResNet)
- SDSS lenses: **>99.9%** `no_substructure` (scientifically correct at survey resolution)
- HEALSwin: 5-epoch mode collapse — physics models need longer training to learn θ_E first
