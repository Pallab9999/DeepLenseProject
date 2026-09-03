"""
Build exam deliverables without LaTeX:
  - deep_lense_project_report.pdf  (exactly 2 pages, with figures)
  - deep_lense_presentation.pdf    (10 slides)

Usage (from project root):
    python scripts/build_exam_deliverables.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "results" / "plots"
REPORT_OUT = ROOT / "deep_lense_project_report.pdf"
SLIDES_OUT = ROOT / "deep_lense_presentation.pdf"

NAVY = (10, 25, 47)
BODY = (40, 40, 40)
MUTED = (100, 100, 100)


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"Page {self.page_no()}/2  |  Pallab Mondal 946804  |  AI Models for Physics 2025/2026", align="C")


def _body_text(pdf: FPDF, text: str, size: float = 9.0, lh: float = 4.2) -> None:
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(*BODY)
    pdf.multi_cell(0, lh, text)
    pdf.ln(1)


def build_report() -> None:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(18, 16, 18)

    # ── Page 1 ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 6, "Physics-Informed ML and Domain Adaptation for\nDark Matter Classification in Gravitational Lenses")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5, "Pallab Mondal  |  Student ID: 946804  |  AI Models for Physics 2025/2026  |  May 2026")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 5, "Summary")
    pdf.ln(4)
    _body_text(
        pdf,
        "Strong gravitational lensing probes dark matter substructure via arc distortions. "
        "We compare ResNet-18, LensPINN (physics-informed CNN with differentiable SIS lens inversion), "
        "and HEALSwin on DeepLense Model I (30,000 train / 7,500 test; 64x64; classes: no_substructure, "
        "cdm_subhalos, axion). Phase 2 applies ADDA domain adaptation and classifies real SDSS cutouts. "
        "Code: https://github.com/Pallab9999/DeepLenseProject",
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 5, "Method")
    pdf.ln(4)
    _body_text(
        pdf,
        "ResNet-18 baseline: spatial convolutions on raw pixels. LensPINN: predict Einstein radius theta_E, "
        "invert beta = theta - alpha(theta) with SIS alpha(theta) = theta_E * theta/|theta|, reconstruct source "
        "I_source, form residual I_residual = I_orig - I_source; decoder uses original + source + residual. "
        "HEALSwin: HEALPix + Swin attention + physics head. ADDA: freeze source encoder M_s, train target "
        "encoder M_t adversarially vs discriminator D to align simulated and telescope features.",
        size=8.5,
        lh=4.0,
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 5, "Results (Phase 1: Model I)")
    pdf.ln(3)

    col_w = [42, 38, 38, 38]
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(240, 240, 240)
    for i, h in enumerate(["Metric", "ResNet-18", "LensPINN", "HEALSwin"]):
        pdf.cell(col_w[i], 6, h, border=1, fill=True, align="C" if i else "L")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    rows = [
        ("Accuracy", "43.41%", "52.35%", "33.33%"),
        ("Macro F1", "0.355", "0.511", "0.167"),
        ("CDM recall", "2%", "28%", "0%"),
        ("Smooth / Axion AUC", "0.64 / 0.68", "0.79 / 0.76", "N/A"),
    ]
    for row in rows:
        for i, val in enumerate(row):
            pdf.cell(col_w[i], 6, val, border=1, align="C" if i else "L")
        pdf.ln()

    pdf.ln(2)
    _body_text(
        pdf,
        "LensPINN improves accuracy by +9.0% absolute by subtracting the smooth Einstein ring and exposing "
        "subhalo shears. ResNet collapses on CDM (2% recall). HEALSwin at 5 epochs shows mode collapse (33.3%): "
        "physics-informed models must learn theta_E before classification converges.",
        size=8.5,
        lh=4.0,
    )

    y = pdf.get_y()
    if y < 175:
        pdf.set_y(y)
    w = 82
    x1, x2 = 18, 108
    h_img = 42
    for path, x, cap in [
        (PLOTS / "lens_pinn_model_i_cm.png", x1, "Fig 1: LensPINN confusion matrix"),
        (PLOTS / "lens_pinn_model_i_curves.png", x2, "Fig 2: Training curves"),
    ]:
        if path.exists():
            pdf.image(str(path), x=x, y=pdf.get_y(), w=w, h=h_img)
    pdf.ln(h_img + 2)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 4, "Fig 1: LensPINN confusion matrix          Fig 2: Training curves")

    # ── Page 2 ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 5, "Results (Phase 2: SDSS + ADDA)")
    pdf.ln(4)
    _body_text(
        pdf,
        "Real RGB cutouts from SDSS SkyServer were grayscale-converted, center-cropped, and rescaled to 64x64. "
        "ADDA aligned target features to simulations. LensPINN classified Cosmic Horseshoe, Cheshire Cat, and "
        "Five-Image Quasar as no_substructure with >99.9% confidence - consistent with smooth macro-lenses at "
        "survey resolution (zero false-positive substructure detections).",
        size=9,
    )

    horseshoe = PLOTS / "cosmic_horseshoe_64.png"
    if horseshoe.exists():
        pdf.image(str(horseshoe), x=55, y=pdf.get_y(), w=100, h=50)
        pdf.ln(52)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 4, "Fig 3: Preprocessed Cosmic Horseshoe (64x64); Cheshire Cat and Five-Image Quasar processed identically.", align="C")
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 5, "Conclusions")
    pdf.ln(4)
    _body_text(
        pdf,
        "(1) Embedding SIS ray-tracing in the forward pass beats a pure CNN on subhalo detection. "
        "(2) ADDA plus coordinate-based SDSS ingestion enables sim-to-real inference. "
        "(3) Extended training is required for transformer-based physics models. "
        "Future work: JWST/LSST resolution and multi-class domain adaptation.",
    )

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.ln(2)
    pdf.cell(0, 5, "Key files for oral code discussion")
    pdf.ln(3)
    pdf.set_font("Courier", "", 8)
    pdf.set_text_color(*BODY)
    for line in [
        "my_work/utils/physics.py       - SIS deflection and inversion",
        "my_work/models/lens_pinn.py    - theta_E head + residual channels",
        "my_work/train.py / evaluate.py - training and metrics",
        "my_work/train_adda.py          - domain adaptation demo",
        "my_work/classify_real_lenses.py - SDSS inference",
    ]:
        pdf.cell(0, 4, line)
        pdf.ln()

    if pdf.page_no() > 2:
        raise RuntimeError(f"Report exceeded 2 pages ({pdf.page_no()} pages). Trim content.")

    pdf.output(str(REPORT_OUT))
    print(f"Wrote {REPORT_OUT} ({pdf.page_no()} pages)")


class SlidePDF(FPDF):
    def slide_title(self, title: str, subtitle: str = "") -> None:
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 28, style="F")
        self.set_xy(14, 8)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.multi_cell(182, 7, title)
        if subtitle:
            self.set_xy(14, 22)
            self.set_font("Helvetica", "", 9)
            self.cell(0, 5, subtitle)


def _slide_body(pdf: SlidePDF, lines: list[str], size: float = 11, lh: float = 6) -> None:
    pdf.set_xy(14, 36)
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(*BODY)
    for line in lines:
        if line.startswith("## "):
            pdf.set_font("Helvetica", "B", size)
            pdf.multi_cell(182, lh, line[3:])
            pdf.set_font("Helvetica", "", size)
        else:
            pdf.multi_cell(182, lh, line)
        pdf.ln(1)


def build_slides() -> None:
    pdf = SlidePDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(False)

    slides: list[tuple[str, str, list[str]]] = [
        (
            "Physics-Informed ML and Domain Adaptation for Dark Matter Classification",
            "Pallab Mondal | 946804 | AI Models for Physics 2025/2026",
            [
                "MSc Examination Project - Oral Presentation (10 slides)",
                "",
                "Phase 1: ResNet-18 vs LensPINN vs HEALSwin on DeepLense Model I",
                "Phase 2: ADDA domain adaptation + real SDSS lens inference",
            ],
        ),
        (
            "The Astrophysical Problem",
            "",
            [
                "## Strong gravitational lensing",
                "Background light is bent by a foreground mass, forming Einstein rings and arcs.",
                "",
                "## Competing dark matter models",
                "- CDM: point-mass subhalos",
                "- Axion/Fuzzy DM: interference wiggles in arcs",
                "",
                "## Objective",
                "Classify substructure from lensing pixels: simulated training, observational adaptation.",
            ],
        ),
        (
            "Architectures Compared",
            "",
            [
                "1. ResNet-18 - standard CNN on raw pixels (baseline)",
                "2. LensPINN - SIS lens inversion + residual map + CNN decoder",
                "3. HEALSwin - HEALPix projection + Swin Transformer + physics head",
                "4. ADDA - adversarial alignment sim -> telescope domain (Phase 2)",
            ],
        ),
        (
            "LensPINN: Physics-Informed Inversion",
            "",
            [
                "Lens equation:  beta = theta - alpha(theta, theta_E)",
                "SIS deflection: alpha(theta) = theta_E * theta / |theta|",
                "",
                "Pipeline:",
                "1. Predict theta_E from image",
                "2. Reconstruct unlensed source I_source",
                "3. Residual: I_residual = I_orig - I_source",
                "4. Classify from [original, source, residual]",
            ],
        ),
        (
            "Phase 1 Results (Model I Test Set)",
            "",
            [
                "30,000 train / 7,500 test | 64x64 | 3 classes",
                "",
                "Accuracy:  ResNet 43.41%  |  LensPINN 52.35%  |  HEALSwin 33.33%",
                "Macro F1:  0.355       |  0.511            |  0.167",
                "CDM recall: 2%         |  28%              |  0%",
                "",
                "LensPINN +9.0% accuracy; residual subtraction exposes subhalo shears.",
            ],
        ),
        (
            "Why Physics Helps / HEALSwin Lesson",
            "",
            [
                "ResNet: bright Einstein ring masks tiny CDM shears -> 2% CDM recall.",
                "LensPINN: physical background subtraction -> 28% CDM recall.",
                "",
                "HEALSwin at 5 epochs: 33.3% mode collapse.",
                "Physics-informed models must learn theta_E first (LensPINN flat until epoch 16).",
                "Transformers need 30-50 epochs for stable convergence.",
            ],
        ),
        (
            "Phase 2: Domain Adaptation (ADDA)",
            "",
            [
                "Telescope images have noise, PSF blur, atmospheric effects -> domain shift.",
                "",
                "ADDA: freeze source encoder M_s; train target encoder M_t vs discriminator D.",
                "M_t learns to map real images into simulation feature space.",
                "",
                "Live demo: python my_work/train_adda.py --limit_batches 50",
            ],
        ),
        (
            "Real SDSS Targets + Inference",
            "",
            [
                "Downloaded via SDSS SkyServer coordinates:",
                "- Cosmic Horseshoe (J1011+0143)",
                "- Cheshire Cat (J1038+4849)",
                "- Five-Image Quasar (J1004+4112)",
                "",
                "Preprocess: grayscale, center-crop, 64x64, normalize [0,1].",
                "LensPINN predictions: no_substructure at >99.9% (scientifically correct).",
            ],
        ),
        (
            "Code Architecture",
            "",
            [
                "my_work/models/lens_pinn.py  - physics encoder + classifier",
                "my_work/utils/physics.py     - differentiable SIS inversion",
                "my_work/train.py             - supervised training",
                "my_work/evaluate.py          - metrics + confusion matrices",
                "my_work/train_adda.py        - ADDA training",
                "my_work/classify_real_lenses.py - SDSS inference",
                "",
                "Results: results/checkpoints/, results/plots/",
            ],
        ),
        (
            "Conclusions",
            "",
            [
                "1. LensPINN outperforms ResNet on subhalo detection via ray-tracing residuals.",
                "2. ADDA + SDSS pipeline demonstrates sim-to-real inference path.",
                "3. HEALSwin shows physics-parameter learning needs longer training.",
                "",
                "Repository: https://github.com/Pallab9999/DeepLenseProject",
                "",
                "Thank you - questions welcome.",
            ],
        ),
    ]

    assert len(slides) == 10, f"Expected 10 slides, got {len(slides)}"

    for title, subtitle, body in slides:
        pdf.add_page()
        pdf.slide_title(title, subtitle)
        _slide_body(pdf, body)

    pdf.output(str(SLIDES_OUT))
    print(f"Wrote {SLIDES_OUT} ({len(slides)} slides)")


def main() -> None:
    missing = [p for p in [
        PLOTS / "lens_pinn_model_i_cm.png",
        PLOTS / "lens_pinn_model_i_curves.png",
        PLOTS / "cosmic_horseshoe_64.png",
    ] if not p.exists()]
    if missing:
        print("Warning: missing plot files (report may omit figures):")
        for p in missing:
            print(f"  - {p}")

    build_report()
    build_slides()
    print("Done. Verify report is exactly 2 pages before emailing examiners.")


if __name__ == "__main__":
    main()
