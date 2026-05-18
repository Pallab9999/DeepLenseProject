import os

try:
    from fpdf import FPDF
except ImportError:
    import sys
    print("fpdf not found, please ensure it's installed.")
    sys.exit(1)

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'PROPOSAL OF EXAMINATION PROJECT', 0, 1, 'C')
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Models for Physics 2025/2026', 0, 1, 'C')
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)

# Information
info_text = (
    "Name: Pallab    Last name: Mondal    Student ID: 946804    Date: 16/04/2026\n\n"
    "Title: Physics-Informed ML and Domain Adaptation for Gravitational Lensing\n\n"
    "Learning method/algorithm used:\n"
    "Physics-Informed Neural Networks (LensPINN), Vision Transformers (HEALSwin), "
    "Unsupervised Domain Adaptation (ADDA).\n\n"
    "Objective: (max 3 lines):\n"
    "To build a physics-informed model classifying 4-class dark matter substructures in\n"
    "gravitational lenses, adapting from synthetic dataset simulations to real telescope imagery\n"
    "(HSC) using unsupervised domain adaptation.\n\n"
    "Based on a previously existing project:\n"
    "( ) No, it is totally new not based on code of others\n"
    "(X) Yes, I have partly used code from the following repositories:\n"
    "url: https://github.com/ML4SCI/DeepLense\n\n"
    "Reference used:\n"
    "Link(s):\n"
    "http://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf (LensPINN)\n"
    "http://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_252.pdf (HEAL-PINN)\n"
    "http://arxiv.org/abs/2112.12121 (Domain Adaptation)\n"
    "http://arxiv.org/abs/1702.05464 (ADDA)\n\n"
    "Needed dataset:\n"
    "( ) Not needed\n"
    "(X) Yes: I have original data from a research project (DeepLense synthetic datasets "
    "provided by the ML4SCI collaboration)\n"
    "( ) Yes: Published here url: http//....\n"
)

# Since FPDF might have trouble with automatic wrapping over very long strings without multi_cell
for line in info_text.split('\n'):
    pdf.multi_cell(0, 8, line)

out_path = os.path.join(os.path.dirname(__file__), "examination_project_proposal_946804.pdf")
pdf.output(out_path, 'F')
print(f"PDF generated successfully at {out_path}")
