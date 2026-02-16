"""
Biomarker Analysis Pipeline — Configuration
============================================
Edit this file to adapt the pipeline to your dataset.
All notebooks import from here so you only change things once.
"""

from pathlib import Path

# ── File paths ────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Main results file: an Excel workbook where each sheet = one analyte
# Expected columns per sheet: <ID_COL>, <BIOSAMPLE_COL>, <VALUE_COL>
RESULTS_FILE = DATA_DIR / "biomarker_results.xlsx"

# Clinical metadata file (single sheet)
METADATA_FILE = DATA_DIR / "metadata.xlsx"
METADATA_SHEET = "Clinical info"

# ── Column naming conventions ─────────────────────────────────
ID_COL = "Donor.ID"
GROUP_COL = "Dx.Status"
BIOSAMPLE_COL = "biosample.type"
VALUE_COL = "Dilution.Corrected.Conc"

# Group labels
CASE_LABEL = "ALS"
CONTROL_LABEL = "HC"

# ── Metadata columns ──────────────────────────────────────────
META_CONTINUOUS = ["Age", "ALSFRSR"]

META_CATEGORICAL = {
    "Sex": {"Female": 0, "Male": 1},
    "Site_of_Onset": {"Limb": 1, "Bulbar": 2, "Respiratory": 3, "Cognitive": 4},
    "Dx.Detailed": {"sALS": 1, "fALS": 2, "HC_carrier": 3, "HC": 4},
}

MUTATION_COL = "Dx.Mutation"
META_BINARY = ["Radicava", "Rilutek", "Tofersen"]
META_ORDINAL = ["El_Escorial_num"]
META_NUMERIC_KEEP = META_ORDINAL + META_BINARY + ["ALSFRSR", "Sex_num", "Age"]

# ── Derived ratio columns ─────────────────────────────────────
RATIO_PAIRS = [("AB42", "AB40", "AB42/AB40")]

# ── Transformation settings ───────────────────────────────────
LOG10_EPSILON = 1e-12
SKIP_LOG_COLS = []
UNTRANSFORMED_COLS = []

# ── Platform exclusion ────────────────────────────────────────
EXCLUDE_PLATFORM_KEYWORD = "luminex"  # set to None to include all

# ── Statistical testing ───────────────────────────────────────
ALPHA = 0.05
WELCH_TRIM_K = 1.5
USE_ROBUST_SE = True

VOLCANO_SPECS = {
    "Sex+Age":       ["Sex_num", "Age"],
    "Sex+Age+PC1":   ["Sex_num", "Age", "PC1"],
    "Sex+Age+PC2":   ["Sex_num", "Age", "PC2"],
    "PC1 only":      ["PC1"],
    "PC2 only":      ["PC2"],
}

# ── Random Forest ─────────────────────────────────────────────
RF_N_ESTIMATORS = 500
RF_CV_FOLDS = 5
RF_RANDOM_STATE = 42
TOP_N_FEATURES = 30

# ── Alluvial / cluster ────────────────────────────────────────
N_FAMILIES = 5
MIN_FLOW_SIZE = 1

# ── Visualization ─────────────────────────────────────────────
PLOTS_PER_PAGE = 16
FIG_DPI = 220
PALETTE = {"HC": "steelblue", "ALS": "crimson"}
EXPORT_PPTX = True

# ── Cross-platform concordance pairs ─────────────────────────
CONCORDANCE_PAIRS_PLASMA = []
CONCORDANCE_PAIRS_CSF = []
