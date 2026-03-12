# Biomarker Analysis Pipeline

A modular Python pipeline for multi-analyte biomarker case-control studies. Handles multi-sheet Excel data, clinical metadata and produces statistical analyses and visualisations.

Originally developed for ALS CSF/Plasma proteomics but designed to work with **any two-group biomarker dataset**.

## Pipeline Overview

| Notebook | What it does | Key outputs |
|---|---|---|
| **01 — Data Processing** | Excel → long → wide, metadata merge, categorical encoding, log₁₀ + scaling, imputation | `wide_scaled_imputed.csv`, `dfm_merged.csv` |
| **02 — Statistical Analysis** | Box/strip plots with significance brackets, PCA, OLS volcano plots with FDR | PPTX of analyte plots, volcano figures |
| **03 — Correlation & ML** | Correlation heatmaps, top-pair scatter grids, Random Forest classification, dendrograms, alluvial plots | Feature importance rankings, cluster visualisations |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/biomarker-pipeline.git
cd biomarker-pipeline
pip install -r requirements.txt

# 2. Generate sample data (or place your own in data/)
python scripts/generate_sample_data.py

# 3. Run notebooks
#    Option A: Open notebooks/ in JupyterLab or VS Code
#    Option B: Run the .py scripts directly
python scripts/01_data_processing.py
python scripts/02_statistical_analysis.py
python scripts/03_correlation_and_ml.py
```

## Adapting to Your Data

All configuration lives in **`config.py`** — edit once, all notebooks update:

1. **File paths**: Point `RESULTS_FILE` and `METADATA_FILE` to your Excel files
2. **Column names**: Set `ID_COL`, `GROUP_COL`, `BIOSAMPLE_COL`, `VALUE_COL`
3. **Group labels**: Set `CASE_LABEL` / `CONTROL_LABEL` (e.g. `"Disease"` / `"Control"`)
4. **Metadata mappings**: Update `META_CATEGORICAL`, `META_BINARY`, etc.
5. **Platform exclusion**: Set `EXCLUDE_PLATFORM_KEYWORD` or `None` to include all

### Expected Data Format

See [`data/README.md`](data/README.md) for full schema details.

## Project Structure

```
biomarker-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                     # All configurable parameters
├── utils.py                      # Shared transformers, stats, plotting
│
├── notebooks/                    # Jupyter notebooks (rendered on GitHub)
│   ├── generate_sample_data.ipynb
│   ├── 01_data_processing.ipynb
│   ├── 02_statistical_analysis.ipynb
│   └── 03_correlation_and_ml.ipynb
│
├── scripts/                      # Equivalent .py files (cleaner git diffs)
│   ├── generate_sample_data.py
│   ├── 01_data_processing.py
│   ├── 02_statistical_analysis.py
│   └── 03_correlation_and_ml.py
│
├── data/                         # Input data (gitignored, see data/README.md)
│   └── README.md
│
└── outputs/                      # Generated outputs (gitignored)
    └── .gitkeep
```

## Key Methods

- **Transformation**: log₁₀(x + ε) → StandardScaler, with configurable skip lists
- **Group comparison**: Welch's t-test with Tukey outlier trimming (k=1.5 IQR), Mann-Whitney fallback
- **Volcano plots**: OLS with HC3 robust standard errors, multiple covariate specifications, Benjamini-Hochberg FDR
- **Correlation**: Pairwise Pearson r with significance masking (p ≥ 0.05 marked with ×)
- **Classification**: Random Forest (500 trees, balanced classes) with correlation-weighted features
- **Clustering**: Hierarchical clustering on 1−|r| distance, alluvial plots showing HC→ALS cluster reassignment

## License

MIT
