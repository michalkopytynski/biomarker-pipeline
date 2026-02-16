import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# %% [markdown]
# # Notebook 01 — Data Processing & Transformation
#
# **Pipeline**: Load multi-sheet Excel → long format → wide format → metadata merge
# → categorical encoding → ratio computation → log10 + scaling → imputation summary
#
# **Outputs saved to `outputs/`:**
# - `wide_raw.csv` — wide-format raw concentrations
# - `wide_scaled.csv` — transformed & scaled (non-imputed)
# - `wide_scaled_imputed.csv` — transformed, scaled & median-imputed
# - `metadata_numerical.csv` — encoded metadata
# - `imputation_summary.csv`

# %%
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import *
from utils import (
    long_to_wide, compute_ratios, encode_categorical,
    transform_wide_df, Log10Transformer
)

# %% [markdown]
# ## 1 · Load analyte results (one sheet per analyte)

# %%
xls = pd.ExcelFile(RESULTS_FILE, engine="openpyxl")
sheet_names = xls.sheet_names
print(f"Found {len(sheet_names)} analyte sheets")

frames = []
for sheet in sheet_names:
    tmp = xls.parse(sheet)
    # Extract analyte name from sheet (sheet format: Platform_Analyte_Biotype)
    tmp["Analyte"] = sheet
    frames.append(tmp[[ID_COL, BIOSAMPLE_COL, VALUE_COL, "Analyte"]])

long_df = pd.concat(frames, ignore_index=True)
print(f"Long-format shape: {long_df.shape}")
long_df.head()

# %% [markdown]
# ## 2 · Pivot to wide format

# %%
wide_raw = long_to_wide(long_df, ID_COL, BIOSAMPLE_COL, VALUE_COL, "Analyte")
print(f"Wide-format shape: {wide_raw.shape}  ({wide_raw.shape[1] - 1} analyte columns)")
wide_raw.head()

# %% [markdown]
# ## 3 · Load & merge clinical metadata

# %%
meta = pd.read_excel(METADATA_FILE, sheet_name=METADATA_SHEET, engine="openpyxl")
print(f"Metadata shape: {meta.shape}")

# Keep only relevant columns that exist
meta_cols_to_keep = [ID_COL, GROUP_COL] + META_CONTINUOUS + list(META_CATEGORICAL.keys())
meta_cols_to_keep += [MUTATION_COL] + META_BINARY + META_ORDINAL
meta_cols_to_keep += [c for c in ["Dx.Detailed"] if c in meta.columns]
meta_cols_to_keep = [c for c in meta_cols_to_keep if c in meta.columns]
meta = meta[list(dict.fromkeys(meta_cols_to_keep))]  # deduplicate, preserve order

meta.head()

# %% [markdown]
# ## 4 · Encode categorical variables

# %%
metadata_numerical = meta.copy()
metadata_numerical = encode_categorical(metadata_numerical, META_CATEGORICAL, suffix="_num")
print("Encoded columns:", [c for c in metadata_numerical.columns if c.endswith("_num")])
metadata_numerical.head()

# %% [markdown]
# ## 5 · Compute derived ratio columns

# %%
if RATIO_PAIRS:
    wide_raw = compute_ratios(wide_raw, RATIO_PAIRS, epsilon=LOG10_EPSILON)
    ratio_cols = [c for c in wide_raw.columns if any(r[2] in c for r in RATIO_PAIRS)]
    print(f"Added ratio columns: {ratio_cols}")
else:
    print("No ratio pairs configured.")

# %% [markdown]
# ## 6 · Log10 transform + StandardScaler (with and without imputation)

# %%
# Non-imputed version
wide_scaled, _ = transform_wide_df(
    wide_raw, ID_COL,
    skip_log_cols=SKIP_LOG_COLS,
    untransformed_cols=UNTRANSFORMED_COLS,
    epsilon=LOG10_EPSILON,
    impute=False
)

# Imputed version
wide_scaled_imputed, imputation_summary = transform_wide_df(
    wide_raw, ID_COL,
    skip_log_cols=SKIP_LOG_COLS,
    untransformed_cols=UNTRANSFORMED_COLS,
    epsilon=LOG10_EPSILON,
    impute=True
)

print(f"Scaled shape (no imputation): {wide_scaled.shape}")
print(f"Scaled shape (imputed):       {wide_scaled_imputed.shape}")
print(f"\nImputation summary (top 10 by missing %):")
imputation_summary.sort_values("pre_missing_pct", ascending=False).head(10)

# %% [markdown]
# ## 7 · Merge metadata with scaled data

# %%
dfm = pd.merge(metadata_numerical, wide_scaled_imputed, on=ID_COL, how="inner")
print(f"Merged DataFrame shape: {dfm.shape}")

# Identify analyte columns (everything that's not metadata)
meta_col_set = set(metadata_numerical.columns)
analyte_cols = [c for c in dfm.columns if c not in meta_col_set]
print(f"Analyte columns: {len(analyte_cols)}")

# Split by biosample type
plasma_cols = [c for c in analyte_cols if c.endswith("_Plasma")]
csf_cols    = [c for c in analyte_cols if c.endswith("_CSF")]
print(f"  Plasma: {len(plasma_cols)}  |  CSF: {len(csf_cols)}")

# %% [markdown]
# ## 8 · Save outputs

# %%
wide_raw.to_csv(OUTPUT_DIR / "wide_raw.csv", index=False)
wide_scaled.to_csv(OUTPUT_DIR / "wide_scaled.csv", index=False)
wide_scaled_imputed.to_csv(OUTPUT_DIR / "wide_scaled_imputed.csv", index=False)
metadata_numerical.to_csv(OUTPUT_DIR / "metadata_numerical.csv", index=False)
imputation_summary.to_csv(OUTPUT_DIR / "imputation_summary.csv", index=False)
dfm.to_csv(OUTPUT_DIR / "dfm_merged.csv", index=False)

print("✓ All outputs saved to outputs/")
print("  → wide_raw.csv")
print("  → wide_scaled.csv")
print("  → wide_scaled_imputed.csv")
print("  → metadata_numerical.csv")
print("  → imputation_summary.csv")
print("  → dfm_merged.csv")
