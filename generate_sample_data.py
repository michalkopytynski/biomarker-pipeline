import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# %% [markdown]
# # Generate Synthetic Biomarker Data
#
# Creates two Excel files that mimic a real multi-analyte biomarker study:
# - `data/biomarker_results.xlsx` — one sheet per analyte (long format)
# - `data/metadata.xlsx` — clinical metadata (one sheet)
#
# Run standalone: `python generate_sample_data.py`

# %%
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# %% [markdown]
# ## Study Design & Analyte Definitions

# %%
N_ALS = 50
N_HC = 30
N = N_ALS + N_HC

donor_ids = [f"D{i:03d}" for i in range(1, N + 1)]
dx_status = ["ALS"] * N_ALS + ["HC"] * N_HC

# ── Analytes ──────────────────────────────────────────────────
# (name, platform_prefix, biotype, ALS_mean, HC_mean, shared_sd)
analyte_defs = [
    ("AB40",  "ELISA",     "Plasma", 250, 220, 40),
    ("AB42",  "ELISA",     "Plasma", 35,  45,  10),
    ("NfL",   "Quanterix", "Plasma", 45,  15,  12),
    ("GFAP",  "Quanterix", "Plasma", 180, 90,  50),
    ("CHIT1", "ELISA",     "Plasma", 30,  20,  8),
    ("IL-6",  "Splex1",    "Plasma", 8,   5,   3),
    ("IL-18", "Uplex1",    "Plasma", 280, 250, 60),
    ("BM-A",  "ELISA",     "Plasma", 2.5, 2.0, 0.8),
    ("BM-B",  "ELISA",     "Plasma", 1.2, 1.0, 0.4),
    ("BM-C",  "ELISA",     "Plasma", 0.9, 0.7, 0.3),
    ("BM-D",  "ELISA",     "Plasma", 0.5, 0.4, 0.2),
    ("BM-E",  "ELISA",     "Plasma", 3.0, 2.2, 1.0),
    ("BM-F",  "Uplex1",    "Plasma", 400, 350, 80),
    ("BM-G",  "Uplex2",    "Plasma", 120, 80,  30),
    ("BM-H",  "Vplex1",    "Plasma", 350, 300, 60),
    ("BM-I",  "Vplex1",    "Plasma", 15,  12,  4),
    ("BM-J",  "Vplex1",    "Plasma", 600, 500, 120),
    ("BM-K",  "Luminex",   "Plasma", 340, 300, 55),
    ("BM-L",  "Luminex",   "Plasma", 14,  11,  4),
    ("BM-M",  "Luminex",   "Plasma", 7,   5,   3),

    ("AB40",  "ELISA",     "CSF", 5500, 6200, 800),
    ("AB42",  "ELISA",     "CSF", 400,  600,  120),
    ("NfL",   "Quanterix", "CSF", 3500, 800,  900),
    ("GFAP",  "Quanterix", "CSF", 9000, 4500, 2000),
    ("CHIT1", "ELISA",     "CSF", 4000, 2000, 1000),
    ("IL-6",  "Splex1",    "CSF", 4.0,  2.5,  1.5),
    ("IL-18", "Uplex3",    "CSF", 180,  150,  40),
    ("BM-A",  "ELISA",     "CSF", 1.8,  1.5,  0.5),
    ("BM-D",  "ELISA",     "CSF", 0.3,  0.25, 0.1),
    ("BM-E",  "ELISA",     "CSF", 2.0,  1.5,  0.7),
    ("BM-F",  "Uplex3",    "CSF", 80,   60,   25),
    ("BM-N",  "Uplex1",    "CSF", 5.0,  3.0,  2.0),
    ("BM-K",  "Luminex",   "CSF", 500,  450,  80),
    ("BM-O",  "Luminex",   "CSF", 170,  150,  35),
    ("BM-P",  "Luminex",   "CSF", 1200, 1000, 250),
]

biosample_types = ["Plasma", "CSF"]

# %% [markdown]
# ## Generate Results Workbook

# %%
Path("data").mkdir(exist_ok=True)

sheets = {}
for (analyte, platform, biotype, als_mu, hc_mu, sd) in analyte_defs:
    sheet_name = f"{platform}_{analyte}_{biotype}"
    rows = []
    for i, did in enumerate(donor_ids):
        mu = als_mu if dx_status[i] == "ALS" else hc_mu
        val = max(0, np.random.normal(mu, sd))
        # Sprinkle ~5 % missing
        if np.random.rand() < 0.05:
            val = np.nan
        rows.append({
            "Donor.ID": did,
            "biosample.type": biotype,
            "Dilution.Corrected.Conc": val,
        })
    sheets[sheet_name] = pd.DataFrame(rows)

with pd.ExcelWriter("data/biomarker_results.xlsx", engine="openpyxl") as w:
    for name, df in sheets.items():
        df.to_excel(w, sheet_name=name[:31], index=False)  # Excel 31-char limit

print(f"✓ Created data/biomarker_results.xlsx  ({len(sheets)} sheets, {N} donors)")

# %% [markdown]
# ## Generate Metadata Workbook

# %%
mutations = ["C9ORF72", "SOD1", "ANXA11", "FUS", "TARDBP", None]
onset_sites = ["Limb", "Bulbar", "Respiratory", "Cognitive"]
detailed_dx = []
mutation_list = []

for i, dx in enumerate(dx_status):
    if dx == "ALS":
        detailed_dx.append(np.random.choice(["sALS", "fALS"], p=[0.7, 0.3]))
        mutation_list.append(
            np.random.choice(mutations, p=[0.15, 0.08, 0.03, 0.02, 0.02, 0.70])
        )
    else:
        detailed_dx.append(np.random.choice(["HC", "HC_carrier"], p=[0.85, 0.15]))
        mutation_list.append(
            np.random.choice([None, "C9ORF72", "SOD1"], p=[0.90, 0.05, 0.05])
        )

meta = pd.DataFrame({
    "Donor.ID": donor_ids,
    "Dx.Status": dx_status,
    "Dx.Detailed": detailed_dx,
    "Dx.Mutation": mutation_list,
    "Sex": np.random.choice(["Male", "Female"], N, p=[0.55, 0.45]),
    "Age": np.random.normal(62, 10, N).astype(int).clip(30, 90),
    "Site_of_Onset": [
        np.random.choice(onset_sites, p=[0.6, 0.25, 0.1, 0.05])
        if dx == "ALS" else np.nan
        for dx in dx_status
    ],
    "El_Escorial_num": [
        np.random.choice([1, 2, 3, 4]) if dx == "ALS" else np.nan
        for dx in dx_status
    ],
    "ALSFRSR": [
        round(max(0, min(48, np.random.normal(36, 8))), 1) if dx == "ALS" else np.nan
        for dx in dx_status
    ],
    "Radicava":  [int(np.random.rand() < 0.15) if dx == "ALS" else 0 for dx in dx_status],
    "Rilutek":   [int(np.random.rand() < 0.60) if dx == "ALS" else 0 for dx in dx_status],
    "Tofersen":  [int(np.random.rand() < 0.05) if dx == "ALS" else 0 for dx in dx_status],
})

with pd.ExcelWriter("data/metadata.xlsx", engine="openpyxl") as w:
    meta.to_excel(w, sheet_name="Clinical info", index=False)

print(f"✓ Created data/metadata.xlsx  ({N} donors, {meta.shape[1]} columns)")
print("\nSample data ready — run the notebooks!")
