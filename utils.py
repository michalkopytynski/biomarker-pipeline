"""
Biomarker Analysis Pipeline — Shared Utilities
===============================================
Transformers, statistical helpers, and plotting functions
used by all three notebooks.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ═══════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════

def long_to_wide(long_df, id_col, biosample_col, value_col, analyte_col="Analyte"):
    """
    Pivot a long-format biomarker DataFrame to wide format.
    If the analyte column already contains the biosample type (detected by
    checking if columns end with _Plasma / _CSF), use it directly.
    Otherwise, columns become <Analyte>_<BiosampleType>.
    """
    long_df = long_df.copy()
    # Check if analyte names already embed biosample type
    sample_names = long_df[analyte_col].unique()
    has_biotype = any(n.endswith("_Plasma") or n.endswith("_CSF") for n in sample_names)

    if has_biotype:
        long_df["_col"] = long_df[analyte_col]
    else:
        long_df["_col"] = long_df[analyte_col] + "_" + long_df[biosample_col]

    wide = long_df.pivot_table(
        index=id_col, columns="_col", values=value_col, aggfunc="first"
    )
    wide.columns.name = None
    return wide.reset_index()


def compute_ratios(df, ratio_pairs, biosample_types=None, epsilon=1e-12):
    """
    For each (numerator, denominator, name) tuple, compute ratio columns.
    Searches for columns that contain the analyte name AND end with _<biotype>.
    """
    if biosample_types is None:
        biosample_types = sorted({
            c.rsplit("_", 1)[-1] for c in df.columns
            if "_" in c and c.rsplit("_", 1)[-1] in ("Plasma", "CSF")
        })

    def _find_col(df, analyte, biotype):
        """Find a column containing `analyte` and ending with `_<biotype>`."""
        suffix = f"_{biotype}"
        candidates = [c for c in df.columns if analyte in c and c.endswith(suffix)]
        return candidates[0] if len(candidates) == 1 else None

    for num, denom, name in ratio_pairs:
        for bs in biosample_types:
            num_col = _find_col(df, num, bs)
            denom_col = _find_col(df, denom, bs)
            new_col = f"{name}_{bs}"
            if num_col and denom_col:
                df[new_col] = df[num_col] / (df[denom_col] + epsilon)
    return df


def encode_categorical(df, mapping_dict, suffix="_num"):
    """Map categorical columns to integers, creating new <col><suffix> columns."""
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col + suffix] = df[col].map(mapping)
    return df


# ═══════════════════════════════════════════════════════════════
# TRANSFORMERS
# ═══════════════════════════════════════════════════════════════

class Log10Transformer(BaseEstimator, TransformerMixin):
    """Log10(x + epsilon) transformer compatible with sklearn Pipeline."""

    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log10(X + self.epsilon)

    def inverse_transform(self, X):
        return 10.0 ** np.asarray(X, dtype=float) - self.epsilon


def build_transform_pipeline(do_log=True, do_impute=False, epsilon=1e-12):
    """Return an sklearn Pipeline for log10 → StandardScaler (with optional imputation)."""
    steps = []
    if do_impute:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if do_log:
        steps.append(("log10", Log10Transformer(epsilon=epsilon)))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def transform_wide_df(wide_df, id_col, skip_log_cols=None, untransformed_cols=None,
                       epsilon=1e-12, impute=True):
    """
    Apply column-wise transformation to a wide-format analyte DataFrame.
    Returns: (transformed_df, imputation_summary)
    """
    skip_log_cols = set(skip_log_cols or [])
    untransformed_cols = set(untransformed_cols or [])
    analyte_cols = [c for c in wide_df.columns if c != id_col]

    # Track imputation
    pre_missing = wide_df[analyte_cols].isna().sum()
    result = wide_df[[id_col]].copy()
    summary_rows = []

    for col in analyte_cols:
        vals = wide_df[[col]].values.astype(float)
        n_missing = int(np.isnan(vals).sum())

        if col in untransformed_cols:
            pipe = build_transform_pipeline(do_log=False, do_impute=impute, epsilon=epsilon)
        elif col in skip_log_cols:
            pipe = build_transform_pipeline(do_log=False, do_impute=impute, epsilon=epsilon)
        else:
            pipe = build_transform_pipeline(do_log=True, do_impute=impute, epsilon=epsilon)

        try:
            transformed = pipe.fit_transform(vals)
            result[col] = transformed.ravel()
        except Exception:
            result[col] = vals.ravel()

        summary_rows.append({
            "column": col,
            "pre_missing": n_missing,
            "pre_missing_pct": n_missing / len(vals) * 100,
            "imputed": n_missing if impute else 0,
        })

    summary = pd.DataFrame(summary_rows)
    return result, summary


# ═══════════════════════════════════════════════════════════════
# STATISTICAL TESTING
# ═══════════════════════════════════════════════════════════════

def welch_t_trimmed(x, y, k=1.5, combine=False):
    """
    Welch's t-test with Tukey outlier trimming.
    Fences: [Q1 - k*IQR, Q3 + k*IQR].
    Falls back to Mann-Whitney U if Welch fails.
    Returns: (p_value, test_name, test_statistic)
    """
    def _trim(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < 3:
            return a
        q1, q3 = np.percentile(a, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return a[(a >= lo) & (a <= hi)]

    xt, yt = _trim(x), _trim(y)

    if len(xt) < 2 or len(yt) < 2:
        return np.nan, "insufficient_data", np.nan

    try:
        stat, p = stats.ttest_ind(xt, yt, equal_var=False)
        return p, "welch_t", stat
    except Exception:
        pass

    try:
        stat, p = stats.mannwhitneyu(xt, yt, alternative="two-sided")
        return p, "mann_whitney", stat
    except Exception:
        return np.nan, "failed", np.nan


def welch_t_trimmed_safe(df, col, group_col, g1, g2, combine=True, k=1.5):
    """Convenience wrapper: extract groups from DataFrame and run trimmed test."""
    a = df.loc[df[group_col] == g1, col].dropna().values
    b = df.loc[df[group_col] == g2, col].dropna().values
    return welch_t_trimmed(a, b, k=k)


def p_to_stars(p):
    """Convert p-value to significance stars."""
    if p is None or np.isnan(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ═══════════════════════════════════════════════════════════════
# CORRELATION HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_corr_and_p(X):
    """Pairwise Pearson r and p-values for all columns in X."""
    cols = X.columns
    n = len(cols)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i in range(n):
        xi = X.iloc[:, i].astype(float)
        for j in range(i, n):
            xj = X.iloc[:, j].astype(float)
            valid = xi.notna() & xj.notna()
            if valid.sum() >= 3:
                r, p = pearsonr(xi[valid], xj[valid])
                corr.iat[i, j] = corr.iat[j, i] = r
                pmat.iat[i, j] = pmat.iat[j, i] = p
    return corr, pmat


def compute_corr_matrix(X):
    """Pearson correlation matrix (no p-values), filling diagonal with 1."""
    cols = X.columns
    n = len(cols)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i in range(n):
        xi = X.iloc[:, i].astype(float)
        for j in range(i, n):
            xj = X.iloc[:, j].astype(float)
            valid = xi.notna() & xj.notna()
            if valid.sum() >= 3:
                r, _ = pearsonr(xi[valid], xj[valid])
                corr.iat[i, j] = corr.iat[j, i] = r
    for i in range(n):
        corr.iat[i, i] = 1.0
    return corr


# ═══════════════════════════════════════════════════════════════
# FEATURE MATRIX BUILDER
# ═══════════════════════════════════════════════════════════════

def build_feature_matrix(df, analyte_cols, mutation_col=None,
                         meta_numeric_keep=None, exclude_keyword=None,
                         include_mutation=True, include_meta=True):
    """
    Build a numeric feature matrix from analyte + optional metadata columns.
    Excludes platform keyword, drops all-NaN and zero-variance columns.
    """
    if exclude_keyword:
        analyte_cols = [c for c in analyte_cols if exclude_keyword not in c.lower()]

    X = df[analyte_cols].apply(pd.to_numeric, errors="coerce")
    extras = []

    if include_mutation and mutation_col and mutation_col in df.columns:
        dummies = pd.get_dummies(df[mutation_col], prefix="Dx.Mutation", dummy_na=False)
        if not dummies.empty:
            extras.append(dummies)

    if include_meta and meta_numeric_keep:
        add_cols = [c for c in meta_numeric_keep if c in df.columns]
        if add_cols:
            extras.append(df[add_cols].apply(pd.to_numeric, errors="coerce"))

    if extras:
        X = pd.concat([X] + extras, axis=1)

    # Drop non-informative columns
    drop_cols = (
        list(X.columns[X.isna().all()]) +
        list(X.columns[X.nunique(dropna=True) <= 1])
    )
    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X


# ═══════════════════════════════════════════════════════════════
# LABEL CLEANING
# ═══════════════════════════════════════════════════════════════

def clean_label(name, remove_parts=None):
    """Strip platform prefixes and biosample suffixes for plot labels."""
    if remove_parts is None:
        remove_parts = [
            "ELISA", "Quanterix", "Vplex1", "Uplex1", "Uplex2", "Uplex3",
            "Splex1", "Splex2", "Dx.Detailed_", "Dx.Mutation_",
            "_Plasma", "_CSF",
        ]
    s = str(name)
    for part in remove_parts:
        s = s.replace(part, "")
    return s.replace("__", "_").replace("_", " ").strip(" _")


def clean_labels(names, remove_parts=None):
    """Vectorised version of clean_label."""
    return [clean_label(n, remove_parts) for n in names]


# ═══════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════

def plot_analyte_box(ax, df, col, group_col, order, palette=None):
    """Box + strip plot for one analyte with significance bracket."""
    import seaborn as sns

    if palette is None:
        palette = {"HC": "steelblue", "ALS": "crimson"}

    sns.boxplot(data=df, x=group_col, y=col, showfliers=True,
                whis=1.5, order=order, ax=ax, palette=palette)
    sns.stripplot(data=df, x=group_col, y=col, color="0.3",
                  alpha=0.5, dodge=True, jitter=0.2, order=order, ax=ax)

    y_max = df[col].max(skipna=True)
    y_min = df[col].min(skipna=True)
    y_range = y_max - y_min if np.isfinite(y_max - y_min) else 1.0
    y = y_max + 0.06 * y_range
    y_line = y_max + 0.03 * y_range

    p, _, _ = welch_t_trimmed_safe(df, col, group_col, g1=order[1], g2=order[0])
    label = p_to_stars(p)

    x1, x2 = 0, 1
    ax.plot([x1, x1, x2, x2], [y_line, y, y, y_line], lw=1.2, c="k")
    ax.text((x1 + x2) / 2, y + 0.01 * y_range, label,
            ha="center", va="bottom", fontsize=9)

    ax.set_title(col, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Scaled value" if "scaled" in col.lower() else "Value", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_ylim(top=y + 0.10 * y_range)


def plot_corr_heatmap(corr, pmat, title, subtitle=None, cmap="coolwarm",
                      fmt=".2f", alpha=0.05, tick_fontsize=8, annot_fontsize=7):
    """Upper-triangular correlation heatmap with X marks for non-significant cells."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr = corr.copy()
    pmat = pmat.copy()
    corr.index = clean_labels(corr.index)
    corr.columns = clean_labels(corr.columns)
    pmat.index = corr.index
    pmat.columns = corr.columns

    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(16, 12), constrained_layout=False)
    ax = sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
        annot=True, fmt=fmt, annot_kws={"size": annot_fontsize},
        square=True, linewidths=0.4,
        cbar_kws={"shrink": 0.7, "label": "Pearson r", "pad": 0.06}
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.0, 1.04, subtitle, transform=ax.transAxes,
                fontsize=11, va="bottom", ha="left", color="dimgray")

    plt.subplots_adjust(bottom=0.25, left=0.18, right=0.95, top=0.92)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i > j:
                p = pmat.iloc[i, j]
                if not np.isnan(p) and p >= alpha:
                    ax.text(j + 0.5, i + 0.5, "X", ha="center", va="center",
                            color="black", fontsize=10, fontweight="bold")

    ax.text(1.01, -0.08, f"X: p ≥ {alpha}", transform=ax.transAxes,
            fontsize=10, color="black")
    plt.tight_layout(rect=[0.05, 0.12, 0.95, 0.92])
    plt.show()
