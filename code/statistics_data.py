import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from statsmodels.stats.multitest import multipletests

INPUT = "path/to/UNesT_SynthSeg_volume_comparison.csv"
OUT = "path/to/UNesT_SynthSeg_volume_statistics.csv"

df = pd.read_csv(INPUT)

def icc_2_1(x, y):
    data = np.column_stack([x, y])
    n, k = data.shape
    grand_mean = data.mean()

    mean_subject = data.mean(axis=1)
    mean_method = data.mean(axis=0)

    ss_subject = k * np.sum((mean_subject - grand_mean) ** 2)
    ss_method = n * np.sum((mean_method - grand_mean) ** 2)
    residual = data - mean_subject[:, None] - mean_method[None, :] + grand_mean
    ss_error = np.sum(residual ** 2)

    ms_subject = ss_subject / (n - 1)
    ms_method = ss_method / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    return (ms_subject - ms_error) / (
        ms_subject + (k - 1) * ms_error + k * (ms_method - ms_error) / n
    )

results = []

for (canonical_id, laterality), g in df.groupby(
    ["canonical_id", "laterality"]
):

    # Complete pairs only
    g = g.dropna(
        subset=["UNesT_volume_mm3", "SynthSeg_volume_mm3"]
    )

    if len(g) < 3:
        continue

    x = g["UNesT_volume_mm3"].to_numpy(float)
    y = g["SynthSeg_volume_mm3"].to_numpy(float)

    diff = x - y
    mean_xy = (x + y) / 2

    bias = diff.mean()
    sd_diff = diff.std(ddof=1)

    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff

    # Correlations
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    # Proportional bias
    slope, intercept, r_value, p_slope, stderr = linregress(
        mean_xy, diff
    )

    icc = icc_2_1(x, y)

    results.append({
        "canonical_id": canonical_id,
        "laterality": laterality,
        "n": len(g),

        "UNesT_mean_mm3": x.mean(),
        "SynthSeg_mean_mm3": y.mean(),

        "ICC_2_1": icc,

        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,

        "Spearman_r": spearman_r,
        "Spearman_p": spearman_p,

        "BA_bias_mm3": bias,
        "BA_SD_mm3": sd_diff,
        "BA_LOA_lower_mm3": loa_lower,
        "BA_LOA_upper_mm3": loa_upper,

        "BA_proportional_slope": slope,
        "BA_proportional_p": p_slope,
    })


stats_df = pd.DataFrame(results)

# --------------------------------------------------
# FDR correction
# --------------------------------------------------

for col in ["Pearson_p", "Spearman_p", "BA_proportional_p"]:
    mask = stats_df[col].notna()

    _, qvals, _, _ = multipletests(
        stats_df.loc[mask, col],
        method="fdr_bh"
    )

    stats_df.loc[mask, col.replace("_p", "_q")] = qvals


stats_df = stats_df.sort_values(
    ["canonical_id", "laterality"]
)

print(stats_df[
    [
        "canonical_id",
        "laterality",
        "n",
        "ICC_2_1",
        "Pearson_r",
        "Spearman_r",
        "BA_bias_mm3",
        "BA_LOA_lower_mm3",
        "BA_LOA_upper_mm3",
        "BA_proportional_p"
    ]
].sort_values("ICC_2_1", ascending=False))

stats_df["BA_bias_percent"] = (
    stats_df["BA_bias_mm3"] /
    ((stats_df["UNesT_mean_mm3"] + stats_df["SynthSeg_mean_mm3"]) / 2)
) * 100

print(
    stats_df[
        [
            "canonical_id",
            "laterality",
            "n",
            "ICC_2_1",
            "BA_bias_mm3",
            "BA_bias_percent",
            "BA_proportional_p"
        ]
    ].sort_values("ICC_2_1")
)