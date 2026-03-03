
import os
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# ----------------------------
# CONFIG
# ----------------------------
CONTROL = "C0"

MOISTURE_COL = "SUBSTRATE MOISTURE"
STRAIN_COL = "INOCULATION VARIANT"
REP_COL = "REPLICATION"
YEAR_COL = "year"

CFU_COL = "MICROBIAL COUNT [cfu per 1g substrate]"
YIELD_COL = "YIELD [g_plant-1]"

# Traits used in PDMI (exclude yield; yield is for validation / orientation)
PDMI_TRAITS = [
    "Chl total",
    "Carotenoids",
    "Photosyntetic rate (A) [_mol_m-2_s-1]",
    "Stomatal conductance (gs) [mmol_m-2_s-1]",
    "Water Use Efficiency (A/E) [mmol_mol-1]",
    "Fv/Fm",
    "PI Inst.",
    "ELECTROLYTE LEAKAGE (EL) [%]",
    "Relative Water Content (RWC)",
    "Water Saturation Deficit (WSD)",
]

# Cross-year direction estimation (train->test)
TRAIN_YEAR = 2021
TEST_YEAR = 2022

# Bootstrap params
B = 2000
RANDOM_SEED = 42


# ----------------------------
# IO + CLEANING
# ----------------------------
def read_table(path: str) -> pd.DataFrame:
    """
    Reads CSV/TSV robustly:
    - tries separator inference
    - keeps strings, then converts numeric with comma decimal
    """
    # sep inference: if user has TSV-like
    df = pd.read_csv(path, sep=";", engine="python")
    return df


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Converts strings like '1,234' to float 1.234 and keeps NaN where impossible.
    Also strips spaces.
    """
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")

    ss = s.astype(str).str.strip()
    ss = ss.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    # remove thousands separators if any (rare) and convert decimal commas
    # heuristic: if both ',' and '.' exist, assume ',' are thousands -> remove ','
    has_comma = ss.str.contains(",", na=False)
    has_dot = ss.str.contains(r"\.", na=False)
    both = has_comma & has_dot
    ss.loc[both] = ss.loc[both].str.replace(",", "", regex=False)

    # now replace decimal comma with dot
    ss = ss.str.replace(",", ".", regex=False)

    # remove any non-numeric leftovers except ., -, e/E
    ss = ss.str.replace(r"[^0-9eE\.\-\+]", "", regex=True)
    return pd.to_numeric(ss, errors="coerce")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # basic column strip
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # required columns check (soft)
    required = [MOISTURE_COL, STRAIN_COL, REP_COL, YEAR_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # normalize strain strings
    df[STRAIN_COL] = df[STRAIN_COL].astype(str).str.strip()

    # numeric conversions
    numeric_cols = []
    for c in df.columns:
        if c in [MOISTURE_COL, STRAIN_COL]:
            continue
        # treat year/rep as numeric too
        numeric_cols.append(c)

    for c in numeric_cols:
        df[c] = coerce_numeric_series(df[c])

    # force ints where appropriate
    df[REP_COL] = df[REP_COL].astype("Int64")
    df[YEAR_COL] = df[YEAR_COL].astype("Int64")

    # log_cfu
    if CFU_COL in df.columns:
        df["log_cfu"] = np.log10(df[CFU_COL].astype(float).where(df[CFU_COL] > 0))
    else:
        df["log_cfu"] = np.nan

    return df


# ----------------------------
# CORE: replication-level DiD effects (strain vs control)
# ----------------------------
def did_effects_replication_level(
    df: pd.DataFrame,
    trait: str,
    drought_label: str = "DMC",
    optimal_label: str = "OMC",
) -> pd.DataFrame:
    """
    Computes Difference-in-Differences (DiD) at replication level:

    For each (year, rep, strain != C0):
        effect = (DMC_strain - OMC_strain) - (DMC_control - OMC_control)

    This is a clean, publishable definition of "drought mitigation vs control".

    Returns:
        columns: year, replication, strain, trait, effect_vs_control
    """
    if trait not in df.columns:
        raise KeyError(f"Trait not found: {trait}")

    d = df[[MOISTURE_COL, STRAIN_COL, REP_COL, YEAR_COL, trait]].copy()
    d = d.dropna(subset=[MOISTURE_COL, STRAIN_COL, REP_COL, YEAR_COL, trait])

    # Keep only OMC/DMC
    d = d[d[MOISTURE_COL].isin([drought_label, optimal_label])].copy()

    # pivot to have OMC and DMC per (year, rep, strain)
    pv = (
        d.pivot_table(
            index=[YEAR_COL, REP_COL, STRAIN_COL],
            columns=MOISTURE_COL,
            values=trait,
            aggfunc="mean",
        )
        .reset_index()
    )

    # Require both moisture levels for DiD
    if drought_label not in pv.columns or optimal_label not in pv.columns:
        raise ValueError(f"Missing required moisture levels for DiD: {drought_label}/{optimal_label}")

    pv = pv.dropna(subset=[drought_label, optimal_label]).copy()
    pv["delta_moisture"] = pv[drought_label] - pv[optimal_label]

    # bring control baseline per (year, rep)
    ctrl = pv[pv[STRAIN_COL] == CONTROL][[YEAR_COL, REP_COL, "delta_moisture"]].rename(
        columns={"delta_moisture": "delta_ctrl"}
    )
    out = pv.merge(ctrl, on=[YEAR_COL, REP_COL], how="left")

    # drop control itself from outputs; keep only strains
    out = out[out[STRAIN_COL] != CONTROL].copy()

    out["effect_vs_control"] = out["delta_moisture"] - out["delta_ctrl"]
    out["trait"] = trait

    return out[[YEAR_COL, REP_COL, STRAIN_COL, "trait", "effect_vs_control"]]


def build_replication_effects(df: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    parts = []
    for tr in traits:
        parts.append(did_effects_replication_level(df, tr))
    return pd.concat(parts, ignore_index=True)


# ----------------------------
# ORIENTATION (data-driven direction) trained on 2021
# ----------------------------
def yield_effect_replication_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yield DiD effect at replication level:
        YieldEffect_rep = (DMC_strain - OMC_strain) - (DMC_control - OMC_control)
    """
    y = did_effects_replication_level(df, YIELD_COL)
    y = y.rename(columns={"effect_vs_control": "YieldEffect_rep"})
    return y[[YEAR_COL, REP_COL, STRAIN_COL, "YieldEffect_rep"]]


def estimate_trait_direction_from_yield(
    rep_eff: pd.DataFrame,
    y_eff: pd.DataFrame,
    train_year: int = TRAIN_YEAR,
    min_n: int = 10,
) -> dict[str, int]:
    """
    Direction per trait estimated on TRAIN_YEAR only:
      dir(trait) = sign(corr(effect_trait, YieldEffect_rep))

    Returns:
      dict {trait: +1/-1}
    """
    merged = rep_eff.merge(y_eff, on=[YEAR_COL, REP_COL, STRAIN_COL], how="inner")
    merged = merged[merged[YEAR_COL] == train_year].copy()

    directions = {}
    for tr, sub in merged.groupby("trait"):
        sub = sub.dropna(subset=["effect_vs_control", "YieldEffect_rep"])
        if sub.shape[0] < min_n:
            directions[tr] = 1
            continue
        x = sub["effect_vs_control"].to_numpy(dtype=float)
        y = sub["YieldEffect_rep"].to_numpy(dtype=float)
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            directions[tr] = 1
            continue
        r = np.corrcoef(x, y)[0, 1]
        directions[tr] = int(np.sign(r)) if np.isfinite(r) and r != 0 else 1

    return directions


# ----------------------------
# PDMI computation (strain-level) with FDR across strains per trait
# ----------------------------
def strain_level_effects_with_stats(rep_eff: pd.DataFrame) -> pd.DataFrame:
    """
    For each trait and strain:
      - mean effect across replications in TRAIN+TEST combined (for ranking)
      - t-test vs 0 using replication-level values (one-sample t via OLS intercept)
      - p-value + BH-FDR across strains (within trait)
    """
    rows = []
    for (tr, s), sub in rep_eff.groupby(["trait", STRAIN_COL]):
        vals = sub["effect_vs_control"].dropna().to_numpy(dtype=float)
        n = vals.size
        if n < 5:
            continue
        # One-sample test mean != 0 via OLS intercept
        X = np.ones((n, 1))
        model = sm.OLS(vals, X).fit()
        p = float(model.pvalues[0])
        mean_eff = float(model.params[0])
        ci_low, ci_high = model.conf_int()[0]
        rows.append({
            "trait": tr,
            "strain": s,
            "delta": mean_eff,
            "p": p,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n": int(n),
        })

    eff = pd.DataFrame(rows)
    if eff.empty:
        return eff

    # BH-FDR within each trait across strains
    eff["p_adj"] = np.nan
    for tr, sub_idx in eff.groupby("trait").groups.items():
        idx = list(sub_idx)
        pvals = eff.loc[idx, "p"].to_numpy(dtype=float)
        _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        eff.loc[idx, "p_adj"] = p_adj

    return eff


def compute_pdmi(
    rep_eff: pd.DataFrame,
    trait_dir: dict[str, int],
    use_fdr_weights: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds PDMI ranking:
      - aggregates replication-level effects -> strain-level per trait (mean)
      - aligns direction using data-driven trait_dir
      - z-scores across strains within each trait
      - optional weight = -log10(p_adj) (capped) to emphasize robust traits
      - PDMI = mean(weighted z_aligned) across traits

    Returns:
      pdmi_rank (strain, PDMI, n_traits, mean_p_adj)
      trait_effects (trait, strain, delta, p, p_adj, z_aligned, weight)
    """
    eff = strain_level_effects_with_stats(rep_eff)
    if eff.empty:
        return eff, eff

    # align direction
    def get_dir(tr):
        return trait_dir.get(tr, 1)

    eff["dir"] = eff["trait"].map(get_dir).astype(int)
    eff["delta_aligned"] = eff["delta"] * eff["dir"]

    # z-score within trait across strains
    eff["z"] = np.nan
    for tr, idx in eff.groupby("trait").groups.items():
        sub = eff.loc[list(idx), "delta_aligned"].to_numpy(dtype=float)
        mu, sd = np.nanmean(sub), np.nanstd(sub, ddof=0)
        if sd == 0 or not np.isfinite(sd):
            eff.loc[list(idx), "z"] = 0.0
        else:
            eff.loc[list(idx), "z"] = (sub - mu) / sd

    # weights
    if use_fdr_weights:
        # -log10(p_adj), cap to avoid single trait dominating
        w = -np.log10(eff["p_adj"].astype(float).clip(lower=1e-300))
        eff["weight"] = w.clip(upper=5.0)
    else:
        eff["weight"] = 1.0

    eff["z_w"] = eff["z"] * eff["weight"]

    # PDMI = mean weighted z across traits
    pdmi = (
        eff.groupby("strain")
        .agg(
            PDMI=("z_w", "mean"),
            n_traits=("trait", "nunique"),
            mean_p_adj=("p_adj", "mean"),
        )
        .reset_index()
        .sort_values("PDMI", ascending=False)
    )

    return pdmi, eff


# ----------------------------
# Replication-level validation: YieldEffect_rep ~ PDMI(strain)
# done per year (esp. TEST_YEAR)
# ----------------------------
def validate_pdmi_replication_level(
    rep_eff: pd.DataFrame,
    y_eff: pd.DataFrame,
    pdmi_rank: pd.DataFrame,
    year: int,
):
    """
    Uses replication-level YieldEffect as dependent variable and PDMI(strain) as predictor.
    Unit of observation: (year, rep, strain)
    """
    base = y_eff[y_eff[YEAR_COL] == year].merge(
        pdmi_rank[["strain", "PDMI"]],
        left_on=STRAIN_COL,
        right_on="strain",
        how="inner"
    ).copy()

    # drop missing
    base = base.dropna(subset=["YieldEffect_rep", "PDMI"])
    if base.shape[0] < 20:
        print(f"[WARN] Low N for replication-level validation year={year}: N={base.shape[0]}")

    X = sm.add_constant(base["PDMI"].astype(float))
    y = base["YieldEffect_rep"].astype(float)
    model = sm.OLS(y, X).fit()

    return base, model


# ----------------------------
# Bootstrap ranking stability
# resample replicate blocks (year, rep) WITHIN year to keep structure.
# ----------------------------
def bootstrap_pdmi(
    rep_eff: pd.DataFrame,
    y_eff: pd.DataFrame,
    trait_dir: dict[str, int],
    B: int = B,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # unique replicate blocks (year, rep)
    blocks = rep_eff[[YEAR_COL, REP_COL]].drop_duplicates().to_numpy()
    block_list = [tuple(x) for x in blocks]

    strains = sorted(rep_eff[STRAIN_COL].unique().tolist())
    out = []

    for b in range(B):
        sampled_blocks = rng.choice(len(block_list), size=len(block_list), replace=True)
        keep = set(block_list[i] for i in sampled_blocks)

        boot = rep_eff.merge(
            pd.DataFrame(list(keep), columns=[YEAR_COL, REP_COL]),
            on=[YEAR_COL, REP_COL],
            how="inner"
        )

        pdmi_b, _ = compute_pdmi(boot, trait_dir, use_fdr_weights=True)
        pdmi_b = pdmi_b.set_index("strain")["PDMI"].reindex(strains)

        # store
        row = {"b": b}
        for s in strains:
            row[s] = float(pdmi_b.loc[s]) if pd.notna(pdmi_b.loc[s]) else np.nan
        out.append(row)

    return pd.DataFrame(out)


def summarize_bootstrap(boot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per strain:
      mean, sd, CI95, median rank and rank stability
    """
    strains = [c for c in boot_df.columns if c != "b"]
    stats = []
    vals = boot_df[strains].copy()

    # ranks per bootstrap (higher PDMI -> rank 1)
    ranks = vals.rank(axis=1, ascending=False, method="average")

    for s in strains:
        x = vals[s].to_numpy(dtype=float)
        r = ranks[s].to_numpy(dtype=float)
        stats.append({
            "strain": s,
            "PDMI_mean": np.nanmean(x),
            "PDMI_sd": np.nanstd(x),
            "PDMI_ci_low": np.nanpercentile(x, 2.5),
            "PDMI_ci_high": np.nanpercentile(x, 97.5),
            "rank_median": np.nanmedian(r),
            "rank_iqr": np.nanpercentile(r, 75) - np.nanpercentile(r, 25),
        })

    return pd.DataFrame(stats).sort_values("PDMI_mean", ascending=False)


# ----------------------------
# Visualizations: Heatmap + PCA
# ----------------------------
def plot_heatmap_trait_matrix(trait_effects: pd.DataFrame, title: str = "Aligned effects (z)"):
    """
    Heatmap of z-scores (aligned, weighted not used here) for strain x trait.
    """
    mat = trait_effects.pivot_table(index="strain", columns="trait", values="z", aggfunc="mean")
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]  # order strains
    mat = mat[PDMI_TRAITS]  # keep trait order

    fig, ax = plt.subplots(figsize=(12, 6))
    v = np.nanmax(np.abs(mat.to_numpy()))
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    im = ax.imshow(mat.to_numpy(), aspect="auto", norm=norm)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist())
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("z-score (aligned)")

    plt.tight_layout()
    plt.show()


def plot_pca_trait_space(trait_effects: pd.DataFrame, title: str = "PCA of aligned effects (z)"):
    """
    PCA on strain x trait matrix of z (aligned).
    """
    mat = trait_effects.pivot_table(index="strain", columns="trait", values="z", aggfunc="mean")
    mat = mat[PDMI_TRAITS].copy()
    mat = mat.dropna(axis=0, how="any")  # keep complete strains

    X = mat.to_numpy(dtype=float)
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(Z[:, 0], Z[:, 1])
    for i, s in enumerate(mat.index.tolist()):
        ax.text(Z[i, 0], Z[i, 1], s)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def run_pdmi_pipeline(
    data_path: str,
    out_dir: str = ".",
):
    os.makedirs(out_dir, exist_ok=True)

    df_raw = read_table(data_path)
    df = clean_df(df_raw)

    # Basic info
    strains = sorted(df[STRAIN_COL].unique().tolist())
    years = sorted(df[YEAR_COL].dropna().unique().tolist())
    print(f"Rows: {df.shape[0]} | Strains: {len(strains)} | Years: {years}")

    # 1) Replication-level effects for PDMI traits
    rep_eff = build_replication_effects(df, PDMI_TRAITS)

    # 2) Replication-level yield effect
    y_eff = yield_effect_replication_level(df)

    # 3) Direction (data-driven) estimated on TRAIN_YEAR only
    trait_dir = estimate_trait_direction_from_yield(rep_eff, y_eff, train_year=TRAIN_YEAR, min_n=10)
    print("\nTrait directions (trained on 2021):")
    for k in PDMI_TRAITS:
        print(f"  {k:45s} -> {trait_dir.get(k, 1):+d}")

    # 4) PDMI ranking (full data for ranking), using these directions
    pdmi_rank, trait_effects = compute_pdmi(rep_eff, trait_dir, use_fdr_weights=True)

    print("\nTOP PDMI:")
    print(pdmi_rank.head(10))

    # Save core tables
    pdmi_rank.to_csv(os.path.join(out_dir, "pdmi_ranking.csv"), index=False)
    trait_effects.to_csv(os.path.join(out_dir, "pdmi_trait_effects.csv"), index=False)

    # 5) Replication-level validation (esp. TEST_YEAR)
    print(f"\nValidation (rep-level): YieldEffect_rep ~ PDMI on year={TEST_YEAR}")
    base_test, model_test = validate_pdmi_replication_level(rep_eff, y_eff, pdmi_rank, year=TEST_YEAR)
    print(model_test.summary())

    base_test.to_csv(os.path.join(out_dir, f"validation_rep_level_{TEST_YEAR}.csv"), index=False)

    # 6) Bootstrap stability of PDMI ranking
    print(f"\nBootstrapping PDMI stability (B={B}) ...")
    boot = bootstrap_pdmi(rep_eff, y_eff, trait_dir, B=B, seed=RANDOM_SEED)
    boot.to_csv(os.path.join(out_dir, "pdmi_bootstrap_raw.csv"), index=False)

    boot_sum = summarize_bootstrap(boot)
    boot_sum.to_csv(os.path.join(out_dir, "pdmi_bootstrap_summary.csv"), index=False)

    print("\nBootstrap summary (top):")
    print(boot_sum.head(10))

    # 7) Visualizations
    plot_heatmap_trait_matrix(trait_effects, title="PDMI: aligned trait effects (z), strain × trait")
    plot_pca_trait_space(trait_effects, title="PDMI: PCA of aligned trait effects (z)")

    print("\nSaved:")
    print(" - pdmi_ranking.csv")
    print(" - pdmi_trait_effects.csv")
    print(f" - validation_rep_level_{TEST_YEAR}.csv")
    print(" - pdmi_bootstrap_raw.csv")
    print(" - pdmi_bootstrap_summary.csv")

    return {
        "df": df,
        "rep_eff": rep_eff,
        "y_eff": y_eff,
        "trait_dir": trait_dir,
        "pdmi_rank": pdmi_rank,
        "trait_effects": trait_effects,
        "validation_test_data": base_test,
        "validation_test_model": model_test,
        "bootstrap_raw": boot,
        "bootstrap_summary": boot_sum,
    }


# ----------------------------
# RUN (edit path)
# ----------------------------
results = run_pdmi_pipeline("data-2.csv", out_dir="out_pdmi")


#Odczyt i czyszczenie danych
df_raw = read_table("data-2.csv")  # <-- podaj ścieżkę
df = clean_df(df_raw)

#Replication-level efekty dla cech PDMI
rep_eff = build_replication_effects(df, PDMI_TRAITS)

# Replication-level yield (potrzebne było do kierunku)
y_eff = yield_effect_replication_level(df)

# Kierunki estymowane na 2021
trait_dir = estimate_trait_direction_from_yield(
    rep_eff,
    y_eff,
    train_year=TRAIN_YEAR,
    min_n=10
)

# ==========================================
# REP-LEVEL PDMI vs canonical stress markers
# ==========================================

import statsmodels.api as sm
from scipy.stats import pearsonr

# Zbuduj PDMI na poziomie replikacji

def compute_pdmi_rep_level(rep_eff, trait_dir):
    """
    Tworzy PDMI per (year, rep, strain)
    """
    df = rep_eff.copy()

    df["dir"] = df["trait"].map(trait_dir).astype(int)
    df["aligned"] = df["effect_vs_control"] * df["dir"]

    # z-score within trait (global)
    df["z"] = 0.0
    for tr in df["trait"].unique():
        mask = df["trait"] == tr
        vals = df.loc[mask, "aligned"].to_numpy(dtype=float)
        mu = vals.mean()
        sd = vals.std()
        if sd == 0:
            df.loc[mask, "z"] = 0.0
        else:
            df.loc[mask, "z"] = (vals - mu) / sd

    pdmi_rep = (
        df.groupby([YEAR_COL, REP_COL, STRAIN_COL])
        .agg(PDMI_rep=("z", "mean"))
        .reset_index()
    )

    return pdmi_rep


pdmi_rep = compute_pdmi_rep_level(rep_eff, trait_dir)


# obierz rep-level wartości EL, Fv/Fm, RWC

def get_rep_level_trait(df, trait):
    tmp = df[
        [YEAR_COL, REP_COL, STRAIN_COL, trait]
    ].dropna()

    return tmp.rename(columns={trait: trait + "_rep"})


el_rep = get_rep_level_trait(df, "ELECTROLYTE LEAKAGE (EL) [%]")
fvfm_rep = get_rep_level_trait(df, "Fv/Fm")
rwc_rep = get_rep_level_trait(df, "Relative Water Content (RWC)")


# Połącz wszystko

phys_df = (
    pdmi_rep
    .merge(el_rep, on=[YEAR_COL, REP_COL, STRAIN_COL])
    .merge(fvfm_rep, on=[YEAR_COL, REP_COL, STRAIN_COL])
    .merge(rwc_rep, on=[YEAR_COL, REP_COL, STRAIN_COL])
)

print("N rep-level:", phys_df.shape[0])


# Korelacje + modele liniowe

def run_correlation_model(df, y_col):

    x = df["PDMI_rep"].astype(float)
    y = df[y_col].astype(float)

    r, p_corr = pearsonr(x, y)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    print("\n===================================")
    print(f"PDMI_rep ~ {y_col}")
    print("Pearson r =", round(r, 3), "p =", round(p_corr, 5))
    print(model.summary())

    return r, p_corr, model


r_el, p_el, m_el = run_correlation_model(phys_df, "ELECTROLYTE LEAKAGE (EL) [%]_rep")
r_fv, p_fv, m_fv = run_correlation_model(phys_df, "Fv/Fm_rep")
r_rwc, p_rwc, m_rwc = run_correlation_model(phys_df, "Relative Water Content (RWC)_rep")
