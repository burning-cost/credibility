# credibility
[![Tests](https://github.com/burning-cost/credibility/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/credibility/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/credibility)](https://pypi.org/project/credibility/)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Bühlmann-Straub credibility models for non-life insurance pricing. For UK scheme and affinity pricing teams who currently do this in Excel.

---

## The problem

You price a portfolio of schemes. Scheme A has three years of data and a 72% loss ratio. The book average is 58%. How much do you trust scheme A's own experience when setting next year's rate?

If you give it full weight, you are at the mercy of three years of claims volatility. If you ignore it entirely and use the book rate, you are mispricing the scheme's risk profile. The right answer is somewhere between the two - and the position depends on:

1. How much exposure scheme A has (more exposure -> more trust in its own data)
2. How noisy year-to-year loss ratios are across the portfolio (noisier -> trust the book more)
3. How genuinely different schemes are from each other (more heterogeneous -> trust each scheme's own data more)

Bühlmann-Straub credibility theory gives a mathematically rigorous way to estimate all three and compute the optimal blend. This package implements it in Python, with no dependencies beyond NumPy and Polars.

---

## Blog post

[Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](https://burning-cost.github.io/2026/03/06/buhlmann-straub-credibility-in-python/)

---

## What it does

- **`BuhlmannStraub`**: the standard two-parameter model. Given a panel of loss rates with exposure weights, it estimates the structural parameters (EPV, VHM, Bühlmann's k) and produces credibility-weighted premiums for each group.

- **`HierarchicalBuhlmannStraub`**: multi-level extension for nested structures. Useful for geographic rating (region -> district -> sector) or organisational hierarchies (portfolio -> scheme -> sub-scheme), where thin nodes borrow strength from their parent.

---

## Installation

```bash
pip install credibility
```

Or with uv:

```bash
uv add credibility
```

Requires Python 3.9+ and polars, numpy.

If you want to pass pandas DataFrames as input (they are converted to Polars internally), install the optional pandas bridge:

```bash
pip install "credibility[pandas]"
```

---

## Quick start

### Scheme pricing

```python
import polars as pl
from credibility import BuhlmannStraub

# One row per scheme per year. loss_rate is ultimate loss ratio.
# exposure is earned premium (or earned car years, or policy count -
# whatever you are weighting by).
df = pl.DataFrame({
    "scheme":    ["Motor Guild", "Motor Guild", "Motor Guild",
                  "Teachers",    "Teachers",    "Teachers",
                  "NHS Staff",   "NHS Staff",   "NHS Staff"],
    "year":      [2021, 2022, 2023] * 3,
    "loss_rate": [0.72, 0.68, 0.74,   # Motor Guild: volatile, above-average
                  0.55, 0.52, 0.57,   # Teachers: stable, below-average
                  0.63, 0.60, 0.61],  # NHS Staff: stable, near book
    "exposure":  [1_200, 1_350, 1_100,
                  4_500, 4_800, 5_100,
                  800,   850,   900],
})

bs = BuhlmannStraub()
bs.fit(df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

bs.summary()
```

Output:

```
Bühlmann-Straub Credibility Model
==========================================
  Collective mean    mu  = 0.6153
  Process variance   v   = 0.000526   (EPV, within-group)
  Between-group var  a   = 0.004018   (VHM, between-group)
  Credibility param  k   = 0.131      (v / a)

  Interpretation: a group needs exposure = k to achieve Z = 0.50

  group         Exposure  Obs. Mean  Z       Cred. Premium  Complement
  Motor Guild     3650     0.713    0.9996      0.713          0.615
  Teachers       14400     0.547    1.0000      0.547          0.615
  NHS Staff       2550     0.613    0.9995      0.613          0.615
```

The structural parameters:

- **mu = 0.615**: book-wide weighted mean loss ratio - the complement all groups blend toward.
- **v = 0.000526**: EPV, the expected within-scheme year-to-year variance. Low: these schemes are fairly stable year to year.
- **a = 0.004018**: VHM, the variance of true underlying loss ratios between schemes. Relatively high compared to v - the schemes genuinely differ. This drives Bühlmann's k very low (k = 0.131), which means Z is close to 1 even for small exposures.
- **k = 0.131**: a scheme only needs 131 units of exposure to reach Z = 0.50. With the exposures here, all three schemes are at near-full credibility.

For less heterogeneous portfolios (where k is much larger), the blend matters more.

### Accessing results programmatically

```python
bs.mu_hat_   # 0.6153 - collective mean
bs.v_hat_    # 0.000526 - EPV (within-group variance)
bs.a_hat_    # 0.004018 - VHM (between-group variance)
bs.k_        # 0.131 - Bühlmann's k

# z_ is a pl.DataFrame with columns ["group", "Z"]
bs.z_
# shape: (3, 2)
# group        Z
# Motor Guild  0.9996
# Teachers     1.0000
# NHS Staff    0.9995

# Look up a specific group's Z:
bs.z_.filter(pl.col("group") == "Teachers")["Z"][0]

# premiums_ is a pl.DataFrame with group, exposure, observed_mean, Z,
# credibility_premium, complement columns
bs.premiums_
```

### Hierarchical credibility for geographic rating

Three-level postcode hierarchy. Each postcode sector borrows strength from its district, each district from its area.

```python
import polars as pl
from credibility import HierarchicalBuhlmannStraub

model = HierarchicalBuhlmannStraub(level_cols=["area", "district", "sector"])
model.fit(
    df,
    period_col="year",
    loss_col="loss_rate",
    weight_col="exposure",
)

model.summary()                      # structural parameters at each level
model.premiums_at("sector")         # blended sector-level premiums (pl.DataFrame)
model.premiums_at("district")       # blended district-level premiums
model.level_results_["district"].k  # Bühlmann's k at district level
```

The model fits variance components bottom-up (sector -> district -> area) and then computes premiums top-down, so each sector's final premium reflects the information at all three levels.

### Using pandas DataFrames

If you already have pandas DataFrames, the library accepts them directly and converts internally. The output is always Polars.

```python
import pandas as pd
from credibility import BuhlmannStraub

df_pd = pd.read_csv("scheme_data.csv")  # pandas DataFrame
bs = BuhlmannStraub()
bs.fit(df_pd, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

# premiums_ is a pl.DataFrame regardless of input type
bs.premiums_
```

---

## Structural parameters - what they tell you

| Parameter | Symbol | Meaning |
|---|---|---|
| Collective mean | mu | Grand weighted average loss rate across all groups |
| Process variance | v (EPV) | Average within-group year-to-year variance. High v -> noisy data -> lean on book |
| Between-group variance | a (VHM) | Variance of true loss rates between groups. High a -> groups genuinely differ -> trust their own data |
| Bühlmann's k | k = v/a | Exposure at which a group reaches Z = 0.5. Small k -> fast path to credibility |
| Credibility factor | Z_i | W_i / (W_i + k), where W_i is the group's total exposure |
| Credibility premium | P_i | Z_i * X_bar_i + (1 - Z_i) * mu |

If the a_hat estimate is negative (rare, but possible with fewer than ~5 groups or a very homogeneous portfolio), the model truncates it to zero and sets all Z = 0. This means the data gives no evidence of between-group heterogeneity - every group gets the collective mean.

---

## The Hachemeister benchmark

The standard validation dataset is Hachemeister (1975): 5 US states, 12 quarters of bodily injury liability loss ratios with exposure weights. The R package `actuar` ships this dataset and uses it to validate `cm()`.

Our model produces the following structural parameters on this dataset:

| Parameter | Value |
|---|---|
| mu | 1832.82 |
| v (EPV) | 136,793,601 |
| a (VHM) | 100,302 |
| k | 1363.82 |

The large v reflects substantial quarter-to-quarter claim severity variation within each state. k = 1364 means a state needs roughly 1,364 claims to reach Z = 0.50 - State 1 with 108,722 total claims reaches Z = 0.988 (near-full credibility), while State 4 with 9,959 claims reaches Z = 0.880.

---

## Design decisions

**Why Polars instead of pandas?** The core operations are group aggregations and joins. Polars expresses these without the index machinery that pandas requires, the API is explicit about column names throughout, and the lazy evaluation makes the within-group variance computation efficient without Python loops. The `_to_polars()` bridge means existing pandas users are not broken.

**Why no scipy?** The structural parameter estimators (Bühlmann-Straub 1970) are closed-form. There is no optimisation step, no matrix decomposition beyond what numpy handles. Keeping the dependency list minimal (numpy, polars) makes the package easier to deploy in restricted environments.

**Why a `group` column instead of a DataFrame index?** Polars does not have a row index in the pandas sense. Using a named `group` column is explicit, composable (you can join it to other DataFrames trivially), and avoids the confusion that arises when an index has a name in pandas but behaves differently from a column.

**Why truncate negative a_hat rather than using an iterative estimator?** The closed-form estimator is transparent and exactly matches actuar's default `method="unbiased"`. Negative a_hat is a signal, not a failure - it tells you the portfolio appears homogeneous. Truncating at zero is the standard actuarial convention (Bühlmann & Gisler, 2005, §4.3).

**Why separate `HierarchicalBuhlmannStraub` instead of a `levels` argument?** The hierarchical model has genuinely different output (per-level parameters, multi-level premiums). Merging it with `BuhlmannStraub` would make both classes harder to use.

---

## Relationship to other methods

- **Ridge regression on dummy variables is mathematically equivalent to Bühlmann-Straub credibility** when the ridge parameter K is estimated from data (Ohlsson, 2008). If you are using L2-penalised GLMs (e.g., glum), you are already doing credibility weighting. This package makes the K, Z, and structural parameters explicit.

- **statsmodels MixedLM** fits the same model under Normal response. It does not produce actuarial output (Z factors, structural parameters in v/a form) and has no hierarchical Bühlmann-Straub equivalent.

- **actuar (R)** is the gold standard reference. This package matches its output on the Hachemeister dataset.

- **[insurance-multilevel](https://github.com/burning-cost/insurance-multilevel)** extends these ideas into the GBM world. If your portfolio has high-cardinality group factors (hundreds of brokers, schemes, or affinity partners) alongside non-linear individual risk factors, insurance-multilevel runs a two-stage CatBoost + REML approach: CatBoost handles the individual risk factors, REML applies Bühlmann-Straub credibility weighting to the group-level residuals. The credibility_summary() output mirrors the mu, v, a, k notation used here.

---

## What next?

If this library covers your needs for scheme or geographic credibility, that's the end of the road. But if you are working with a portfolio where group factors (brokers, schemes, affinity partners) sit alongside genuinely non-linear individual risk factors - age bands, vehicle type, claims history - you will hit the limits of a purely credibility-based approach. The individual risk factors belong in a GBM; the group factors need shrinkage.

[insurance-multilevel](https://github.com/burning-cost/insurance-multilevel) is built for exactly that problem. It runs CatBoost on individual risk factors in Stage 1, then applies REML random effects (the same credibility weighting logic as here) to the group-level residuals in Stage 2. The two libraries share the same structural parameter notation (mu, v, a, k) and are designed to be used together.

---

## Source repos

This package consolidates two previously separate libraries:

- `credibility` — original Bühlmann-Straub implementation (v0.1.x)
- `insurance-credibility` — archived, hierarchical extension merged into this package

---

## References

1. Bühlmann, H. (1967). Experience rating and credibility. *ASTIN Bulletin*, 4(3), 199-207.
2. Bühlmann, H. & Straub, E. (1970). Glaubwürdigkeit für Schadensätze. *Mitteilungen VSVM*, 70, 111-133.
3. Jewell, W.S. (1975). Regularity conditions for exact credibility. *ASTIN Bulletin*, 8(3), 336-341.
4. Hachemeister, C.A. (1975). Credibility for regression models with application to trend. In *Credibility: Theory and Applications*, Academic Press.
5. Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and its Applications*. Springer.
6. Ohlsson, E. (2008). Combining generalized linear models and credibility models in practice. *Scandinavian Actuarial Journal*, 2008(4), 301-314.
7. Dutang, C., Goulet, V. & Pigeon, M. actuar: Actuarial functions and heavy-tailed distributions. R package, CRAN.

---

---

## Performance

Benchmarked on synthetic Poisson frequency data with known ground truth — 20 segments (5 thin with ~250 total policies, 15 thick with ~25,000 total policies), 5 accident years each. MSE evaluated against true DGP segment frequencies. Full notebook: `notebooks/benchmark.py`.

| Subset | Grand mean | Raw mean | Credibility (Bühlmann-Straub) |
|--------|-----------|---------|-------------------------------|
| Thin segments (n=5) | moderate MSE | highest MSE | lower than raw |
| Thick segments (n=15) | highest MSE | lowest MSE | matches raw |
| All segments (n=20) | — | — | lowest overall |

The benchmark estimates structural parameters from the data (mu, v, a, k) and shows Bühlmann's k automatically setting near-zero credibility weights (Z ≈ 0.10–0.20) for thin segments and near-full weights (Z ≈ 0.99) for thick ones — without any manual specification of which segments are reliable.

On thin segments the raw mean has the highest variance; on thick segments the grand mean has the highest bias. Credibility is the Bayes-optimal blend under the random-effects model, so it beats both extremes in MSE whenever the portfolio has genuine between-segment heterogeneity and mixed exposure depths.

**When to use:** Commercial fleet schemes, affinity groups, rating territories, or occupation classes where exposure per segment varies substantially — some segments are data-rich, others barely have enough claims to compute a stable rate. The closed-form estimator is milliseconds on any panel dataset a pricing team would encounter.

**When NOT to use:** When you have fewer than five or six groups (structural parameters become unstable), when group membership is itself a non-linear rating factor that interacts with individual risk characteristics (use a GLM or GBM instead), or when `a_hat` estimates as zero — which the model will tell you explicitly, meaning the data show no detectable between-group heterogeneity.


## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [rate-optimiser](https://github.com/burning-cost/rate-optimiser) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

**Spatial**

| Library | Description |
|---------|-------------|
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial territory ratemaking for UK personal lines |

[All libraries](https://burning-cost.github.io)

---

## Licence

MIT. See LICENSE.
