# Tutorial — Parameter Interpretability

This tutorial explains how to use the **Cellfoundry Interpretability Module** (`optimizer/analyze.py`)
to understand *why* a parameter combination works and *which parameters* matter most for each
biological objective.

After running an optimization study with `optimizer/optimize.py`, you will have a database of
hundreds (or thousands) of parameter trials, each with one or more objective (error) values.
The interpretability module turns this raw data into a layered HTML report that answers questions like:

- Which parameters drive cell behaviour most?
- Do any parameters interact non-linearly?
- Are the biological objectives in conflict with each other?
- What are the distinct mechanistic "regimes" within the Pareto front (the set of best compromise solutions in a multi-objective optimisation problem)?
- Where in parameter space do the best solutions live?

> **Prerequisites** — a completed optimization study (`.db` file) and the Python packages below.
> Install with:
> ```
> conda activate flamegpu_py310
> pip install scikit-learn scipy matplotlib plotly seaborn pyyaml
> ```

---

## 1. Quick start

```bash
# Auto-detect the study (works if the DB contains exactly one study):
python -m optimizer.analyze --storage sqlite:///cellfoundry_radial_glia.db

# Explicit study name and output directory:
python -m optimizer.analyze \
    --storage sqlite:///cellfoundry_radial_glia.db \
    --study   cellfoundry_radial_glia \
    --out-dir optimizer/analysis_results/radial_glia/

# Named objectives (override or supplement YAML auto-detection):
python -m optimizer.analyze \
    --storage sqlite:///cellfoundry_radial_glia.db \
    --objective-names "N Large Clusters,Rosette Maturity,Compactness,RG Fraction"

# Multi-study DB — generate one report per study:
python -m optimizer.analyze \
    --storage sqlite:///cellfoundry_cell_speed.db \
    --all-studies

# List available studies before generating reports:
python -m optimizer.analyze \
    --storage sqlite:///cellfoundry_cell_speed.db \
    --list-studies
```

The module writes three files to `--out-dir`:

| File | Description |
|---|---|
| `analysis_<study>_<timestamp>.html` | Self-contained HTML report (all figures embedded) |
| `pareto_front.csv` | Pareto-optimal trial data for further analysis |
| `analysis_summary.json` | Machine-readable top findings (top parameters, best values, …) |

---

## 2. Project layout

```
optimizer/
├── analyze.py                      # This module
├── optuna_config_radial_glia.yaml  # YAML config (source of objective names)
├── reference_data/
│   └── target_rg_n_large_clusters.csv
└── analysis_results/
    └── cellfoundry_radial_glia/
        ├── analysis_cellfoundry_radial_glia_20250601_120000.html
        ├── pareto_front.csv
        └── analysis_summary.json
```

**Objective name auto-detection:** if a YAML config file exists whose name matches the study
(e.g., `optuna_config_radial_glia.yaml` for study `cellfoundry_radial_glia`), the module reads
the `objectives[*].kwargs.metric` fields as human-readable objective names.  You can always
override them with `--objective-names`.

---

## 3. Programmatic use

```python
from optimizer.analyze import run_analysis

report_path = run_analysis(
    storage="sqlite:///cellfoundry_radial_glia.db",
    study_name="cellfoundry_radial_glia",
    objective_names=["N Large Clusters", "Rosette Maturity", "Compactness", "RG Fraction"],
    out_dir="optimizer/analysis_results/radial_glia",
    n_top_params=8,            # number of parameters used in pairwise and slice plots
)
print(f"Report: {report_path.resolve()}")
```

---

## 4. Understanding the report — layer by layer

The HTML report is divided into eight layers of increasing complexity, each addressing a different
facet of the optimization results.  **All objective values in Cellfoundry are error metrics —
lower values = better match to the biological target.**

### Layer 0 — Surrogate model importance

**Method:** A [Gradient Boosting Regressor](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
(GBR) is fitted per objective using all completed trials as training data.
[Permutation importance](https://scikit-learn.org/stable/inspection.html#permutation-importance) then
randomly shuffles each parameter one at a time and measures how much the surrogate's predictive accuracy
decreases.

Raw feature importances from tree-based models indicate which variables were frequently or effectively used to split the data, but they are model-specific and can sometimes be misleading, especially when parameters are correlated or have different value ranges. In contrast, permutation importance directly measures how much the model’s predictive performance worsens when the values of one parameter are randomly shuffled. Therefore, it answers a more practical question: *how much does the trained model actually rely on this parameter to make accurate predictions?*

This is particularly useful when some parameters contain overlapping information. For example, if two parameters are strongly correlated, the model may rely mainly on one of them while treating the other as redundant. In that case, shuffling the truly useful parameter will strongly degrade model performance, while shuffling the redundant one may have little effect because the model can still recover similar information from the correlated parameter. However, permutation importance should still be interpreted carefully: when two correlated parameters are both used interchangeably by the model, their individual importance values may be underestimated. For this reason, permutation importance should be understood as a measure of model reliance, rather than definitive causal importance.

### Simple example

Imagine a model trained to predict tumour organoid growth from several simulation parameters. Two of these parameters are:

- `A`: oxygen diffusion coefficient in the extracellular matrix
- `B`: average oxygen concentration inside the organoid

These two parameters are correlated: if oxygen diffuses more efficiently through the matrix, the average oxygen concentration inside the organoid will often be higher.

Suppose the model learns that organoid growth is mainly determined by the average internal oxygen concentration, `B`, because this is the quantity most directly related to cell survival and proliferation.

If we compute permutation importance:

- When `B` is shuffled, the relationship between oxygen availability and growth is disrupted. The model’s predictions become much worse, so `B` receives high permutation importance.
- When `A` is shuffled, the model may still predict growth reasonably well because `B` still contains the oxygen information it needs. Therefore, `A` receives lower permutation importance.

In this example, `A` is biologically relevant, but `B` is the parameter the trained model relies on more directly. Permutation importance helps reveal this distinction.

**The cross-validated R² indicator:**
The cross-validated R² (5-fold by default) tells you how trustworthy the surrogate is:

| R² | Interpretation |
|---|---|
| > 0.7 | Strong surrogate — importance scores are reliable |
| 0.3–0.7 | Moderate surrogate — treat scores as indicative |
| 0–0.3 | Weak surrogate — scores should be compared to fANOVA (Layer 1) |
| < 0 | Surrogate performs *worse* than predicting the mean — this objective's landscape is too noisy, non-linear, or sparse for reliable importance estimation; prefer fANOVA |

**When R² < 0:** this is expected with fewer than ~50 trials, with objectives that are nearly flat
across the search space, or when the sampler has not yet explored the landscape sufficiently.
Continue running the optimizer and re-generate the report.

---

### Layer 1 — Global parameter importance (fANOVA / MDI)

Two complementary importance metrics are computed for each objective across *all* completed trials.

#### fANOVA (functional ANOVA)

fANOVA fits a random forest to the trial data and uses it to decompose the total variance of the
objective into additive contributions from individual parameters and their pairwise interactions.
The result is a fraction between 0 and 1: a parameter with fANOVA importance 0.4 "explains"
40% of the objective's variance across the explored search space.

**Key properties:**
- Accounts for the structure of the search space (log-uniform, categorical, …).
- Unbiased with respect to the number of distinct values a parameter takes.
- Can detect interaction effects (though this report shows only marginal importances).
- Requires at least ~10 completed trials per objective; improves substantially with 50+.

#### MDI (Mean Decrease Impurity)

MDI measures how often each parameter is chosen for a split in the random forest, weighted by
the impurity reduction at that split, then normalised across all trees.

**Caution — high-cardinality bias:** MDI can be inflated for parameters that happen to take many
distinct values (e.g., a continuous parameter sampled log-uniformly over 4 orders of magnitude).
A parameter with many possible split points will appear more important than it truly is, even if
its actual effect on the objective is weak.  Always compare MDI against fANOVA to identify this.

**When to trust each method:**
- fANOVA and MDI *agree* → strong signal.
- fANOVA low, MDI high → likely MDI bias; trust fANOVA.
- fANOVA high, MDI low → unusual; may indicate non-linear effects poorly captured by impurity.

---

### Layer 2 — Pareto-front importance

The same fANOVA/MDI analysis is repeated on *only* the Pareto-optimal trials.

Global importance (Layer 1) is often dominated by the *feasibility boundary* — the transition
between "simulation produces no output" and "simulation produces output".  For example, in a
radial-glia study, a parameter that controls whether any RG cells form at all will score very
high globally, because most of the objective variance comes from the trivial infeasible region.
This is not the most useful signal for experimental design.

The Pareto-only analysis looks *within* the feasible region and identifies which parameters
separate good solutions (low error) from merely acceptable ones — the more actionable question.

The **parallel coordinates plot** shows all Pareto-optimal trials simultaneously.  Lines that
cluster together across multiple axes indicate parameter bundles that consistently produce
low-error outcomes.

---

### Layer 3 — Pairwise interaction plots

For each pair of top-ranked parameters, a 2-D scatter plot coloured by objective value
(red = high error, green = low error) is generated.  Each point is one trial.

#### Iso-contour overlays

The background **iso-contour lines** are computed using Delaunay triangulation of the trial data
— the triangulation connects neighbouring data points to form a mesh, and contour lines are then
traced at constant objective values through this mesh.  The result is an approximate
"objective landscape" map in two-parameter space.

**Interpretation:**
- **Curved iso-contours** → suggest coupling, nonlinear response, or parameter interaction. E.g. The optimal value of parameter A depends on the value of parameter B. These parameters interact and cannot be optimised independently.  An experimental design that
  varies only one at a time will miss the true optimum.
- **Parallel straight iso-contours** → the two parameters act roughly independently (additive effects).
- **Tightly clustered ★ Pareto markers** → suggest that the optimal joint range is well-localised in this two-parameter projection. Confidence in that region is stronger when the cluster is supported by many nearby trials, not only by the contour shape itself.

#### Spearman correlation matrix

The Spearman ρ matrix at the top of this section shows whether the optimizer tended to
sample parameters together.  This reflects the sampling history, not necessarily a causal
relationship. If the correlation is computed over all trials, a positive ρ means that high values of one parameter were often sampled together with high values of the other, or low values with low values. If the correlation is computed only over the best or Pareto trials, then a positive ρ suggests that both parameters tend to be jointly high, or jointly low, in high-performing regions.

---

### Layer 4 — Objective correlation & conflict

#### Spearman ρ between objectives

Spearman ρ measures how monotonically the objectives move together across all trials.

| ρ | Interpretation |
|---|---|
| > 0.5 | Aligned — optimising one tends to improve the other |
| −0.5 to 0.5 | Weak or no relationship |
| < −0.5 | Conflict — improving one worsens the other (true trade-off) |

A statistically significant (★ p<0.05) negative correlation means you cannot simultaneously
minimise both objectives.  Your experimental validation needs to decide which objective to
prioritise, or accept a Pareto-optimal compromise.

#### PCA of the Pareto front in objective space

In a multi-objective optimisation problem, the **Pareto front** is the set of best compromise solutions. A solution belongs to the Pareto front if no other solution improves one objective without worsening at least one other objective.

Principal Component Analysis of the Pareto front decomposes these best compromise solutions into orthogonal axes of variation in objective space. Each principal component (PC) represents a direction along which the Pareto-optimal solutions differ from each other.

This helps identify the main structure of the trade-offs between objectives: whether the Pareto front is mainly organised by one dominant compromise, or whether several trade-off directions are needed to describe it.

* **PC1 with high explained variance** means that most of the variability in the Pareto front lies along a single dominant axis, usually the main trade-off between objectives.
* **PC2, PC3, etc.** describe additional, weaker trade-offs that are not captured by PC1.
* **The loading plot** shows which objectives contribute most strongly to each PC. Objectives with large absolute loadings are the main drivers of that component.
* **Objectives with opposite-sign loadings on the same PC** are in tension: moving along that PC tends to improve one objective while worsening the other.
* **Objectives with same-sign loadings on the same PC** tend to co-vary, meaning that they are often improved or worsened together within the Pareto front.

For example, imagine an optimisation problem where an organoid model is calibrated using three objectives:

* `growth_error`: difference between simulated and experimental organoid growth
* `shape_error`: difference between simulated and experimental organoid morphology
* `population_distribution_error`: difference between simulated and experimental spatial distribution of cell populations

If PC1 explains most of the variance and has a strong positive loading for `growth_error` but strong negative loadings for `shape_error` and `population_distribution_error`, this suggests a dominant trade-off between matching growth and matching spatial organisation. Moving in one direction along PC1 gives solutions with lower growth error but higher shape and population-distribution errors. Moving in the opposite direction gives solutions that better reproduce morphology and cell population organisation, but with a worse match in overall growth.

If `shape_error` and `population_distribution_error` have loadings with the same sign, this suggests that these two objectives tend to improve or worsen together within the Pareto front. In this example, parameter sets that reproduce organoid morphology well also tend to reproduce the spatial organisation of cell populations well.

*Note*: since these are formulated as errors, improving an objective means reducing its value.

---

### Layer 5 — Regime / cluster detection

K-means clustering groups the Pareto-optimal trials in scaled parameter space to find
qualitatively different solution *regimes*.

**What is a regime?**
A regime is a cluster of parameter combinations that all achieve good performance (low error)
through a similar mechanism.  Different regimes represent alternative mechanistic hypotheses.
For example:
- Regime A: low cell stiffness compensated by high adhesion density.
- Regime B: high stiffness with spatially organised fibre alignment.

Both regimes achieve equally low error but via different biophysics — and these can be
distinguished experimentally by targeting a parameter that is high in one regime but low in
the other.

#### Selecting the number of clusters k

The [silhouette coefficient](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)
measures how well each trial fits its assigned cluster relative to the nearest other cluster.
It ranges from −1 (misclassified) to +1 (perfectly isolated).  The module tests k from 2 to 6
and selects the k that maximises the silhouette coefficient.

#### Cluster fingerprints

For each cluster, a z-score fingerprint shows which parameters are unusually high or low
relative to the Pareto average.  A parameter with z > 1.5 is a *defining signature* of that
cluster — the mechanism driving performance in that regime.

---

### Layer 6 — 1-D sensitivity slices

For each top-ranked parameter, a marginal sensitivity plot shows how the objective value
typically changes as that parameter varies, holding everything else as observed.

#### LOWESS smoothing explained

LOWESS (Locally Weighted Scatterplot Smoothing) draws a smooth trend line without assuming
any particular functional form:

1. For each x position, identify a neighbourhood of nearby data points.
2. Weight those points by their distance (closer = higher weight).
3. Fit a low-degree polynomial to the weighted points.
4. The fitted value at x is the local trend estimate.

Repeat across all x values to get a smooth curve.  This gives an honest, non-parametric
summary of the data without imposing linear or exponential shapes.

**Interpreting the slope:**
- **Steep slope** → the objective is highly sensitive to this parameter; even small
  changes produce large changes in error.  High experimental leverage.
- **Flat region** → the objective is insensitive here; the parameter is in a plateau
  or saturation zone.  Less leverage, but also more robust.
- **V/U-shape** → there is an optimal range in the middle; too high or too low
  both lead to worse performance.  A clear target value for experimental design.

**Important caveat:** these are *marginal* trends averaged across all trial values of
the other parameters.  Interactions are averaged out.  If two parameters interact
non-linearly, the Layer 3 pairwise plots will reveal the true joint optimum.

#### Reading the y-axis

The y-axis always shows the objective **error value** — lower = better match to
the biological target.  Red ★ Pareto markers appear near the bottom of the plot,
confirming that the best solutions correctly have the lowest error values.
A parameter with Pareto markers clustered in a narrow x-range identifies a
well-constrained "sweet spot."

---

### Layer 7 — Parameter distributions: Pareto vs all trials

Violin plots compare the full search distribution (blue) against the Pareto-optimal
subset (red) for every parameter.

**How to read them:**

| Pattern | Meaning |
|---|---|
| Narrow red, wide blue | Clear "sweet spot" — only a specific range achieves good performance |
| Shifted red median | Directional preference — higher (or lower) values consistently perform better |
| Red and blue overlap completely | This parameter has little influence on Pareto membership |
| Red median near the boundary of blue | The optimizer may be hitting a search boundary; consider expanding the range |

**Parameters displayed on log₁₀ scale** when the search range spans more than two orders
of magnitude — the violin shape directly shows the density of good solutions in log-space.

**Complementarity with Layer 6:**
- Layer 6 answers "how steep is the sensitivity curve?"
- Layer 7 answers "where do the good solutions actually sit?"

Both together provide a complete picture for designing targeted experiments.

---

## 5. Typical workflow

### Step 1 — Run the optimizer

```bash
python -m optimizer.optimize \
    --config optimizer/optuna_config_radial_glia.yaml \
    --n-trials 200
```

### Step 2 — Generate the report

```bash
python -m optimizer.analyze \
    --storage sqlite:///cellfoundry_radial_glia.db \
    --out-dir optimizer/analysis_results/radial_glia/
```

### Step 3 — Read the report

Open the generated `.html` file in any browser.  Work through the layers:

1. **Start at the objectives table** — confirm the best achieved error values.
   Are the errors plausibly low? If not, more trials may be needed.
2. **Check Layer 0 R²** — if all R² are above 0.3, the surrogate is trustworthy.
   If not, rely more on fANOVA (Layer 1).
3. **Compare Layer 1 and Layer 2** — the difference reveals whether the dominant
   parameters in the full space are the same as those that control quality within
   the Pareto front.
4. **Layer 3 pairs** — look for curved iso-contours indicating non-linear
   interactions.  Parameters that interact need to be co-varied in experiments.
5. **Layer 4 conflict check** — identify which pairs of objectives are in tension.
   This shapes which Pareto solution to commit to for wet-lab validation.
6. **Layer 5 regimes** — if 2+ clusters are detected, each cluster is a separate
   experimental hypothesis.  Design one targeted perturbation experiment per regime.
7. **Layers 6 and 7** — identify the highest-leverage parameters for experimental
   manipulation and the precise ranges to test.

### Step 4 — Form hypotheses

Based on the report, write down 2–3 concrete mechanistic hypotheses.  Each hypothesis
should be falsifiable with a targeted parameter perturbation experiment.

**Example:**
> "Cluster 1 fingerprint: high `cell_adhesion_strength` (z=+2.1), low `migration_speed` (z=−1.8).
> Hypothesis: strong adhesion restricts migration and promotes the compact rosette morphology
> we observe in the target data.  Test: reduce `cell_adhesion_strength` in vitro using a
> function-blocking antibody and observe whether the rosette phenotype is lost."

---

## 6. Multi-study databases

Some databases contain multiple studies with different configurations.
Use `--list-studies` to inspect the contents:

```bash
python -m optimizer.analyze \
    --storage sqlite:///cellfoundry_cell_speed.db \
    --list-studies
# Studies in sqlite:///cellfoundry_cell_speed.db:
#   cellfoundry_cell_speed_control
#   cellfoundry_cell_speed_tgfb_chemokinesis
```

Then either target one study:

```bash
python -m optimizer.analyze \
    --storage sqlite:///cellfoundry_cell_speed.db \
    --study cellfoundry_cell_speed_control
```

Or generate reports for all studies at once:

```bash
python -m optimizer.analyze \
    --storage sqlite:///cellfoundry_cell_speed.db \
    --all-studies
# Writes:
#   optimizer/analysis_results/cellfoundry_cell_speed_control/analysis_*.html
#   optimizer/analysis_results/cellfoundry_cell_speed_tgfb_chemokinesis/analysis_*.html
```

---

## 7. CLI reference

```
usage: python -m optimizer.analyze [options]

Required:
  --storage URL         Optuna storage URL (e.g. sqlite:///my_study.db)

Optional:
  --study NAME          Study name; auto-detected if DB contains exactly one study
  --objective-names S   Comma-separated list of objective names (positional)
                        Overrides YAML auto-detection
  --out-dir DIR         Output directory (default: optimizer/analysis_results)
  --n-top-params N      Number of top parameters used in pairwise/slice analyses
                        (default: 8)
  --all-studies         Generate one report per study in the DB
  --list-studies        Print study names and exit
```

---

## 8. Tips and troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Layer 0 shows R² < 0 | Too few trials or flat landscape | Run ≥50–100 more trials; rely on fANOVA (Layer 1) |
| Layer 5 not shown | Fewer than 20 Pareto trials | Continue optimization; Layer 5 needs sufficient Pareto diversity |
| Importance scores near zero for all parameters | Objectives are nearly flat | Check objective function; ensure it is sensitive to parameter changes |
| Layer 3 contours missing | Fewer than 8 trial data points for that pair | Run more trials |
| Parallel coordinates plot is missing | `plotly` not installed | `pip install plotly` |
| Objective names show "Objective 0", "Objective 1" | YAML not found | Check naming convention or use `--objective-names` |
| Study not found error | Wrong `--study` name | Use `--list-studies` to check exact names |

---

## See also

- [Tutorial: Parameter Optimization](Tutorial-Parameter-Optimization) — how to set up and run
  the optimization study that feeds this interpretability report.
- [Tutorial: Parameter Overriding](Tutorial-Parameter-Overriding) — running single simulations
  with specific parameter values identified by the analysis.
