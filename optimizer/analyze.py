"""
Cellfoundry Optimizer — Interpretability & Analysis Module
===========================================================

Generates a comprehensive HTML report explaining *why* a particular combination
of parameters works, organized as layered analyses:

    Layer 0 — Surrogate model importance    (GBR + permutation importance)
    Layer 1 — Global parameter importance   (fANOVA / MDI)
    Layer 2 — Pareto-front importance       (restricted to best trials)
    Layer 3 — Pairwise interaction plots    (scatter + iso-contours)
    Layer 4 — Objective correlation         (Spearman + PCA of Pareto front)
    Layer 5 — Regime / cluster detection    (K-means in parameter space)
    Layer 6 — 1-D sensitivity slices        (LOWESS curves from trial data)
    Layer 7 — Parameter distributions       (Pareto vs all-trials violin plots)

Outputs
-------
    <out_dir>/analysis_<study>_<timestamp>.html  — self-contained report
    <out_dir>/pareto_front.csv                   — Pareto-optimal trial data
    <out_dir>/analysis_summary.json              — machine-readable top findings

Usage
-----
    # Single study (auto-detected if only one study in DB):
    python -m optimizer.analyze \\
        --storage sqlite:///cellfoundry_radial_glia.db

    # Explicit study name and objective labels:
    python -m optimizer.analyze \\
        --storage sqlite:///cellfoundry_radial_glia.db \\
        --study   cellfoundry_radial_glia \\
        --objective-names "n_large_clusters,maturity,compactness,rg_fraction" \\
        --out-dir optimizer/analysis_results/

    # Generate one report per study in the DB:
    python -m optimizer.analyze \\
        --storage sqlite:///cellfoundry_cell_speed.db \\
        --all-studies

    # List available studies in a DB:
    python -m optimizer.analyze \\
        --storage sqlite:///cellfoundry_cell_speed.db \\
        --list-studies

    # Programmatic use:
    from optimizer.analyze import run_analysis
    run_analysis("sqlite:///cellfoundry_radial_glia.db", "cellfoundry_radial_glia")
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import io
import re
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d

# Suppress the KMeans/MKL memory-leak warning on Windows — it is cosmetic only.
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak",
    category=UserWarning,
    module="sklearn",
)

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

import optuna
from optuna.importance import FanovaImportanceEvaluator

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import plotly.graph_objects as go
    from plotly.io import to_html as plotly_to_html
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import yaml as _yaml_mod
    _YAML = True
except ImportError:
    _YAML = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PARETO_COLOR = "#e74c3c"
_ALL_COLOR = "#3498db"
_FIGURE_DPI = 120
_MAX_CONTOUR_PAIRS = 6
_MAX_SLICE_PARAMS = 15
_KMEANS_MAX_K = 6
_KMEANS_MIN_TRIALS = 20


# ===========================================================================
# YAML / objective metadata helpers
# ===========================================================================

def _find_yaml_for_study(db_path: str, study_name: str) -> Path | None:
    """Try to locate the optuna_config_*.yaml matching the study name."""
    suffix = re.sub(r"^cellfoundry_", "", study_name)
    candidates = [
        f"optuna_config_{suffix}.yaml",
        f"optuna_config_{study_name}.yaml",
    ]
    search_dirs: list[Path] = [
        Path.cwd() / "optimizer",
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]
    if db_path.startswith("sqlite:///"):
        db_file = Path(db_path.replace("sqlite:///", ""))
        if not db_file.is_absolute():
            db_file = Path.cwd() / db_file
        search_dirs.insert(0, db_file.parent / "optimizer")
        search_dirs.insert(0, db_file.parent)

    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.exists():
                return p

    # Progressive suffix shortening: try removing trailing _word segments one at a time
    # e.g. "organoid_growth" -> "organoid", "diff_modulus_high_res" -> "diff_modulus_high" -> "diff_modulus"
    suffix_parts = suffix.split("_")
    for k in range(len(suffix_parts) - 1, 0, -1):
        shorter = "_".join(suffix_parts[:k])
        for d in search_dirs:
            p = d / f"optuna_config_{shorter}.yaml"
            if p.exists():
                return p
    return None


def _parse_yaml_objectives(yaml_path: Path) -> list[dict]:
    """Return a list of objective metadata dicts from an optuna YAML config."""
    if not _YAML:
        return []
    try:
        with open(yaml_path, encoding="utf-8") as f:
            config = _yaml_mod.safe_load(f)
    except Exception:
        return []
    objectives = config.get("objectives") or config.get("objective")
    if objectives is None:
        objectives = []
    if not isinstance(objectives, list):
        objectives = [objectives]
    result = []
    for obj in objectives:
        kwargs = obj.get("kwargs", {}) or {}
        metric = kwargs.get("metric", "")
        func = obj.get("function", "")
        raw_name = metric if metric else func
        name_display = raw_name.replace("_", " ").title()
        # Resolve target value: explicit target_value or target_metric in kwargs
        target_val = kwargs.get("target_value") if kwargs.get("target_value") is not None \
            else kwargs.get("target_metric")
        # Resolve reference CSV path
        ref_str = obj.get("reference")
        resolved_ref: Path | None = None
        if ref_str:
            ref_path = Path(ref_str)
            for base in [yaml_path.parent, yaml_path.parent.parent, Path.cwd()]:
                candidate = base / ref_path
                if candidate.exists():
                    resolved_ref = candidate
                    break
        result.append({
            "name": name_display,
            "raw_name": raw_name,
            "function": func,
            "reference": ref_str,
            "resolved_reference": resolved_ref,
            "target_value": target_val,
        })
    return result


def _read_csv_target(resolved_ref: "Path | None") -> str | None:
    """Try to read a meaningful target description from a reference CSV file."""
    if resolved_ref is None or not resolved_ref.exists():
        return None
    try:
        df_ref = pd.read_csv(resolved_ref)
    except Exception:
        return None
    if df_ref.empty:
        return None
    # Drop any column that looks like a time/index column to find the value column(s)
    value_cols = [c for c in df_ref.columns if c.lower() not in ("time", "index", "t", "step")]
    n_rows = len(df_ref)
    has_time = any(c.lower() in ("time", "t", "step") for c in df_ref.columns)
    if n_rows == 1 and value_cols:
        # Single-row: show value
        vals = df_ref[value_cols].iloc[0].values
        if len(vals) == 1:
            v = vals[0]
            return f"{v:.4g} <em style='color:#888;font-size:0.85em'>*from CSV</em>"
        else:
            v_str = ", ".join(f"{v:.4g}" for v in vals[:3])
            return f"[{v_str}] <em style='color:#888;font-size:0.85em'>*from CSV</em>"
    else:
        # Multi-row: time series or value array
        if has_time and value_cols:
            return (f"<em style='color:#888'>time series from CSV "
                    f"({n_rows}&nbsp;points)</em>")
        else:
            return (f"<em style='color:#888'>{n_rows} values from CSV</em>")


def _load_icon_b64() -> str:
    icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon_cellfoundry.png"
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def _build_objectives_block(
    obj_names: list[str],
    yaml_objectives: list[dict],
    best_values: list[float],
    directions: list[str],
) -> str:
    rows = []
    for i, name in enumerate(obj_names):
        meta = yaml_objectives[i] if i < len(yaml_objectives) else {}
        func = meta.get("function") or "—"
        ref = meta.get("reference") or "—"
        target = meta.get("target_value")
        if target is not None:
            target_str = f"{target:.4g}"
        else:
            csv_str = _read_csv_target(meta.get("resolved_reference"))
            target_str = csv_str if csv_str else ("see reference CSV" if ref != "—" else "—")
        best = best_values[i] if i < len(best_values) else float("nan")
        best_str = f"{best:.5g}" if np.isfinite(best) else "—"
        direction = directions[i] if i < len(directions) else "minimize"
        rows.append(
            f"<tr>"
            f"<td><strong>{i}</strong></td>"
            f"<td><strong>{name}</strong></td>"
            f"<td><code>{func}</code></td>"
            f"<td>{target_str}</td>"
            f"<td><strong style='color:#c0392b'>{best_str}</strong></td>"
            f"<td>{direction}</td>"
            f"</tr>"
        )
    return (
        "<h3>Objectives at a glance</h3>"
        "<p>All values in this report are <em>error metrics</em> — "
        "<strong>lower = better match to the biological target</strong>.</p>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse;font-size:0.92em'>"
        "<tr><th>#</th><th>Name</th><th>Function</th>"
        "<th>Target value</th><th>Best achieved</th><th>Direction</th></tr>"
        + "\n".join(rows) + "</table>"
    )


# ===========================================================================
# Data loading
# ===========================================================================

def list_study_names(storage: str) -> list[str]:
    return optuna.get_all_study_names(storage=storage)


def _load_study(storage: str, study_name: str) -> optuna.Study:
    return optuna.load_study(study_name=study_name, storage=storage)


def _trials_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        raise RuntimeError("No COMPLETE trials found in the study.")

    all_params = sorted({k for t in trials for k in t.params})
    rows = []
    for t in trials:
        row: dict = {"trial_number": t.number}
        for p in all_params:
            row[f"param_{p}"] = t.params.get(p, np.nan)
        for i, v in enumerate(t.values):
            row[f"value_{i}"] = v if v is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    n_obj = len(trials[0].values) if trials[0].values else 0
    pareto_numbers: set[int] = set()
    if n_obj > 1:
        try:
            pareto_numbers = {t.number for t in study.best_trials}
        except Exception:
            pass
    elif n_obj == 1:
        best_val = df["value_0"].min()
        pareto_numbers = set(df.loc[df["value_0"] == best_val, "trial_number"])

    df["is_pareto"] = df["trial_number"].isin(pareto_numbers)
    return df


def _param_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("param_")]


def _value_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("value_")]


def _clean_param_name(col: str) -> str:
    return col.replace("param_", "")


def _objective_label(col: str, names: list[str]) -> str:
    idx = int(col.replace("value_", ""))
    return names[idx] if idx < len(names) else f"Objective {idx}"


# ===========================================================================
# Figure / HTML helpers
# ===========================================================================

def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_FIGURE_DPI, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _img_html(b64: str, caption: str = "") -> str:
    if not b64:
        return "<p><em>Figure could not be generated.</em></p>"
    return (
        f'<figure style="margin:1em 0">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:900px;width:100%;height:auto" alt="{caption}"/>'
        f'{"<figcaption>" + caption + "</figcaption>" if caption else ""}'
        f"</figure>"
    )


def _conclude_box(text: str) -> str:
    return (
        '<div style="background:#eaf4fb;border-left:4px solid #2980b9;'
        'padding:0.7em 1.2em;margin:1em 0;border-radius:0 4px 4px 0">'
        f'<strong>&#128269; Key findings:</strong> {text}</div>'
    )


def _warn_box(text: str) -> str:
    return (
        '<div style="background:#fef9e7;border-left:4px solid #f39c12;'
        'padding:0.7em 1.2em;margin:1em 0;border-radius:0 4px 4px 0">'
        f'<strong>&#9888; Note:</strong> {text}</div>'
    )


def _plotly_div(fig) -> str:
    if not _PLOTLY:
        return "<p><em>plotly not available.</em></p>"
    return plotly_to_html(fig, full_html=False, include_plotlyjs="cdn")


# ===========================================================================
# Layer 0 — Surrogate model (GBR + permutation importance)
# ===========================================================================

def _layer_rf_permutation(df: pd.DataFrame, obj_names: list[str]) -> tuple[list[str], str]:
    html_parts: list[str] = []
    param_cols = _param_columns(df)
    param_labels = [_clean_param_name(c) for c in param_cols]
    value_cols = _value_columns(df)
    n_trials = len(df)

    MIN_TRIALS = 15
    if n_trials < MIN_TRIALS:
        return [], _warn_box(
            f"Only {n_trials} completed trials are available (minimum required: {MIN_TRIALS}). "
            f"A Gradient Boosting surrogate cannot be reliably fitted with so few samples — "
            f"the model would likely overfit and produce meaningless importance scores. "
            f"Run more trials (aim for at least 50–100) to unlock this layer. "
            f"fANOVA in Layer 1 may still provide useful estimates."
        )

    all_importances: dict[str, np.ndarray] = {}
    top_params_by_obj: dict[str, list[str]] = {}
    r2_scores: dict[str, float] = {}
    r2_warnings: list[str] = []

    for i, v_col in enumerate(value_cols):
        obj_label = _objective_label(v_col, obj_names)
        sub = df[param_cols + [v_col]].dropna()
        if len(sub) < MIN_TRIALS:
            r2_scores[obj_label] = float("nan")
            continue
        X = sub[param_cols].values.astype(float)
        y = sub[v_col].values.astype(float)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[valid], y[valid]
        if len(y) < MIN_TRIALS:
            r2_scores[obj_label] = float("nan")
            continue

        gbr = GradientBoostingRegressor(n_estimators=200, random_state=42, subsample=0.8)
        try:
            n_cv = min(5, max(2, len(y) // 10))
            cv_r2 = cross_val_score(gbr, X, y, cv=n_cv, scoring="r2")
            mean_r2 = float(np.mean(cv_r2))
        except Exception:
            mean_r2 = float("nan")
        r2_scores[obj_label] = mean_r2

        if np.isfinite(mean_r2) and mean_r2 < 0:
            r2_warnings.append(
                f"<strong>{obj_label}</strong> has a negative cross-validated R² "
                f"({mean_r2:.3f}). "
                f"This means the surrogate fits <em>worse</em> than simply predicting the mean value. "
                f"Possible causes: fewer than ~50 trials, a noisy or near-flat objective landscape, "
                f"or strong non-linearities the sampler has not yet explored. "
                f"Importance scores for this objective are unreliable — prioritise fANOVA (Layer 1) instead."
            )

        gbr.fit(X, y)
        perm = permutation_importance(gbr, X, y, n_repeats=20, random_state=42, n_jobs=-1)
        importance = perm.importances_mean
        importance_std = perm.importances_std
        all_importances[obj_label] = importance

        top_idx = np.argsort(importance)[::-1]
        top_params_by_obj[obj_label] = [param_labels[j] for j in top_idx[:5]]

        # Sort ascending → most important ends up at TOP of barh
        sorted_idx = np.argsort(importance)
        sorted_labels = [param_labels[j] for j in sorted_idx]
        sorted_imp = importance[sorted_idx]
        sorted_std = importance_std[sorted_idx]
        max_abs = max(float(np.max(np.abs(importance))), 1e-12)
        colors = [
            "#e74c3c" if v >= max_abs * 0.4 else
            "#e67e22" if v >= max_abs * 0.15 else
            "#95a5a6"
            for v in sorted_imp
        ]

        r2_color = "#27ae60" if mean_r2 > 0.3 else ("#e67e22" if mean_r2 >= 0 else "#e74c3c")
        r2_txt = f"{mean_r2:.3f}" if np.isfinite(mean_r2) else "N/A"

        fig, ax = plt.subplots(figsize=(9, max(3.5, len(param_labels) * 0.42)))
        ax.barh(sorted_labels, sorted_imp, xerr=sorted_std, color=colors,
                capsize=3, error_kw=dict(lw=1))
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Mean decrease in accuracy when parameter is shuffled (higher = more important)")
        ax.set_title(f"Permutation importance — {obj_label}", fontsize=10)
        fig.text(0.5, 0.97, f"Surrogate cross-val R\u00b2 = {r2_txt}",
                 ha="center", va="top", fontsize=9, color=r2_color,
                 transform=fig.transFigure)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        html_parts.append(_img_html(_fig_to_base64(fig)))

    # Combined chart
    if len(all_importances) > 1:
        combined = np.zeros(len(param_labels))
        for imp in all_importances.values():
            norm = imp / (float(np.max(np.abs(imp))) + 1e-30)
            combined += norm
        combined /= len(all_importances)
        sorted_idx = np.argsort(combined)
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(param_labels) * 0.42)))
        colors_c = plt.cm.Blues(np.linspace(0.3, 0.85, len(param_labels)))
        ax.barh([param_labels[j] for j in sorted_idx], combined[sorted_idx], color=colors_c)
        ax.set_xlabel("Normalised importance (averaged over all objectives)")
        ax.set_title("Combined importance — averaged across all objectives", fontsize=10)
        plt.tight_layout()
        html_parts.append(_img_html(_fig_to_base64(fig)))

    for w in r2_warnings:
        html_parts.append(_warn_box(w))

    # Auto-conclusion
    if top_params_by_obj:
        parts = []
        for obj, top in top_params_by_obj.items():
            r2 = r2_scores.get(obj, float("nan"))
            r2_note = f" (R\u00b2={r2:.2f})" if np.isfinite(r2) else ""
            parts.append(
                f"For <strong>{obj}</strong>{r2_note}: top driver is <code>{top[0]}</code>"
                + (f", followed by <code>{top[1]}</code>." if len(top) > 1 else ".")
            )
        html_parts.append(_conclude_box(" &nbsp;|&nbsp; ".join(parts)))

    # Global top params
    global_top: list[str] = []
    seen: set[str] = set()
    for top in top_params_by_obj.values():
        for p in top:
            if p not in seen:
                global_top.append(p)
                seen.add(p)

    return global_top, "\n".join(html_parts)


# ===========================================================================
# Layer 1 — Global parameter importance (fANOVA / MDI)
# ===========================================================================

def _layer1_importance(
    study: optuna.Study,
    df: pd.DataFrame,
    obj_names: list[str],
    *,
    pareto_only: bool = False,
) -> tuple[dict, str]:
    n_obj = len(_value_columns(df))
    param_cols = _param_columns(df)
    param_names = [_clean_param_name(c) for c in param_cols]
    subset_df = df[df["is_pareto"]].copy() if pareto_only else df.copy()
    subset_df = subset_df.dropna(subset=param_cols + _value_columns(df))

    MIN_TRIALS = 10
    if len(subset_df) < MIN_TRIALS:
        label = "Pareto-only" if pareto_only else "all trials"
        return {}, _warn_box(
            f"Only {len(subset_df)} trials available for {label} importance analysis "
            f"(minimum: {MIN_TRIALS}). Collect more trials to enable this layer."
        )

    fanova_results: dict[str, dict] = {}
    mdi_results: dict[str, dict] = {}

    for i in range(n_obj):
        v_col = f"value_{i}"
        obj_name = _objective_label(v_col, obj_names)
        y = subset_df[v_col].values
        X = subset_df[param_cols].values
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[valid], y[valid]
        if len(y) < MIN_TRIALS:
            continue

        try:
            evaluator = FanovaImportanceEvaluator(seed=42)
            if n_obj > 1:
                imp = optuna.importance.get_param_importances(
                    study, evaluator=evaluator,
                    target=lambda t, _i=i: t.values[_i],
                )
            else:
                imp = optuna.importance.get_param_importances(study, evaluator=evaluator)
            fanova_results[obj_name] = dict(imp)
        except Exception as exc:
            warnings.warn(f"fANOVA failed for {obj_name}: {exc}")
            fanova_results[obj_name] = {}

        try:
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            mdi_results[obj_name] = dict(zip(param_names, rf.feature_importances_))
        except Exception as exc:
            warnings.warn(f"MDI failed for {obj_name}: {exc}")
            mdi_results[obj_name] = {}

    objs_with_data = [o for o in fanova_results if fanova_results[o]]
    if not objs_with_data:
        return {}, "<p>fANOVA produced no results.</p>"

    label = "Pareto-only" if pareto_only else "All trials"
    n_plots = len(objs_with_data)
    fig, axes = plt.subplots(n_plots, 2, figsize=(11, max(4, 3.5 * n_plots)), squeeze=False)
    fig.suptitle(f"Parameter Importance — {label}", fontsize=14, fontweight="bold")

    for row_i, obj_name in enumerate(objs_with_data):
        for col_i, (results_dict, method_name) in enumerate(
            [(fanova_results, "fANOVA"), (mdi_results, "MDI (Random Forest)")]
        ):
            ax = axes[row_i][col_i]
            data = results_dict.get(obj_name, {})
            if not data:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{obj_name} — {method_name}")
                continue

            # Sort ASCENDING so the most important parameter ends up at the TOP of the chart
            sorted_items = sorted(data.items(), key=lambda x: x[1])
            params_asc = [k for k, _ in sorted_items]
            values_asc = [v for _, v in sorted_items]
            max_v = max(values_asc) if values_asc else 1.0
            colors = [
                "#e74c3c" if v >= max_v * 0.6 else
                "#e67e22" if v >= max_v * 0.3 else
                "#95a5a6"
                for v in values_asc
            ]
            bars = ax.barh(params_asc, values_asc, color=colors)
            ax.set_xlabel("Importance score")
            ax.set_title(f"{obj_name} — {method_name}", fontsize=10)
            ax.set_xlim(0, max_v * 1.18)
            # bars[i] corresponds to params_asc[i] and values_asc[i] — no reversal needed
            for bar, val in zip(bars, values_asc):
                ax.text(
                    val + max_v * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8,
                )

    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    html = _img_html(b64, f"Parameter Importance ({label}) — most important at top, least at bottom")

    # Auto-conclusion
    lines = []
    for obj_name in objs_with_data:
        imp = fanova_results.get(obj_name, {})
        if not imp:
            continue
        top2 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:2]
        mdi = mdi_results.get(obj_name, {})
        mdi_top = sorted(mdi.items(), key=lambda x: x[1], reverse=True)[:1] if mdi else []
        agree = mdi_top and mdi_top[0][0] == top2[0][0]
        agree_txt = (" Both fANOVA and MDI agree on this top parameter." if agree
                     else (f" MDI ranks <code>{mdi_top[0][0]}</code> first — possible high-cardinality bias."
                           if mdi_top else ""))
        lines.append(
            f"<strong>{obj_name}</strong>: <code>{top2[0][0]}</code> accounts for "
            f"{top2[0][1]:.1%} of objective variance"
            + (f"; <code>{top2[1][0]}</code> is second ({top2[1][1]:.1%})." if len(top2) > 1 else ".")
            + agree_txt
        )
    conclusion_html = _conclude_box(" &nbsp;|&nbsp; ".join(lines)) if lines else ""
    return fanova_results, html + "\n" + conclusion_html


# ===========================================================================
# Layer 2 — Pareto-front importance + parallel coordinates
# ===========================================================================

def _layer2_pareto(
    study: optuna.Study,
    df: pd.DataFrame,
    obj_names: list[str],
    fanova_global: dict,
) -> str:
    html_parts: list[str] = []
    pareto_df = df[df["is_pareto"]].copy()
    n_pareto, n_total = len(pareto_df), len(df)
    param_cols = _param_columns(df)
    value_cols = _value_columns(df)

    html_parts.append(
        f"<p><strong>Pareto-optimal trials:</strong> {n_pareto} / {n_total} "
        f"({100*n_pareto/max(n_total,1):.1f}%)</p>"
    )

    if n_pareto < 5:
        html_parts.append(_warn_box(
            f"Only {n_pareto} Pareto-optimal trials found (minimum needed: 5). "
            f"Continue running the optimizer to populate the Pareto front."
        ))
        return "\n".join(html_parts)

    _, pareto_imp_html = _layer1_importance(study, df, obj_names, pareto_only=True)
    html_parts.append(pareto_imp_html)

    # Parallel coordinates
    if _PLOTLY and n_pareto >= 3:
        try:
            display_cols = param_cols + value_cols
            plot_df = pareto_df[["trial_number"] + display_cols].copy()
            plot_df.columns = (
                ["Trial"]
                + [_clean_param_name(c) for c in param_cols]
                + [_objective_label(c, obj_names) for c in value_cols]
            )
            first_obj_col = _objective_label(value_cols[0], obj_names) if value_cols else None
            dims = []
            for col in plot_df.columns[1:]:
                vals = plot_df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if vals.empty:
                    continue
                dims.append(dict(
                    label=col,
                    values=plot_df[col].fillna(plot_df[col].median()),
                    range=[float(vals.min()), float(vals.max())],
                ))
            color_col = (first_obj_col if first_obj_col and first_obj_col in plot_df.columns
                         else plot_df.columns[-1])
            color_vals = plot_df[color_col].fillna(0)
            fig_pc = go.Figure(data=go.Parcoords(
                line=dict(color=color_vals, colorscale="RdYlGn_r", showscale=True,
                          colorbar=dict(title=f"{color_col} (↓ better)")),
                dimensions=dims,
            ))
            fig_pc.update_layout(
                title=f"Parallel Coordinates — Pareto trials  |  colour = {color_col} (red = worst, green = best)",
                height=500, margin=dict(l=80, r=80, t=60, b=20),
            )
            html_parts.append('<div style="overflow-x:auto">' + _plotly_div(fig_pc) + "</div>")
        except Exception as exc:
            warnings.warn(f"Parallel coordinates (plotly) failed: {exc}")
            _draw_parallel_coords_mpl(pareto_df, param_cols, value_cols, obj_names, html_parts)
    else:
        _draw_parallel_coords_mpl(pareto_df, param_cols, value_cols, obj_names, html_parts)

    if n_pareto >= 5:
        html_parts.append(_conclude_box(
            f"The Pareto front contains {n_pareto} of {n_total} trials ({100*n_pareto/n_total:.1f}%). "
            f"In the parallel coordinates plot, lines that converge to the same narrow band across "
            f"multiple parameter axes identify the most constrained parameters — those where only a "
            f"specific range leads to Pareto-optimal performance."
        ))
    return "\n".join(html_parts)


def _draw_parallel_coords_mpl(pareto_df, param_cols, value_cols, obj_names, html_parts):
    cols = param_cols[:8] + value_cols
    labels = [
        _clean_param_name(c) if c.startswith("param_") else _objective_label(c, obj_names)
        for c in cols
    ]
    data = pareto_df[cols].copy()
    norm_data = (data - data.min()) / (data.max() - data.min() + 1e-12)
    n_axes = len(cols)
    fig, ax = plt.subplots(figsize=(min(max(8, 2*n_axes), 14), 4))
    cmap = plt.cm.RdYlGn_r
    first_obj = value_cols[0] if value_cols else cols[-1]
    raw_c = pareto_df[first_obj].values
    color_n = (raw_c - np.nanmin(raw_c)) / (np.nanmax(raw_c) - np.nanmin(raw_c) + 1e-12)
    for idx, (_, row) in enumerate(norm_data.reset_index(drop=True).iterrows()):
        ax.plot(range(n_axes), row[cols].values, color=cmap(color_n[idx]), alpha=0.5, lw=0.8)
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_title("Parallel Coordinates — Pareto trials (colour = first objective, lower = better)")
    plt.tight_layout()
    html_parts.append(_img_html(_fig_to_base64(fig), "Pareto parallel coordinates"))


# ===========================================================================
# Layer 3 — Pairwise interactions + iso-contour overlays
# ===========================================================================

def _layer3_interactions(
    df: pd.DataFrame,
    obj_names: list[str],
    top_params: list[str],
) -> str:
    html_parts: list[str] = []
    param_cols = [f"param_{p}" for p in top_params if f"param_{p}" in df.columns]
    value_cols = _value_columns(df)
    n_obj = len(value_cols)

    if len(param_cols) < 2:
        return "<p>Not enough parameters for interaction analysis.</p>"

    # Spearman correlation matrix
    param_data = df[param_cols].copy().dropna()
    param_data.columns = [_clean_param_name(c) for c in param_cols]
    top_corr_pairs: list[tuple[str, str, float]] = []

    if len(param_data) > 5 and len(param_cols) > 1:
        corr_result = spearmanr(param_data)
        raw_corr = corr_result.statistic if hasattr(corr_result, "statistic") else corr_result[0]
        n_p = len(param_data.columns)
        if np.ndim(raw_corr) == 0:
            corr = np.ones((n_p, n_p))
            if n_p == 2:
                corr[0, 1] = corr[1, 0] = float(raw_corr)
        else:
            corr = np.atleast_2d(raw_corr)

        fig, ax = plt.subplots(figsize=(max(6, n_p * 0.9), max(5, n_p * 0.85)))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_p)); ax.set_yticks(range(n_p))
        ax.set_xticklabels(param_data.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(param_data.columns, fontsize=8)
        plt.colorbar(im, ax=ax, label="Spearman \u03c1")
        ax.set_title("Parameter Spearman Correlation Matrix (all trials)")
        for ii in range(corr.shape[0]):
            for jj in range(corr.shape[1]):
                ax.text(jj, ii, f"{corr[ii,jj]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(corr[ii,jj]) > 0.6 else "black")
        for ii in range(corr.shape[0]):
            for jj in range(ii+1, corr.shape[1]):
                if abs(corr[ii, jj]) > 0.4:
                    top_corr_pairs.append(
                        (param_data.columns[ii], param_data.columns[jj], corr[ii, jj])
                    )
        plt.tight_layout()
        html_parts.append(_img_html(_fig_to_base64(fig),
            "Parameter Spearman \u03c1 — positive = sampled together; negative = explored in opposition"))

    # 2-D scatter + iso-contour per pair x objective
    pairs = [
        (param_cols[i], param_cols[j])
        for i in range(len(param_cols))
        for j in range(i+1, len(param_cols))
    ][:_MAX_CONTOUR_PAIRS]

    for pair in pairs:
        p1, p2 = pair
        n1, n2 = _clean_param_name(p1), _clean_param_name(p2)
        valid_mask = df[[p1, p2]].notna().all(axis=1)
        sub = df[valid_mask].copy()
        if len(sub) < 10:
            continue

        fig, axes = plt.subplots(1, n_obj, figsize=(min(5.5*n_obj, 16), 5), squeeze=False)
        fig.suptitle(f"Interaction: {n1}  \u00d7  {n2}", fontsize=11, fontweight="bold")

        for i, vcol in enumerate(value_cols):
            ax = axes[0][i]
            obj_label = _objective_label(vcol, obj_names)
            x, y, z = sub[p1].values, sub[p2].values, sub[vcol].values
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            xv, yv, zv = x[valid], y[valid], z[valid]
            if len(xv) < 5:
                continue

            use_logx = np.all(xv > 0) and (np.max(xv) / np.min(xv) > 100)
            use_logy = np.all(yv > 0) and (np.max(yv) / np.min(yv) > 100)
            xp = np.log10(xv) if use_logx else xv
            yp = np.log10(yv) if use_logy else yv

            # Iso-contour background (Delaunay triangulation; needs >= 8 unique points)
            contour_drawn = False
            if len(xp) >= 8:
                try:
                    triang = Triangulation(xp, yp)
                    # Mask long-edge "sliver" triangles that cause noise artefacts
                    tri_verts = np.stack([xp[triang.triangles], yp[triang.triangles]], axis=2)
                    # shape (n_tri, 3, 2) — compute max edge length per triangle
                    e01 = np.hypot(tri_verts[:,1,0]-tri_verts[:,0,0], tri_verts[:,1,1]-tri_verts[:,0,1])
                    e12 = np.hypot(tri_verts[:,2,0]-tri_verts[:,1,0], tri_verts[:,2,1]-tri_verts[:,1,1])
                    e20 = np.hypot(tri_verts[:,0,0]-tri_verts[:,2,0], tri_verts[:,0,1]-tri_verts[:,2,1])
                    max_edge = np.max(np.stack([e01, e12, e20], axis=1), axis=1)
                    threshold = np.percentile(max_edge, 80)
                    triang.set_mask(max_edge > threshold)
                    tcf = ax.tricontourf(triang, zv, levels=6, cmap="RdYlGn_r",
                                         alpha=0.30, zorder=0)
                    ax.tricontour(triang, zv, levels=6, colors="gray",
                                  alpha=0.30, linewidths=0.4, zorder=1)
                    plt.colorbar(tcf, ax=ax, label=f"{obj_label} (iso)")
                    contour_drawn = True
                except Exception:
                    pass

            sc = ax.scatter(xp, yp, c=zv, cmap="RdYlGn_r", alpha=0.7, s=18,
                            edgecolors="none", zorder=2)
            if not contour_drawn:
                plt.colorbar(sc, ax=ax, label=obj_label)

            pareto_sub = sub[sub["is_pareto"] & valid_mask.loc[sub.index]]
            if not pareto_sub.empty:
                px = np.log10(pareto_sub[p1].values) if use_logx else pareto_sub[p1].values
                py = np.log10(pareto_sub[p2].values) if use_logy else pareto_sub[p2].values
                ax.scatter(px, py, c="white", s=55, marker="*",
                           edgecolors="black", lw=0.7, label="Pareto", zorder=5)
                ax.legend(fontsize=7)

            ax.set_xlabel(f"log\u2081\u2080({n1})" if use_logx else n1, fontsize=9)
            ax.set_ylabel(f"log\u2081\u2080({n2})" if use_logy else n2, fontsize=9)
            ax.set_title(f"{obj_label} (\u2193 better)", fontsize=9)

        plt.tight_layout()
        html_parts.append(_img_html(_fig_to_base64(fig),
            f"Interaction: {n1} \u00d7 {n2} — background iso-contours show equipotential lines "
            f"(same objective value); \u2605 = Pareto trials"))

    # Auto-conclusion
    conclusion_parts: list[str] = []
    if top_corr_pairs:
        strong = [(a, b, r) for a, b, r in top_corr_pairs if abs(r) > 0.5]
        if strong:
            items = [f"<code>{a}</code>/<code>{b}</code> (\u03c1={r:+.2f})" for a, b, r in strong[:3]]
            conclusion_parts.append(
                f"Strongly correlated parameter pairs: {', '.join(items)}. "
                f"Positive \u03c1 means the optimizer tends to set both high or both low together; "
                f"negative \u03c1 means they trade off against each other."
            )
    if not conclusion_parts:
        conclusion_parts.append(
            "No strongly correlated parameter pairs detected (all |\u03c1| < 0.5). "
            "Parameters appear to be explored roughly independently in this study."
        )
    conclusion_parts.append(
        "In the scatter/contour plots, the \u2605 Pareto markers cluster in the "
        "region of lowest objective value — that region defines the empirically "
        "optimal joint range for those two parameters."
    )
    html_parts.append(_conclude_box(" ".join(conclusion_parts)))
    return "\n".join(html_parts)


# ===========================================================================
# Layer 4 — Objective correlation & conflict
# ===========================================================================

def _layer4_objectives(df: pd.DataFrame, obj_names: list[str]) -> str:
    html_parts: list[str] = []
    value_cols = _value_columns(df)
    n_obj = len(value_cols)

    if n_obj < 2:
        return "<p>Single-objective study — no conflict analysis applicable.</p>"

    obj_df = df[value_cols + ["is_pareto"]].copy()
    obj_labels = [_objective_label(c, obj_names) for c in value_cols]
    obj_df.columns = obj_labels + ["is_pareto"]
    obj_df = obj_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(obj_df) < 5:
        return _warn_box("Insufficient data for objective correlation analysis.")

    corr_result = spearmanr(obj_df[obj_labels].values)
    if n_obj == 2:
        r = corr_result.statistic if hasattr(corr_result, "statistic") else corr_result[0]
        p = corr_result.pvalue if hasattr(corr_result, "pvalue") else corr_result[1]
        corr_mat = np.array([[1.0, r], [r, 1.0]])
        pval_mat = np.array([[0.0, p], [p, 0.0]])
    else:
        corr_mat = np.atleast_2d(
            corr_result.statistic if hasattr(corr_result, "statistic") else corr_result[0]
        )
        pval_mat = np.atleast_2d(
            corr_result.pvalue if hasattr(corr_result, "pvalue") else corr_result[1]
        )

    fig, ax = plt.subplots(figsize=(max(5, n_obj * 1.6), max(4, n_obj * 1.5)))
    im = ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Spearman \u03c1")
    ax.set_xticks(range(n_obj)); ax.set_yticks(range(n_obj))
    ax.set_xticklabels(obj_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(obj_labels, fontsize=9)
    ax.set_title("Objective Correlation (Spearman)\n\u2605 = p<0.05")
    for ii in range(n_obj):
        for jj in range(n_obj):
            txt = f"{corr_mat[ii,jj]:.2f}"
            if ii != jj and pval_mat[ii, jj] < 0.05:
                txt += "\u2605"
            ax.text(jj, ii, txt, ha="center", va="center",
                    fontsize=8, color="white" if abs(corr_mat[ii,jj]) > 0.6 else "black")
    plt.tight_layout()
    html_parts.append(_img_html(_fig_to_base64(fig), "Objective Correlation Heatmap"))

    # Pairwise scatter
    n_row = n_obj - 1
    fig2, axes2 = plt.subplots(n_row, n_row, figsize=(4*n_row, 3.5*n_row), squeeze=False)
    for i in range(n_row):
        for j in range(n_row):
            ax = axes2[i][j]
            if j < i:
                ax.set_visible(False); continue
            xi, yi = obj_labels[j+1], obj_labels[i]
            ax.scatter(obj_df[xi], obj_df[yi], s=8, alpha=0.3, c=_ALL_COLOR, edgecolors="none")
            p_sub = obj_df[obj_df["is_pareto"]]
            ax.scatter(p_sub[xi], p_sub[yi], s=25, alpha=0.85, c=_PARETO_COLOR, edgecolors="none", label="Pareto")
            ax.set_xlabel(xi, fontsize=8); ax.set_ylabel(yi, fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig2.suptitle("Objective Pairwise Scatter  (red = Pareto-optimal)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    html_parts.append(_img_html(_fig_to_base64(fig2), "Objective pairwise scatter matrix"))

    # PCA of Pareto front
    pareto_vals = obj_df.loc[obj_df["is_pareto"], obj_labels].values
    if len(pareto_vals) >= 3:
        try:
            qt = QuantileTransformer(output_distribution="normal", random_state=42,
                                     n_quantiles=min(len(pareto_vals), 200))
            scaled = qt.fit_transform(pareto_vals)
            pca = PCA()
            pca.fit(scaled)
            explained = pca.explained_variance_ratio_
            loadings = pca.components_

            fig3, (ax_ev, ax_ld) = plt.subplots(1, 2, figsize=(10, 4))
            ax_ev.bar(range(1, len(explained)+1), explained*100,
                      color=plt.cm.tab10.colors[:len(explained)])
            ax_ev.set_xlabel("Principal Component"); ax_ev.set_ylabel("Explained Variance (%)")
            ax_ev.set_title("PCA of Pareto Front (objective space)")
            ax_ev.set_xticks(range(1, len(explained)+1))

            im2 = ax_ld.imshow(np.abs(loadings), cmap="Reds", aspect="auto", vmin=0, vmax=1)
            ax_ld.set_xticks(range(n_obj)); ax_ld.set_yticks(range(len(loadings)))
            ax_ld.set_xticklabels(obj_labels, rotation=30, ha="right", fontsize=8)
            ax_ld.set_yticklabels(
                [f"PC{k+1} ({explained[k]*100:.1f}%)" for k in range(len(loadings))], fontsize=8
            )
            plt.colorbar(im2, ax=ax_ld, label="|Loading|")
            ax_ld.set_title("Which objectives define each trade-off axis?\n(|loading| = contribution to that PC)")
            for ii in range(loadings.shape[0]):
                for jj in range(loadings.shape[1]):
                    ax_ld.text(jj, ii, f"{loadings[ii,jj]:.2f}", ha="center", va="center",
                               fontsize=7, color="white" if abs(loadings[ii,jj]) > 0.5 else "black")
            plt.tight_layout()
            html_parts.append(_img_html(_fig_to_base64(fig3), "PCA of the Pareto front in objective space"))

            trade_offs: list[str] = []
            conflicts: list[tuple[str, str]] = []
            for k, (pc, ev) in enumerate(zip(loadings, explained)):
                pos = [obj_labels[j] for j in range(n_obj) if pc[j] > 0.3]
                neg = [obj_labels[j] for j in range(n_obj) if pc[j] < -0.3]
                if pos and neg:
                    trade_offs.append(
                        f"<li><strong>PC{k+1} ({ev*100:.1f}%):</strong> Trade-off — "
                        f"<em>{', '.join(pos)}</em> (high) vs <em>{', '.join(neg)}</em> (low). "
                        f"Improving one worsens the other.</li>"
                    )
                    conflicts.extend([(p, n) for p in pos for n in neg])
                elif pos:
                    trade_offs.append(
                        f"<li><strong>PC{k+1} ({ev*100:.1f}%):</strong> "
                        f"{', '.join(pos)} co-vary — these objectives are aligned.</li>"
                    )
            if trade_offs:
                html_parts.append("<h4>Principal trade-off axes:</h4><ul>" + "\n".join(trade_offs) + "</ul>")

            if conflicts:
                conf_str = "; ".join(
                    f"<strong>{a}</strong> vs <strong>{b}</strong>" for a, b in conflicts[:3]
                )
                html_parts.append(_conclude_box(
                    f"Key trade-offs detected: {conf_str}. "
                    f"No single parameter set can perfectly minimise all objectives simultaneously — "
                    f"select a Pareto-optimal solution that best reflects your experimental priorities."
                ))
            else:
                html_parts.append(_conclude_box(
                    "Objectives appear largely aligned. "
                    "A parameter set that improves one objective tends to improve the others."
                ))
        except Exception as exc:
            warnings.warn(f"Pareto PCA failed: {exc}")

    return "\n".join(html_parts)


# ===========================================================================
# Layer 5 — Regime / cluster detection
# ===========================================================================

def _layer5_clusters(df: pd.DataFrame, obj_names: list[str]) -> tuple[pd.DataFrame, str]:
    html_parts: list[str] = []
    param_cols = _param_columns(df)
    value_cols = _value_columns(df)
    pareto_df = df[df["is_pareto"]].copy().dropna(subset=param_cols)

    if len(pareto_df) < _KMEANS_MIN_TRIALS:
        return pd.DataFrame(), _warn_box(
            f"Only {len(pareto_df)} Pareto trials (minimum: {_KMEANS_MIN_TRIALS}). "
            f"Run more trials to enable cluster detection."
        )

    X_raw = pareto_df[param_cols].values.astype(float)
    for j in range(X_raw.shape[1]):
        col_vals = X_raw[:, j]
        valid = col_vals[np.isfinite(col_vals) & (col_vals > 0)]
        if len(valid) > 1 and np.max(valid) / np.min(valid) > 100:
            X_raw[:, j] = np.log10(col_vals + 1e-300)

    qt = QuantileTransformer(output_distribution="normal", random_state=42,
                             n_quantiles=min(len(pareto_df), 200))
    X = qt.fit_transform(X_raw)

    max_k = min(_KMEANS_MAX_K, len(pareto_df) // 5)
    best_k, best_score = 2, -1.0
    scores: dict[int, float] = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sc = silhouette_score(X, labels)
        scores[k] = sc
        if sc > best_score:
            best_k, best_score = k, sc

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    pareto_df = pareto_df.copy()
    pareto_df["cluster"] = km_final.fit_predict(X)

    if len(scores) > 1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(list(scores.keys()), list(scores.values()), "o-", color=_PARETO_COLOR)
        ax.axvline(best_k, ls="--", color="gray",
                   label=f"Best k={best_k} (silhouette={best_score:.2f})")
        ax.set_xlabel("Number of clusters k"); ax.set_ylabel("Silhouette score")
        ax.set_title("K-means cluster quality"); ax.legend()
        plt.tight_layout()
        html_parts.append(_img_html(_fig_to_base64(fig), f"Silhouette scores — best k={best_k}"))

    param_labels = [_clean_param_name(c) for c in param_cols]
    obj_labels = [_objective_label(c, obj_names) for c in value_cols]

    summary_rows: list[dict] = []
    # Chunked layout: at most MAX_COLS parameters per row to avoid tiny subplots.
    MAX_COLS = 6
    n_p = len(param_cols)
    n_chunks = max(1, (n_p + MAX_COLS - 1) // MAX_COLS)
    n_grid_cols = min(n_p, MAX_COLS)
    n_grid_rows = best_k * n_chunks
    fig_c, axes_c = plt.subplots(
        n_grid_rows, n_grid_cols,
        figsize=(min(2.5 * n_grid_cols, 15), 3 * n_grid_rows),
        squeeze=False,
    )
    fig_c.suptitle(f"Cluster Profiles — {best_k} clusters of Pareto trials",
                   fontsize=12, fontweight="bold")
    # Hide axes slots that have no parameter assigned
    for gr in range(n_grid_rows):
        chunk_idx = gr % n_chunks
        col_start = chunk_idx * MAX_COLS
        n_in_chunk = min(MAX_COLS, n_p - col_start)
        for gc in range(n_in_chunk, n_grid_cols):
            axes_c[gr][gc].set_visible(False)

    for cluster_id in range(best_k):
        mask = pareto_df["cluster"] == cluster_id
        ct = pareto_df[mask]
        n_c = len(ct)
        row: dict = {"cluster": cluster_id, "n_trials": n_c}
        for p_col, p_lbl in zip(param_cols, param_labels):
            row[f"mean_{p_lbl}"] = float(np.nanmean(ct[p_col].values))
            row[f"std_{p_lbl}"] = float(np.nanstd(ct[p_col].values))
        for v_col, v_lbl in zip(value_cols, obj_labels):
            row[f"mean_{v_lbl}"] = float(np.nanmean(ct[v_col].values))
        summary_rows.append(row)

        for j, (p_col, p_lbl) in enumerate(zip(param_cols, param_labels)):
            chunk_idx = j // MAX_COLS
            col_idx = j % MAX_COLS
            row_idx = cluster_id * n_chunks + chunk_idx
            ax = axes_c[row_idx][col_idx]
            ax.hist(pareto_df[p_col].values, bins=15, alpha=0.3, color="gray",
                    density=True, label="All Pareto")
            ax.hist(ct[p_col].values, bins=10, alpha=0.7,
                    color=plt.cm.tab10(cluster_id / max(best_k, 1)), density=True)
            ax.set_title(p_lbl, fontsize=7); ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"Cluster {cluster_id}\n(n={n_c})", fontsize=8)

    plt.tight_layout()
    html_parts.append(_img_html(_fig_to_base64(fig_c),
        "Cluster distributions — grey = all Pareto; coloured = this cluster"))

    summary_df = pd.DataFrame(summary_rows)
    header = "<tr><th>Cluster</th><th>N</th>" + "".join(f"<th>Mean {l}</th>" for l in obj_labels) + "</tr>"
    table_rows = []
    for _, row in summary_df.iterrows():
        cells = [f"<td><strong>{int(row['cluster'])}</strong></td>",
                 f"<td>{int(row['n_trials'])}</td>"]
        for lbl in obj_labels:
            cells.append(f"<td>{row.get(f'mean_{lbl}', float('nan')):.4f}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    html_parts.append(
        "<h4>Cluster mean objective values</h4>"
        "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:0.9em'>"
        + header + "\n".join(table_rows) + "</table>"
    )

    # Fingerprints
    grand_mean = pareto_df[param_cols].mean()
    grand_std = pareto_df[param_cols].std() + 1e-30
    narrative = ["<h4>Cluster parameter fingerprints (z-score vs Pareto average):</h4><ul>"]
    for cluster_id in range(best_k):
        mask = pareto_df["cluster"] == cluster_id
        z = (pareto_df[mask][param_cols].mean() - grand_mean) / grand_std
        z_sorted = z.abs().sort_values(ascending=False)
        top3 = z_sorted.index[:3]
        desc = "; ".join(
            f"<code>{_clean_param_name(pc)}</code> "
            f"{'&#8593; higher' if z[pc] > 0 else '&#8595; lower'} (z={z[pc]:+.2f})"
            for pc in top3
        )
        narrative.append(f"<li><strong>Cluster {cluster_id}:</strong> {desc}</li>")
    narrative.append("</ul>")
    html_parts.append("\n".join(narrative))

    html_parts.append(_conclude_box(
        f"{best_k} distinct solution regimes were identified among the {len(pareto_df)} Pareto trials. "
        f"Each regime is a qualitatively different region of parameter space that still achieves "
        f"good performance — for example, one regime may compensate a low kinetic rate with stronger "
        f"adhesion, while another relies on fast dynamics with tighter spatial patterning. "
        f"The fingerprints above show the defining parameter signature of each regime, "
        f"providing distinct mechanistic hypotheses for experimental validation."
    ))
    return summary_df, "\n".join(html_parts)


# ===========================================================================
# Layer 6 — 1-D sensitivity slices (LOWESS-smoothed)
# ===========================================================================

def _layer6_sensitivity(
    df: pd.DataFrame,
    obj_names: list[str],
    top_params: list[str],
) -> str:
    html_parts: list[str] = []
    param_cols = [f"param_{p}" for p in top_params[:_MAX_SLICE_PARAMS] if f"param_{p}" in df.columns]
    value_cols = _value_columns(df)
    n_obj = len(value_cols)

    html_parts.append(
        "<p><strong>Reading these plots:</strong> "
        "The y-axis shows the <em>objective (error) value</em> — in Cellfoundry this is always an error metric, "
        "so <strong>lower values = better match to the biological target</strong>. "
        "The navy trend line is a LOWESS-smoothed moving average: it answers "
        "'as I change only this parameter, how does the typical error change?' "
        "Red &#9733; markers are Pareto-optimal trials and correctly appear at the "
        "<em>bottom</em> of each plot (lowest error = best solutions). "
        "Steep slopes indicate high sensitivity to this parameter; "
        "flat regions mean the objective is insensitive. "
        "A U- or V-shaped curve suggests an optimal range between the extremes.</p>"
    )

    top_sensitive: list[tuple[str, float]] = []

    for p_col in param_cols:
        p_label = _clean_param_name(p_col)
        valid_mask = df[[p_col]].notna().all(axis=1)
        sub = df[valid_mask].copy()
        if len(sub) < 10:
            continue

        x_raw = sub[p_col].values
        use_log = np.all(x_raw > 0) and (np.max(x_raw) / np.min(x_raw) > 100)
        x_plot = np.log10(x_raw) if use_log else x_raw
        log_note = " (log\u2081\u2080 scale)" if use_log else ""

        fig, axes = plt.subplots(1, n_obj, figsize=(min(6*n_obj, 18), 5), squeeze=False)
        fig.suptitle(f"Sensitivity: {p_label}{log_note}", fontsize=11, fontweight="bold")

        max_slope = 0.0
        for i, v_col in enumerate(value_cols):
            ax = axes[0][i]
            obj_label = _objective_label(v_col, obj_names)
            y = sub[v_col].values
            valid = np.isfinite(x_plot) & np.isfinite(y)
            xv, yv = x_plot[valid], y[valid]
            if len(xv) < 5:
                continue

            ax.scatter(xv, yv, s=8, alpha=0.2, c=_ALL_COLOR, edgecolors="none", label="All trials")
            order = np.argsort(xv)
            xs, ys = xv[order], yv[order]
            window = max(5, len(xs) // 8)
            ys_smooth = uniform_filter1d(ys, size=window, mode="nearest")
            ax.plot(xs, ys_smooth, color="navy", lw=2, label="LOWESS trend")

            if len(xs) > 1:
                x_range = float(np.ptp(xs))
                y_range = float(np.ptp(ys_smooth))
                slope = y_range / (x_range + 1e-30)
                max_slope = max(max_slope, slope)

            pareto_mask = sub["is_pareto"].values & valid
            if np.any(pareto_mask):
                ax.scatter(x_plot[pareto_mask], y[pareto_mask], s=40,
                           c=_PARETO_COLOR, edgecolors="black", lw=0.5,
                           marker="*", label="Pareto (best)", zorder=5)

            xlabel = f"log\u2081\u2080({p_label})" if use_log else p_label
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(f"{obj_label}\n(error — lower = better)", fontsize=8)
            ax.set_title(obj_label, fontsize=9)
            ax.legend(fontsize=7)

        plt.tight_layout()
        html_parts.append(_img_html(_fig_to_base64(fig), f"1-D sensitivity: {p_label}"))
        if max_slope > 0:
            top_sensitive.append((p_label, max_slope))

    if top_sensitive:
        top_sensitive.sort(key=lambda x: x[1], reverse=True)
        most = [p for p, _ in top_sensitive[:3]]
        html_parts.append(_conclude_box(
            f"Most sensitive parameters (steepest error-slope): "
            f"{', '.join(f'<code>{p}</code>' for p in most)}. "
            f"These are the highest-leverage targets for experimental validation — "
            f"small changes in these parameters produce the largest changes in model output."
        ))
    return "\n".join(html_parts)


# ===========================================================================
# Layer 7 — Parameter distributions: Pareto vs all (violin plots)
# ===========================================================================

def _layer7_violin_distributions(df: pd.DataFrame, obj_names: list[str]) -> str:
    """Side-by-side violin plots comparing Pareto-optimal vs all completed trials."""
    html_parts: list[str] = []
    param_cols = _param_columns(df)
    pareto_df = df[df["is_pareto"]].copy()
    all_df = df.copy()

    if len(pareto_df) < 3:
        return _warn_box(
            f"Only {len(pareto_df)} Pareto trials — need at least 3 for violin comparison. "
            f"This layer will become available once more trials are completed."
        )

    n_params = len(param_cols)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(min(4.5*n_cols, 16), 3.5*n_rows), squeeze=False)
    fig.suptitle(
        "Parameter Distributions: All trials (blue) vs Pareto-optimal (red)\n"
        "A narrow red violin = clear 'sweet spot'; a shifted red violin = directional preference",
        fontsize=12, fontweight="bold",
    )

    narrow_params: list[str] = []
    shifted_params: list[str] = []

    for idx, p_col in enumerate(param_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        p_label = _clean_param_name(p_col)

        all_vals = all_df[p_col].replace([np.inf, -np.inf], np.nan).dropna().values
        pareto_vals = pareto_df[p_col].replace([np.inf, -np.inf], np.nan).dropna().values

        if len(all_vals) < 3 or len(pareto_vals) < 2:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(p_label, fontsize=9); continue

        use_log = np.all(all_vals > 0) and (np.max(all_vals) / np.min(all_vals) > 100)
        all_plot = np.log10(all_vals) if use_log else all_vals
        pareto_plot = np.log10(pareto_vals) if use_log else pareto_vals

        vp_all = ax.violinplot([all_plot], positions=[0], showmedians=True, showextrema=True)
        vp_par = ax.violinplot([pareto_plot], positions=[1], showmedians=True, showextrema=True)
        for pc in vp_all["bodies"]:
            pc.set_facecolor(_ALL_COLOR); pc.set_alpha(0.55)
        for pc in vp_par["bodies"]:
            pc.set_facecolor(_PARETO_COLOR); pc.set_alpha(0.65)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in vp_all:
                vp_all[key].set_color(_ALL_COLOR)
            if key in vp_par:
                vp_par[key].set_color(_PARETO_COLOR)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["All trials", "Pareto best"], fontsize=8)
        ax.set_title(f"log\u2081\u2080({p_label})" if use_log else p_label, fontsize=9)
        if use_log:
            ax.set_ylabel("log\u2081\u2080(value)", fontsize=8)

        all_std = float(np.std(all_plot))
        par_std = float(np.std(pareto_plot)) if len(pareto_plot) > 1 else all_std
        all_med = float(np.median(all_plot))
        par_med = float(np.median(pareto_plot))
        if all_std > 0 and par_std < all_std * 0.5:
            narrow_params.append(p_label)
        if all_std > 0 and abs(par_med - all_med) > 0.4 * all_std:
            shifted_params.append(p_label)

    for idx in range(n_params, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    html_parts.append(_img_html(_fig_to_base64(fig),
        "Violin plots — wide blue: all search trials; narrow red: Pareto-optimal trials"))

    parts: list[str] = []
    if narrow_params:
        parts.append(
            f"<strong>Clear sweet spots</strong> (Pareto range much narrower than search range): "
            f"{', '.join(f'<code>{p}</code>' for p in narrow_params[:5])}. "
            f"Experimental interventions should target these specific ranges."
        )
    if shifted_params:
        parts.append(
            f"<strong>Directional preferences</strong> (best solutions biased toward high or low values): "
            f"{', '.join(f'<code>{p}</code>' for p in shifted_params[:5])}. "
            f"These suggest monotonic relationships worth validating experimentally."
        )
    if not parts:
        parts.append(
            "Pareto distributions largely overlap with the full search space. "
            "No single parameter shows a strong directional preference or narrow sweet spot — "
            "the objective landscape may be flat or the search space may need expansion."
        )
    html_parts.append(_conclude_box(" ".join(parts)))
    return "\n".join(html_parts)


# ===========================================================================
# Summary table
# ===========================================================================

def _build_summary_table(df: pd.DataFrame, obj_names: list[str]) -> str:
    value_cols = _value_columns(df)
    param_cols = _param_columns(df)
    pareto_df = df[df["is_pareto"]].copy()
    if pareto_df.empty:
        pareto_df = df.nsmallest(10, "value_0").copy()
    pareto_df = pareto_df.sort_values("value_0").head(20).reset_index(drop=True)
    display = ["trial_number"] + value_cols + param_cols
    pareto_df = pareto_df[display]
    pareto_df.columns = (
        ["Trial"]
        + [_objective_label(c, obj_names) for c in value_cols]
        + [_clean_param_name(c) for c in param_cols]
    )

    def _fmt(v):
        return f"{v:.5g}" if isinstance(v, float) else str(v)

    rows = ["<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
            for _, row in pareto_df.iterrows()]
    header = "".join(f"<th>{c}</th>" for c in pareto_df.columns)
    return (
        "<div style='overflow-x:auto'>"
        "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:0.85em'>"
        f"<tr>{header}</tr>" + "\n".join(rows) + "</table></div>"
    )


# ===========================================================================
# HTML template
# ===========================================================================

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cellfoundry Analysis \u2014 {study_name}</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; max-width: 1600px; margin: 0 auto; padding: 1em 2em; color: #222; }}
  h1 {{ color: #c0392b; border-bottom: 2px solid #c0392b; display:flex; align-items:center; gap:0.4em; line-height:1.3; }}
  h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; margin-top: 2em; }}
  h3 {{ color: #34495e; }}
  h4 {{ color: #555; }}
  figure {{ margin: 1em 0; text-align: center; }}
  figcaption {{ font-size: 0.85em; color: #666; margin-top: 0.3em; font-style: italic; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th {{ background: #2c3e50; color: white; padding: 6px 10px; }}
  td {{ padding: 4px 10px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f8f8f8; }}
  .meta {{ background: #eaf0fb; border-left: 4px solid #2980b9; padding: 0.7em 1em; margin: 1em 0; }}
  .toc a {{ color: #2980b9; display: block; margin: 0.2em 0; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
  .section {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 1em 1.5em; margin: 1.5em 0; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
</style>
{plotly_js}
</head>
<body>

<h1>
  {icon_html}
  Cellfoundry Optimizer \u2014 Interpretability Report
</h1>

<div class="meta">
  <strong>Study:</strong> {study_name}<br>
  <strong>Storage:</strong> {storage}<br>
  <strong>Generated:</strong> {timestamp}<br>
  <strong>Completed trials:</strong> {n_trials} &nbsp;|&nbsp;
  <strong>Pareto-optimal trials:</strong> {n_pareto}<br>
  <strong>Parameters searched:</strong> {n_params}
</div>

<div class="section" id="objectives">
{objectives_block}
</div>

<div class="toc">
  <strong>Contents:</strong>
  <a href="#best-trials">&#9654; Best Trials (Pareto front)</a>
  <a href="#layer-rf">&#9654; Layer 0 \u2014 Surrogate model importance (GBR + permutation)</a>
  <a href="#layer1">&#9654; Layer 1 \u2014 Global parameter importance (fANOVA / MDI)</a>
  <a href="#layer2">&#9654; Layer 2 \u2014 Pareto-front analysis &amp; parallel coordinates</a>
  <a href="#layer3">&#9654; Layer 3 \u2014 Pairwise interactions &amp; iso-contour plots</a>
  <a href="#layer4">&#9654; Layer 4 \u2014 Objective correlation &amp; conflict</a>
  <a href="#layer5">&#9654; Layer 5 \u2014 Regime / cluster detection</a>
  <a href="#layer6">&#9654; Layer 6 \u2014 1-D sensitivity slices</a>
  <a href="#layer7">&#9654; Layer 7 \u2014 Parameter distributions (Pareto vs all)</a>
</div>

<div class="section" id="best-trials">
<h2>Best Trials \u2014 Pareto Front</h2>
<p>Sorted by first objective (ascending = best). All values are <strong>error metrics</strong> \u2014 lower = better match to target.</p>
{best_trials_table}
</div>

<div class="section" id="layer-rf">
<h2>Layer 0 \u2014 Surrogate Model Importance</h2>
<p>A <strong>Gradient Boosting Regressor (GBR)</strong> is fitted per objective on all completed trials.
<strong>Permutation importance</strong> then estimates how much predictive accuracy is lost when each
parameter is randomly shuffled \u2014 a model-agnostic measure robust to correlated parameters and non-linearities.
The cross-validated R\u00b2 tells you how well the surrogate captures the landscape:
R\u00b2=1 is a perfect fit; R\u00b2=0 means no better than predicting the mean;
<strong>R\u00b2&lt;0</strong> means the surrogate fits <em>worse</em> than the mean
(common with &lt;50 trials, noisy objectives, or unexplored regions \u2014 treat such importance scores with caution).</p>
{layer_rf}
</div>

<div class="section" id="layer1">
<h2>Layer 1 \u2014 Global Parameter Importance</h2>
<p>Two complementary methods:
<strong>fANOVA</strong> (functional ANOVA) decomposes total objective variance into contributions from
individual parameters and their interactions; it is unbiased and accounts for the structure of the search space.
<strong>MDI</strong> (Mean Decrease Impurity) measures how often each parameter is chosen for a split
in a random forest weighted by impurity reduction; it is faster but can be biased toward parameters that
take many distinct values (high cardinality) \u2014 for example, a continuous parameter sampled on a log-scale
over 4 decades will have many unique values and may appear inflated in MDI even if its true effect is weak.
Comparing both methods identifies robust vs. potentially biased signals.</p>
{layer1}
</div>

<div class="section" id="layer2">
<h2>Layer 2 \u2014 Pareto-Front Parameter Importance</h2>
<p>The same importance analyses repeated on <em>only</em> Pareto-optimal trials.
Global importance (Layer 1) is often dominated by the boundary between feasible and infeasible regions
(e.g., a parameter that determines whether any RG cells form at all).
Pareto-only importance reveals the finer knobs that control solution <em>quality within</em> the feasible
region \u2014 the more actionable question for experimental design.</p>
{layer2}
</div>

<div class="section" id="layer3">
<h2>Layer 3 \u2014 Pairwise Interactions</h2>
<p>For each pair of top-ranked parameters, a 2-D scatter plot shows the joint effect on each objective.
The background <strong>iso-contour (equipotential) lines</strong> connect parameter combinations that
yield the same objective value, computed via Delaunay triangulation of the trial data.
<em>Curved</em> iso-contours indicate a <strong>non-linear interaction</strong>: the optimal value
of one parameter depends on the value of the other, and they cannot be tuned independently.
<em>Parallel straight</em> iso-contours indicate that the two parameters act independently.
The Spearman correlation matrix at the top shows whether the optimizer tends to sample the two
parameters together (positive \u03c1) or in opposition (negative \u03c1).</p>
{layer3}
</div>

<div class="section" id="layer4">
<h2>Layer 4 \u2014 Objective Correlation &amp; Trade-off Analysis</h2>
<p>Spearman \u03c1 between objectives reveals alignment (positive \u03c1: easy to optimise together)
or conflict (negative \u03c1: trade-off; improving one worsens the other).
PCA of the Pareto front in objective space extracts the <strong>principal trade-off axes</strong>:
each PC is an independent direction of variation among the best solutions.
The loading plot shows which objectives co-vary along each PC \u2014 objectives with loadings of
opposite sign on the same PC are in direct tension with each other.</p>
{layer4}
</div>

<div class="section" id="layer5">
<h2>Layer 5 \u2014 Regime / Cluster Detection</h2>
<p>K-means clustering groups Pareto-optimal trials in scaled parameter space to find qualitatively
different solution <em>regimes</em> \u2014 distinct parameter regions that all achieve good performance
through different mechanisms.
The best k is chosen by maximising the <strong>silhouette coefficient</strong> (ranges \u22121 to +1;
higher means clusters are more compact and well-separated from each other).
Each cluster's <em>fingerprint</em> shows which parameters are unusually high or low relative to the
Pareto average, expressed as a z-score. Different regimes represent distinct mechanistic hypotheses
that can be tested independently in the lab.</p>
{layer5}
</div>

<div class="section" id="layer6">
<h2>Layer 6 \u2014 1-D Sensitivity Slices</h2>
<p>For each important parameter, the marginal relationship with each objective is visualised using a
<strong>LOWESS-smoothed trend line</strong> (Locally Weighted Scatterplot Smoothing: a local moving
average that requires no assumed functional form, computed by fitting a low-degree polynomial to each
local neighbourhood of points weighted by proximity).
The trend answers: <em>"if I vary only this parameter, how does the typical error change?"</em>
Steep slopes = high sensitivity; flat regions = saturation or irrelevance; V/U-shape = optimal range.
These are marginal trends \u2014 interactions with other parameters are averaged out.</p>
{layer6}
</div>

<div class="section" id="layer7">
<h2>Layer 7 \u2014 Parameter Distributions: Pareto vs All Trials</h2>
<p>Violin plots directly show <em>where in the search space</em> the best solutions live.
A <strong>narrow red violin</strong> vs a wide blue violin indicates a clear "sweet spot" \u2014
the optimizer has converged to a specific range and performance degrades outside it.
A <strong>shifted red median</strong> indicates a directional preference (higher or lower values
consistently produce better outcomes).
Complete overlap between red and blue means the parameter has little influence on whether a trial
reaches the Pareto front.
This layer complements Layer 6 (how sensitive is the response?) by focusing on where the good
solutions are, making it more intuitive for experimental planning.</p>
{layer7}
</div>

</body>
</html>
"""


# ===========================================================================
# Main entry point
# ===========================================================================

def run_analysis(
    storage: str,
    study_name: str,
    *,
    objective_names: list[str] | None = None,
    out_dir: str = "optimizer/analysis_results",
    n_top_params: int = 8,
) -> Path:
    """Run the full interpretability analysis and write an HTML report."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] Loading study '{study_name}' from {storage} \u2026")
    study = _load_study(storage, study_name)

    print("[analyze] Building trial DataFrame \u2026")
    df = _trials_to_dataframe(study)

    value_cols = _value_columns(df)
    n_obj = len(value_cols)

    # Resolve objective names from YAML if not provided
    yaml_objectives: list[dict] = []
    yaml_path = _find_yaml_for_study(storage, study_name)
    if yaml_path:
        yaml_objectives = _parse_yaml_objectives(yaml_path)
        print(f"[analyze] Found YAML config: {yaml_path}")

    if objective_names:
        obj_names = list(objective_names)
        while len(obj_names) < n_obj:
            obj_names.append(f"Objective {len(obj_names)}")
    elif yaml_objectives:
        obj_names = [m["name"] for m in yaml_objectives]
        while len(obj_names) < n_obj:
            obj_names.append(f"Objective {len(obj_names)}")
    else:
        obj_names = [f"Objective {i}" for i in range(n_obj)]

    best_values = [float(df[c].min()) if c in df.columns else float("nan") for c in value_cols]
    try:
        directions_list = [d.name.lower() for d in study.directions]
    except Exception:
        directions_list = ["minimize"] * n_obj

    n_pareto = int(df["is_pareto"].sum())
    n_trials = len(df)
    param_cols = _param_columns(df)
    n_params = len(param_cols)
    print(f"[analyze] {n_trials} completed trials, {n_pareto} Pareto-optimal, "
          f"{n_obj} objectives, {n_params} parameters")

    # Export Pareto front CSV
    pareto_csv = out_path / "pareto_front.csv"
    pareto_export = df[df["is_pareto"]].copy()
    pareto_export.columns = [
        c.replace("param_", "").replace("value_", "obj_") for c in pareto_export.columns
    ]
    pareto_export.to_csv(pareto_csv, index=False)
    print(f"[analyze] Pareto front saved \u2192 {pareto_csv}")

    # Run all layers
    print("[analyze] Layer 0 \u2014 surrogate permutation importance \u2026")
    global_top_params, html_rf = _layer_rf_permutation(df, obj_names)

    print("[analyze] Layer 1 \u2014 global fANOVA / MDI importance \u2026")
    fanova_dict, html_l1 = _layer1_importance(study, df, obj_names, pareto_only=False)

    fanova_top: list[str] = []
    for imp_dict in fanova_dict.values():
        if imp_dict:
            for p in sorted(imp_dict, key=imp_dict.get, reverse=True)[:5]:
                if p not in fanova_top:
                    fanova_top.append(p)

    top_params = list(global_top_params[:n_top_params])
    for p in fanova_top:
        if p not in top_params:
            top_params.append(p)
    top_params = top_params[:n_top_params]
    if not top_params:
        top_params = [_clean_param_name(c) for c in param_cols][:n_top_params]

    print("[analyze] Layer 2 \u2014 Pareto-front importance \u2026")
    html_l2 = _layer2_pareto(study, df, obj_names, fanova_dict)

    print("[analyze] Layer 3 \u2014 pairwise interactions \u2026")
    html_l3 = _layer3_interactions(df, obj_names, top_params)

    print("[analyze] Layer 4 \u2014 objective correlation & PCA \u2026")
    html_l4 = _layer4_objectives(df, obj_names)

    print("[analyze] Layer 5 \u2014 cluster detection \u2026")
    cluster_summary_df, html_l5 = _layer5_clusters(df, obj_names)

    print("[analyze] Layer 6 \u2014 1-D sensitivity slices \u2026")
    html_l6 = _layer6_sensitivity(df, obj_names, top_params)

    print("[analyze] Layer 7 \u2014 parameter violin distributions \u2026")
    html_l7 = _layer7_violin_distributions(df, obj_names)

    best_trials_table = _build_summary_table(df, obj_names)
    objectives_block = _build_objectives_block(obj_names, yaml_objectives, best_values, directions_list)

    icon_b64 = _load_icon_b64()
    icon_html = (
        f'<img src="data:image/png;base64,{icon_b64}" '
        f'style="height:1.1em;vertical-align:middle" alt="Cellfoundry">'
        if icon_b64 else ""
    )
    # plotly CDN is now injected per-div via include_plotlyjs='cdn'
    plotly_js = ""

    summary = {
        "study_name": study_name,
        "storage": storage,
        "n_trials": n_trials,
        "n_pareto": n_pareto,
        "n_objectives": n_obj,
        "objective_names": obj_names,
        "best_values": {obj_names[i]: best_values[i] for i in range(n_obj)},
        "top_parameters_globally": top_params,
        "yaml_config": str(yaml_path) if yaml_path else None,
        "generated": datetime.now().isoformat(),
    }
    summary_json = out_path / "analysis_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] Summary JSON saved \u2192 {summary_json}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_name = f"analysis_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = out_path / report_name

    html = _HTML_TEMPLATE.format(
        study_name=study_name,
        storage=storage,
        timestamp=timestamp,
        n_trials=n_trials,
        n_pareto=n_pareto,
        n_params=n_params,
        objectives_block=objectives_block,
        best_trials_table=best_trials_table,
        icon_html=icon_html,
        layer_rf=html_rf,
        layer1=html_l1,
        layer2=html_l2,
        layer3=html_l3,
        layer4=html_l4,
        layer5=html_l5,
        layer6=html_l6,
        layer7=html_l7,
        plotly_js=plotly_js,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[analyze] Report written \u2192 {report_path}")
    return report_path


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cellfoundry Optimizer — Interpretability & Analysis Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Auto-detect single study:
              python -m optimizer.analyze --storage sqlite:///cellfoundry_radial_glia.db

              # Explicit study with named objectives:
              python -m optimizer.analyze \\
                  --storage sqlite:///cellfoundry_radial_glia.db \\
                  --study cellfoundry_radial_glia \\
                  --objective-names "n_large_clusters,maturity,compactness,rg_fraction"

              # One report per study in a multi-study DB:
              python -m optimizer.analyze \\
                  --storage sqlite:///cellfoundry_cell_speed.db --all-studies

              # List available studies:
              python -m optimizer.analyze \\
                  --storage sqlite:///cellfoundry_cell_speed.db --list-studies
        """),
    )
    p.add_argument("--storage", required=True, help="Optuna storage URL (e.g. sqlite:///my.db)")
    p.add_argument("--study", default=None,
                   help="Study name (auto-detected if only one study in DB)")
    p.add_argument("--objective-names", default=None,
                   help="Comma-separated objective names (overrides YAML auto-detection)")
    p.add_argument("--out-dir", default="optimizer/analysis_results",
                   help="Output directory (default: optimizer/analysis_results)")
    p.add_argument("--n-top-params", type=int, default=8,
                   help="Number of top parameters for pairwise/slice analyses (default: 8)")
    p.add_argument("--all-studies", action="store_true",
                   help="Generate one report per study in the DB")
    p.add_argument("--list-studies", action="store_true",
                   help="Print available study names and exit")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_studies:
        names = list_study_names(args.storage)
        print(f"Studies in {args.storage}:")
        for n in names:
            print(f"  {n}")
        return

    obj_names_list: list[str] | None = None
    if args.objective_names:
        obj_names_list = [s.strip() for s in args.objective_names.split(",")]

    if args.all_studies:
        names = list_study_names(args.storage)
        if not names:
            print("No studies found.")
            return
        for name in names:
            out_subdir = str(Path(args.out_dir) / name)
            print(f"\n{'='*60}\nAnalysing study: {name}\n{'='*60}")
            try:
                report = run_analysis(
                    storage=args.storage,
                    study_name=name,
                    objective_names=obj_names_list,
                    out_dir=out_subdir,
                    n_top_params=args.n_top_params,
                )
                print(f"Report: {report.resolve()}")
            except Exception as exc:
                print(f"ERROR analysing {name}: {exc}")
        return

    study_name = args.study
    if study_name is None:
        names = list_study_names(args.storage)
        if len(names) == 1:
            study_name = names[0]
            print(f"[analyze] Auto-detected study: {study_name}")
        elif len(names) == 0:
            print("ERROR: No studies found in the database.")
            return
        else:
            print(
                f"ERROR: Multiple studies found in {args.storage}:\n"
                + "\n".join(f"  {n}" for n in names)
                + "\n\nSpecify one with --study <name>, or use --all-studies."
            )
            return

    report_path = run_analysis(
        storage=args.storage,
        study_name=study_name,
        objective_names=obj_names_list,
        out_dir=args.out_dir,
        n_top_params=args.n_top_params,
    )
    print(f"\nDone!  Open in your browser:\n  {report_path.resolve()}")


if __name__ == "__main__":
    main()
