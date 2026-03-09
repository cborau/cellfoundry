"""
Optuna-based parameter optimization runner for CellFoundry.

Usage
-----
    python -m optimizer.optimize --config optimizer/optuna_config.yaml
    # or from the optimizer/ directory:
    python optimize.py --config optuna_config.yaml

Each trial:
  1. Optuna suggests values for the parameters listed in the YAML config.
  2. A JSON override file is written to a temporary trial directory.
  3. ``model.py`` is launched as a subprocess with ``--overrides`` and
     ``--result-dir`` flags (clean GPU memory between trials).
  4. The resulting pickle is loaded and passed to the chosen objective
     function(s) to compute one or more scalar errors.
  5. The error(s) are returned to Optuna for minimization.

Multi-objective
~~~~~~~~~~~~~~~
To optimize multiple objectives simultaneously, use the ``objectives``
(plural) key in the YAML instead of ``objective`` (singular)::

    study:
      directions: [minimize, minimize]

    objectives:
      - function: organoid_size_error
        kwargs: {target_size: 50.0, metric: radius_of_gyration}
      - function: final_cell_count_error
        kwargs: {target_cell_count: 15}

Optuna will use NSGA-II by default for multi-objective studies and maintain
a Pareto front.  The ``study.best_trials`` list contains Pareto-optimal trials.

Requirements
------------
    pip install optuna pyyaml
    (optional) pip install optuna-dashboard   # for live web dashboard
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Safe unpickler (stubs ModelParameterConfig so old pickles load cleanly)
# ---------------------------------------------------------------------------

class _ModelParameterConfigStub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ModelParameterConfig":
            return _ModelParameterConfigStub
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _normalize_objective_result(result) -> tuple[float, str | None]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            "Objective functions must return a tuple of (error, display_text)."
        )

    error, display_text = result
    if display_text is not None and not isinstance(display_text, str):
        raise ValueError(
            "display_text must be a string or None."
        )

    return float(error), display_text


def _format_objective_label(name: str, error: float, display_text: str | None) -> str:
    label = f"{name}={error:.6f}"
    if not display_text:
        return label
    return f"{label} {display_text}"


def _get_display_text(display_texts: list[str | None], index: int) -> str | None:
    if index < len(display_texts):
        return display_texts[index]
    return None


# ---------------------------------------------------------------------------
# Core trial runner
# ---------------------------------------------------------------------------

def run_trial_subprocess(
    overrides: dict,
    model_script: str,
    result_dir: str,
    timeout: int = 0,
) -> dict:
    """Run model.py as a subprocess with the given parameter overrides.

    Parameters
    ----------
    overrides : dict
        ``{PARAM_NAME: value}`` pairs.
    model_script : str
        Path to ``model.py``.
    result_dir : str
        Directory where the trial results will be written.
    timeout : int
        Maximum wall-clock seconds to wait (0 = unlimited).

    Returns
    -------
    dict
        The deserialized pickle results dictionary.
    """
    override_path = os.path.join(result_dir, "overrides.json")
    with open(override_path, "w") as f:
        json.dump(overrides, f, indent=2)

    cmd = [
        sys.executable, model_script,
        "--overrides", override_path,
        "--result-dir", result_dir,
    ]
    print(f"  [trial] Running: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout if timeout > 0 else None,
    )
    elapsed = time.time() - t0
    print(f"  [trial] Finished in {elapsed:.1f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        # Dump stderr for debugging
        err_log = os.path.join(result_dir, "stderr.log")
        with open(err_log, "w") as f:
            f.write(proc.stderr)
        raise RuntimeError(
            f"model.py exited with code {proc.returncode}. "
            f"See {err_log} for details.\n"
            f"Last 500 chars of stderr:\n{proc.stderr[-500:]}"
        )

    # Find the pickle file
    pickle_path = os.path.join(result_dir, "output_data_0.pickle")
    if not os.path.isfile(pickle_path):
        # Dump full stdout for post-mortem
        stdout_log = os.path.join(result_dir, "stdout.log")
        with open(stdout_log, "w") as f:
            f.write(proc.stdout)
        raise FileNotFoundError(
            f"Expected pickle not found at {pickle_path}. "
            f"Full stdout saved to {stdout_log}.\n"
            f"model.py stdout (last 1500 chars):\n{proc.stdout[-1500:]}"
        )

    return _load_pickle(pickle_path)


# ---------------------------------------------------------------------------
# Objective wrapper
# ---------------------------------------------------------------------------

def make_objective(config: dict, model_script: str, base_result_dir: str):
    """Return an Optuna-compatible objective callable from the YAML config.

    Supports both single-objective (``objective:`` key) and multi-objective
    (``objectives:`` key — returns a *tuple* of floats).
    """
    import optuna
    # Support running as script or as part of the optimizer package
    try:
        from optimizer.objectives import OBJECTIVE_REGISTRY
    except ImportError:
        from objectives import OBJECTIVE_REGISTRY

    # ---- Resolve objective spec(s) ----
    # Accept both singular and plural YAML keys
    if "objectives" in config:
        obj_specs = config["objectives"]  # list of dicts
        multi = True
    elif "objective" in config:
        obj_specs = [config["objective"]]  # wrap in list
        multi = False
    else:
        raise ValueError("Config must contain an 'objective' or 'objectives' key")

    # Validate and resolve objective functions
    obj_funcs = []
    for spec in obj_specs:
        name = spec["function"]
        if name not in OBJECTIVE_REGISTRY:
            raise ValueError(
                f"Unknown objective '{name}'. "
                f"Available: {list(OBJECTIVE_REGISTRY.keys())}"
            )
        obj_funcs.append({
            "func": OBJECTIVE_REGISTRY[name],
            "name": name,
            "reference": spec.get("reference", None),
            "kwargs": spec.get("kwargs", {}),
        })

    param_defs = config["parameters"]
    model_overrides = config.get("model", {}).get("extra_overrides", {})
    trial_timeout = config.get("model", {}).get("timeout", 0)
    cleanup_trials = config.get("model", {}).get("cleanup_trials", False)

    def objective(trial: "optuna.Trial"):
        # 1. Suggest parameter values
        overrides = dict(model_overrides)  # start with fixed overrides
        for param_name, param_cfg in param_defs.items():
            ptype = param_cfg.get("type", "float")
            if ptype == "float":
                val = trial.suggest_float(
                    param_name,
                    param_cfg["low"],
                    param_cfg["high"],
                    log=param_cfg.get("log", False),
                )
            elif ptype == "int":
                val = trial.suggest_int(
                    param_name,
                    param_cfg["low"],
                    param_cfg["high"],
                    log=param_cfg.get("log", False),
                )
            elif ptype == "categorical":
                val = trial.suggest_categorical(param_name, param_cfg["choices"])
            elif ptype == "array_float":
                # Per-element tuning: each specified element index gets its own
                # trial.suggest_float with a "PARAM[i]" override key.  Unspecified
                # indices keep their model.py default.
                elements = param_cfg.get("elements", {})
                arr_log = param_cfg.get("log", False)
                for idx_str, elem_cfg in elements.items():
                    idx = int(idx_str)
                    elem_name = f"{param_name}[{idx}]"
                    elem_val = trial.suggest_float(
                        elem_name,
                        elem_cfg["low"],
                        elem_cfg["high"],
                        log=elem_cfg.get("log", arr_log),
                    )
                    overrides[elem_name] = elem_val
                continue  # element-wise overrides already added; skip the scalar assignment below
            else:
                raise ValueError(f"Unknown parameter type '{ptype}' for {param_name}")
            overrides[param_name] = val

        # 2. Create trial directory
        trial_dir = os.path.join(base_result_dir, f"trial_{trial.number:05d}")
        os.makedirs(trial_dir, exist_ok=True)

        # 3. Run simulation
        try:
            results = run_trial_subprocess(
                overrides=overrides,
                model_script=model_script,
                result_dir=trial_dir,
                timeout=trial_timeout,
            )
        except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  [trial {trial.number}] FAILED: {e}")
            raise optuna.TrialPruned()

        # 4. Compute objective(s)
        errors = []
        display_texts = []
        for obj in obj_funcs:
            try:
                # Pass trial_dir so spatial objectives (e.g. organoid_size)
                # can read VTK files directly
                kw = dict(obj["kwargs"])
                kw["trial_dir"] = trial_dir
                raw_result = obj["func"](results, obj["reference"], **kw)
                error, display_text = _normalize_objective_result(raw_result)
                errors.append(error)
                display_texts.append(display_text)
            except Exception as e:
                print(f"  [trial {trial.number}] Objective '{obj['name']}' failed: {e}")
                raise optuna.TrialPruned()

        label = " | ".join(
            _format_objective_label(obj["name"], error, display_text)
            for obj, error, display_text in zip(obj_funcs, errors, display_texts)
        )
        trial.set_user_attr("display_texts", display_texts)
        print(f"  [trial {trial.number}] {label}")

        # 5. Optionally clean up trial output to save disk space
        if cleanup_trials:
            shutil.rmtree(trial_dir, ignore_errors=True)

        # Return scalar for single-objective, tuple for multi-objective
        return tuple(errors) if multi else errors[0]

    return objective, multi, obj_funcs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna parameter optimization for CellFoundry"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML optimization configuration file",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model.py (default: model.py in the same directory)",
    )
    parser.add_argument(
        "--result-dir", type=str, default=None,
        help="Base directory for trial results (default: optuna_results/)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent          # optimizer/ lives one level below project root
    model_script = args.model or str(project_root / "model.py")
    base_result_dir = args.result_dir or str(
        script_dir / config.get("model", {}).get("result_dir", "optuna_results")
    )
    os.makedirs(base_result_dir, exist_ok=True)

    # Study configuration
    study_cfg = config.get("study", {})
    study_name = study_cfg.get("name", "cellfoundry_optimization")
    storage = study_cfg.get("storage", f"sqlite:///{study_name}.db")
    n_trials = study_cfg.get("n_trials", 20)

    # Build objective(s)
    objective_fn, is_multi, objective_display_specs = make_objective(
        config,
        model_script,
        base_result_dir,
    )

    # ---- Directions ----
    # Multi-objective: study.directions is a list, e.g. [minimize, minimize]
    # Single-objective: study.direction is a string, e.g. minimize
    import optuna
    if is_multi:
        directions = study_cfg.get("directions", None)
        if directions is None:
            # Infer: one 'minimize' per objective
            n_obj = len(config.get("objectives", [config.get("objective")]))
            directions = ["minimize"] * n_obj
        direction_label = ", ".join(directions)
    else:
        direction = study_cfg.get("direction", "minimize")
        direction_label = direction

    # ---- Sampler ----
    sampler_name = study_cfg.get("sampler", None)
    seed = study_cfg.get("seed", None)

    if sampler_name is None:
        # Auto-select: NSGA-II for multi-objective, TPE for single
        if is_multi:
            sampler = optuna.samplers.NSampler(seed=seed) if hasattr(optuna.samplers, "NSampler") else optuna.samplers.NSGAIISampler(seed=seed)
            sampler_name = "NSGA-II"
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)
            sampler_name = "TPE"
    else:
        sampler_name_upper = sampler_name.upper()
        if sampler_name_upper == "TPE":
            sampler = optuna.samplers.TPESampler(seed=seed)
        elif sampler_name_upper == "RANDOM":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        elif sampler_name_upper == "CMAES":
            sampler = optuna.samplers.CmaEsSampler(seed=seed)
        elif sampler_name_upper in ("NSGA-II", "NSGAII", "NSGA2"):
            sampler = optuna.samplers.NSGAIISampler(seed=seed)
            sampler_name = "NSGA-II"
        elif sampler_name_upper in ("NSGA-III", "NSGAIII", "NSGA3"):
            sampler = optuna.samplers.NSGAIIISampler(seed=seed) if hasattr(optuna.samplers, "NSGAIIISampler") else optuna.samplers.NSGAIISampler(seed=seed)
            sampler_name = "NSGA-III"
        else:
            print(f"Unknown sampler '{sampler_name}', defaulting to {'NSGA-II' if is_multi else 'TPE'}")
            sampler = (optuna.samplers.NSGAIISampler(seed=seed) if is_multi
                       else optuna.samplers.TPESampler(seed=seed))

    # ---- Create / load study ----
    if is_multi:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=directions,
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            sampler=sampler,
            load_if_exists=True,
        )

    print(f"\n{'='*60}")
    print(f"  CellFoundry Optimization — {study_name}")
    print(f"  Trials: {n_trials} | Sampler: {sampler_name} | Direction: {direction_label}")
    print(f"  Multi-objective: {is_multi}")
    print(f"  Storage: {storage}")
    print(f"  Results: {base_result_dir}")
    print(f"{'='*60}\n")

    # Run optimization
    study.optimize(objective_fn, n_trials=n_trials)

    # ---- Report results ----
    if is_multi:
        # Pareto front
        pareto_trials = study.best_trials
        print(f"\n{'='*60}")
        print(f"  Pareto front: {len(pareto_trials)} trial(s)")
        for t in pareto_trials:
            display_texts = t.user_attrs.get("display_texts", [])
            labels = " | ".join(
                _format_objective_label(
                    obj["name"],
                    value,
                    _get_display_text(display_texts, idx),
                )
                for idx, (obj, value) in enumerate(zip(objective_display_specs, t.values))
            )
            print(f"  Trial #{t.number}: {labels}")
            for k, v in t.params.items():
                print(f"    {k}: {v}")
        print(f"{'='*60}\n")

        # Save Pareto params
        pareto_path = os.path.join(base_result_dir, "pareto_trials.json")
        pareto_data = [
            {"trial": t.number, "values": list(t.values), "params": t.params}
            for t in pareto_trials
        ]
        with open(pareto_path, "w") as f:
            json.dump(pareto_data, f, indent=2)
        print(f"Pareto front saved to {pareto_path}")
    else:
        best = study.best_trial
        best_display_texts = best.user_attrs.get("display_texts", [])
        best_display_text = best_display_texts[0] if best_display_texts else None
        best_label = _format_objective_label(
            objective_display_specs[0]["name"],
            best.value,
            best_display_text,
        )
        print(f"\n{'='*60}")
        print(f"  Best trial #{best.number}")
        print(f"  Objective: {best_label}")
        print(f"  Parameters:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        print(f"{'='*60}\n")

        best_params_path = os.path.join(base_result_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(best.params, f, indent=2)
        print(f"Best parameters saved to {best_params_path}")
        print(f"Re-run the model with: python model.py --overrides {best_params_path}")


if __name__ == "__main__":
    main()
