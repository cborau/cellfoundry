#!/usr/bin/env python
"""
Optuna-based parameter optimization runner for CellFoundry.

Usage
-----
    python optimize.py --config optuna_config.yaml

Each trial:
  1. Optuna suggests values for the parameters listed in the YAML config.
  2. A JSON override file is written to a temporary trial directory.
  3. ``model.py`` is launched as a subprocess with ``--overrides`` and
     ``--result-dir`` flags (clean GPU memory between trials).
  4. The resulting pickle is loaded and passed to the chosen objective
     function to compute a scalar error.
  5. The error is returned to Optuna for minimization.

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
        return yaml.safe_load(f)


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
        raise FileNotFoundError(
            f"Expected pickle not found at {pickle_path}. "
            f"model.py stdout:\n{proc.stdout[-500:]}"
        )

    return _load_pickle(pickle_path)


# ---------------------------------------------------------------------------
# Objective wrapper
# ---------------------------------------------------------------------------

def make_objective(config: dict, model_script: str, base_result_dir: str):
    """Return an Optuna-compatible objective callable from the YAML config."""
    import optuna
    from objectives import OBJECTIVE_REGISTRY

    obj_cfg = config["objective"]
    obj_func_name = obj_cfg["function"]
    if obj_func_name not in OBJECTIVE_REGISTRY:
        raise ValueError(
            f"Unknown objective '{obj_func_name}'. "
            f"Available: {list(OBJECTIVE_REGISTRY.keys())}"
        )
    obj_func = OBJECTIVE_REGISTRY[obj_func_name]
    reference_path = obj_cfg.get("reference", None)
    obj_kwargs = obj_cfg.get("kwargs", {})
    param_defs = config["parameters"]
    model_overrides = config.get("model", {}).get("extra_overrides", {})
    trial_timeout = config.get("model", {}).get("timeout", 0)
    cleanup_trials = config.get("model", {}).get("cleanup_trials", False)

    def objective(trial: "optuna.Trial") -> float:
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

        # 4. Compute objective
        try:
            error = obj_func(results, reference_path, **obj_kwargs)
        except Exception as e:
            print(f"  [trial {trial.number}] Objective evaluation failed: {e}")
            raise optuna.TrialPruned()

        print(f"  [trial {trial.number}] Error = {error:.6f}")

        # 5. Optionally clean up trial output to save disk space
        if cleanup_trials:
            shutil.rmtree(trial_dir, ignore_errors=True)

        return error

    return objective


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
    model_script = args.model or str(script_dir / "model.py")
    base_result_dir = args.result_dir or str(
        script_dir / config.get("model", {}).get("result_dir", "optuna_results")
    )
    os.makedirs(base_result_dir, exist_ok=True)

    # Study configuration
    study_cfg = config.get("study", {})
    study_name = study_cfg.get("name", "cellfoundry_optimization")
    storage = study_cfg.get("storage", f"sqlite:///{study_name}.db")
    n_trials = study_cfg.get("n_trials", 20)
    direction = study_cfg.get("direction", "minimize")

    # Sampler
    import optuna
    sampler_name = study_cfg.get("sampler", "TPE").upper()
    seed = study_cfg.get("seed", None)
    if sampler_name == "TPE":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "RANDOM":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "CMAES":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    else:
        print(f"Unknown sampler '{sampler_name}', defaulting to TPE")
        sampler = optuna.samplers.TPESampler(seed=seed)

    # Create / load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )

    objective_fn = make_objective(config, model_script, base_result_dir)

    print(f"\n{'='*60}")
    print(f"  CellFoundry Optimization — {study_name}")
    print(f"  Trials: {n_trials} | Sampler: {sampler_name} | Direction: {direction}")
    print(f"  Storage: {storage}")
    print(f"  Results: {base_result_dir}")
    print(f"{'='*60}\n")

    # Run optimization
    study.optimize(objective_fn, n_trials=n_trials)

    # Report best trial
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Best trial #{best.number}")
    print(f"  Error: {best.value:.6f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # Save best params as JSON for easy re-use
    best_params_path = os.path.join(base_result_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Best parameters saved to {best_params_path}")
    print(f"Re-run the model with: python model.py --overrides {best_params_path}")


if __name__ == "__main__":
    main()
