from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

try:
    import pandas as pd
except ImportError:
    pd = None
    
# Press f5 for debugging. The script will load the specified pickle file, check for the given key, and print out information about it. You can inspect the variable 'target' in the debugger to see its contents.


# --- Paths ---
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- Dummy fallback for missing classes ---
class DummyModelParameterConfig:
    pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "helper_module" and name == "ModelParameterConfig":
            return DummyModelParameterConfig
        return super().find_class(module, name)


def load_pickle(path: pathlib.Path):
    with path.open("rb") as f:
        return SafeUnpickler(f).load()


def main():
    parser = argparse.ArgumentParser(description="Inspect a specific key in a pickle file")
    parser.add_argument("pickle_path", type=pathlib.Path, help="Path to .pkl/.pickle file")
    parser.add_argument("--key", type=str, required=True, help="Key to inspect")
    parser.add_argument("--expect-df", action="store_true", help="Expect a pandas DataFrame")

    args = parser.parse_args()

    if not args.pickle_path.exists():
        raise FileNotFoundError(f"Pickle not found: {args.pickle_path}")

    data = load_pickle(args.pickle_path)

    print(f"Loaded: {args.pickle_path}")
    print(f"Top-level type: {type(data)}")

    if not hasattr(data, "keys"):
        raise TypeError("Top-level object is not a mapping (dict-like)")

    if args.key not in data:
        raise KeyError(f"Key '{args.key}' not found. Available keys: {list(data.keys())}")

    target = data[args.key]

    print(f"\n--- Target: {args.key} ---")
    print(f"Type: {type(target)}")

    if pd is not None and isinstance(target, pd.DataFrame):
        print(f"Shape: {target.shape}")
        print(f"Columns: {list(target.columns)}")

    if args.expect_df and (pd is None or not isinstance(target, pd.DataFrame)):
        raise TypeError(f"Expected DataFrame but got {type(target)}")

    # 🔴 DEBUG STOP HERE
    breakpoint()

    # (Optional fallback if not using debugger)
    return target


if __name__ == "__main__":
    main()