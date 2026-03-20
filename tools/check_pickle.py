from __future__ import annotations

import pathlib
import pickle
import sys
from collections.abc import Mapping, Sequence

try:
    import pandas as pd
except ImportError:
    pd = None


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PICKLE_PATH = PROJECT_ROOT / "result_files" / "output_data_0.pickle"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyModelParameterConfig:
    pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "helper_module" and name == "ModelParameterConfig":
            return DummyModelParameterConfig
        return super().find_class(module, name)


def print_scalar(value, indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}{repr(value)}")


def print_dataframe(df, indent: int = 0, n_rows: int = 3) -> None:
    prefix = " " * indent
    print(f"{prefix}DataFrame shape={df.shape}")
    print(f"{prefix}Columns: {list(df.columns)}")
    print(f"{prefix}Head:")
    head_str = df.head(n_rows).to_string()
    for line in head_str.splitlines():
        print(f"{prefix}  {line}")
    print(f"{prefix}Tail:")
    tail_str = df.tail(n_rows).to_string()
    for line in tail_str.splitlines():
        print(f"{prefix}  {line}")


def summarize_value(
    value,
    indent: int = 0,
    max_depth: int = 2,
    max_items: int = 5,
    dataframe_rows: int = 3,
) -> None:
    prefix = " " * indent

    if max_depth < 0:
        print(f"{prefix}...")
        return

    if isinstance(value, (int, float, str, bool, type(None))):
        print_scalar(value, indent=indent)
        return

    if pd is not None and isinstance(value, pd.DataFrame):
        print_dataframe(value, indent=indent, n_rows=dataframe_rows)
        return

    if isinstance(value, Mapping):
        print(f"{prefix}{type(value).__name__} ({len(value)} keys)")
        for i, (key, subvalue) in enumerate(value.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({len(value) - max_items} more keys)")
                break
            print(f"{prefix}  [{repr(key)}] ->")
            summarize_value(
                subvalue,
                indent=indent + 4,
                max_depth=max_depth - 1,
                max_items=max_items,
                dataframe_rows=dataframe_rows,
            )
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        print(f"{prefix}{type(value).__name__} (len={len(value)})")
        n_show = min(len(value), max_items)
        for i in range(n_show):
            print(f"{prefix}  [{i}] ->")
            summarize_value(
                value[i],
                indent=indent + 4,
                max_depth=max_depth - 1,
                max_items=max_items,
                dataframe_rows=dataframe_rows,
            )
        if len(value) > max_items:
            print(f"{prefix}  ... ({len(value) - max_items} more items)")
        return

    if hasattr(value, "shape"):
        print(f"{prefix}{type(value).__name__} shape={value.shape}")
        return

    if hasattr(value, "__dict__"):
        print(f"{prefix}{type(value).__name__}")
        attrs = vars(value)
        if not attrs:
            print(f"{prefix}  <no instance attributes>")
            return
        for i, (key, subvalue) in enumerate(attrs.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({len(attrs) - max_items} more attributes)")
                break
            print(f"{prefix}  .{key} ->")
            summarize_value(
                subvalue,
                indent=indent + 4,
                max_depth=max_depth - 1,
                max_items=max_items,
                dataframe_rows=dataframe_rows,
            )
        return

    print(f"{prefix}{type(value)}")


def load_pickle(pickle_path: pathlib.Path):
    with pickle_path.open("rb") as file:
        return SafeUnpickler(file).load()


def main() -> None:
    pickle_path = DEFAULT_PICKLE_PATH

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    data = load_pickle(pickle_path)

    print(f"Loaded: {pickle_path}")
    print(f"Top-level type: {type(data)}")

    if hasattr(data, "keys"):
        keys = list(data.keys())
        print(f"Top-level keys ({len(keys)}):")
        for key in keys:
            print(f"\n{key}:")
            summarize_value(data[key], indent=2, max_depth=2, max_items=10, dataframe_rows=3)
    else:
        print("\nTop-level content:")
        summarize_value(data, indent=2, max_depth=2, max_items=10, dataframe_rows=3)


if __name__ == "__main__":
    main()