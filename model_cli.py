"""CellFoundry command-line interface; intentionally uses only the standard library.

Keep this independent of model.py, variants and pyflamegpu so --help and argument
errors can exit before model construction, CUDA initialization or file writes.
"""

import argparse


def parse_model_args(args=None):
    """Parse arguments excluding the script name (default: sys.argv[1:])."""
    parser = argparse.ArgumentParser(
        description="Run the CellFoundry model, optionally with a model variant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""Examples:
  python model.py --variant radial_glia --result-dir "results/radial_glia_test"
  python model.py --variant organoid --overrides configs/my_overrides.json

Parameter precedence: JSON overrides > variant PARAMS > model.py defaults.
Model parameters (e.g. STEPS) go in the overrides JSON, not individual CLI flags.
Initial BOUNDARY_COORDS and structural settings (N, grid/array extents, species
counts) are core-controlled. Differing variant/JSON overrides are rejected.
Change these settings in model.py and synchronize RTC constants before running.
See docs/auto/wiki/Tutorial-Model-Variants.md for details.
""",
    )
    parser.add_argument(
        "--variant", metavar="NAME",
        help="variant under variants/<NAME>/ (default: generic core model)",
    )
    parser.add_argument(
        "--overrides", metavar="JSON",
        help="JSON object of parameter overrides, applied after variant PARAMS",
    )
    parser.add_argument(
        "--result-dir", metavar="DIR",
        help="results directory (default: result_files beside model.py); "
             "relative paths use the current working directory",
    )
    return parser.parse_args(args)
