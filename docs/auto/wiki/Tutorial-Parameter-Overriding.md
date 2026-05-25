# Tutorial: Parameter Overriding

This tutorial explains how parameter overriding works in CellFoundry, including precedence order and a minimal example.

## Why Use Overrides

Use overrides when you want to:

- run quick experiments without editing `model.py`
- keep base defaults unchanged
- run repeatable scenarios from JSON files
- combine a variant baseline with experiment-specific tweaks

## Override Priority (Highest Wins)

CellFoundry applies parameters in this order:

1. `model.py` defaults
2. `variant.PARAMS` from `variants/<name>/__init__.py` (if `--variant` is used)
3. JSON passed with `--overrides`

So JSON always has the final word.

## Basic CLI Usage

```bash
python model.py --overrides configs/organoid_test.json
```

With a variant:

```bash
python model.py --variant radial_glia --overrides configs/organoid_test.json
```

In the second command, values in `configs/organoid_test.json` override both base defaults and `radial_glia` variant defaults.

## Minimal Example

Create a file (for example `configs/minimal_run.json`) with only the parameters you want to change:

```json
{
  "STEPS": 200,
  "TIME_STEP": 60.0,
  "SAVE_DATA_TO_FILE": true,
  "SAVE_PICKLE": true,
  "INCLUDE_DIFFUSION": true,
  "INCLUDE_CELLS": true,
  "INCLUDE_CELL_CYCLE": false,
  "INCLUDE_FIBRE_NETWORK": false,
  "INCLUDE_FOCAL_ADHESIONS": false,
  "INCLUDE_VASCULARIZATION": false,
  "INCLUDE_LUMEN": false
}
```

Then run:

```bash
python model.py --overrides configs/minimal_run.json
```

## Result Directory Override

You can also change output location from CLI:

```bash
python model.py --overrides configs/minimal_run.json --result-dir result_files/minimal_demo
```

## Scalar, Full-List, and Indexed Overrides

For list-like model parameters (for example per-cell-type arrays such as `CELL_SPEED_REF`), there are three useful patterns.

### 1. Scalar override (broadcast to all entries)

```json
{
  "CELL_SPEED_REF": 0.002
}
```

If `CELL_SPEED_REF` is a list in `model.py`, this is broadcast to all cell types.

### 2. Full-list override

```json
{
  "CELL_SPEED_REF": [0.002, 0.0015, 0.001]
}
```

Use this when you want explicit values for each cell type.

### 3. Indexed override (single entry only)

```json
{
  "CELL_SPEED_REF[0]": 0.003
}
```

This updates only index `0` of the existing list and leaves the other entries unchanged.


## Important Notes for Indexed Overrides

- The base parameter must already exist and be list-like in the model namespace.
- If the index is out of range, the override is ignored with a warning.
- Avoid mixing a full-list key and an indexed key for the same parameter in one JSON file unless you are very intentional about key order.

Example (not recommended):

```json
{
  "CELL_SPEED_REF": [0.002, 0.002, 0.002],
  "CELL_SPEED_REF[0]": 0.003
}
```

This gives type 0 a special value while keeping other types at the broadcast baseline.

## Related Tutorials

- [Tutorial: First Steps](Tutorial-First-Steps)
- [Tutorial: Model Variants](Tutorial-Model-Variants)
- [Tutorial: Parameter Optimization](Tutorial-Param-Optimization)
