# Tutorial: First Steps

This tutorial covers the minimum practical workflow to run CellFoundry for the first time.

## 1. Activate Environment

Example with Conda:

```powershell
conda activate flamegpu_py310
```

Optional quick check:

```powershell
python -c "import pyflamegpu; print('pyflamegpu OK')"
```

## 2. Run a Minimal Simulation

The idea is to run only the essentials:

- cells enabled
- diffusion enabled
- migration active
- no fibre network, no focal adhesions, no lumen, no vascularization
- save VTK and pickle outputs

Manually set these variables in model.py or create a small override file such as `configs/first_steps_minimal.json`:

```json
{
  "STEPS": 200,
  "TIME_STEP": 60.0,
  "VISUALISATION": false,
  "SHOW_PLOTS": false,
  "SAVE_DATA_TO_FILE": true,
  "SAVE_PICKLE": true,
  "INCLUDE_CELLS": true,
  "INCLUDE_DIFFUSION": true,
  "INCLUDE_CELL_CELL_INTERACTION": true,
  "INCLUDE_CELL_CYCLE": false,
  "INCLUDE_FIBRE_NETWORK": false,
  "INCLUDE_FOCAL_ADHESIONS": false,
  "INCLUDE_VASCULARIZATION": false,
  "INCLUDE_LUMEN": false
}
```

If manually set, run:

```powershell
python model.py 
```
If using an overriding file, run:

```powershell
python model.py --overrides configs/first_steps_minimal.json
```

Outputs are written to `result_files/` by default.

## 3. Postprocess Results

Two common options:

1. Open VTK files directly in ParaView.
2. Use plotting/report scripts from `postprocessing/`, for example:

```powershell
python postprocessing/report_cell_population.py
python postprocessing/plot_diffusion_results.py
```

Note: cell files contain not only cell position and parameters, but also their anchor points, which increases size and complexity of the files. If you want a simplified point-only cell representation (without anchors), run:

```powershell
python tools/remove_anchors_from_cell_vtks.py --input-dir result_files
```

or just this if files were saved in the default folder:

```powershell
python tools/remove_anchors_from_cell_vtks.py 
```

This generates files like `no_anchor_cells_tXXXX.vtk` that are easier to inspect quickly (and lighter).

## 4. Important Note on Hard-Coded Values

Some global values are structural and affect variable sizes/shapes across Python and C++ code, especially:

- `N_CELLS`
- `N`
- `N_SPECIES`

You can change them, but do it carefully.

When changing dimensional parameters such as `N_SPECIES` or `N_CELL_TYPES`:

1. Run `tools/resize_array_variables.py` to resize related arrays/constants.
2. Run `check_hard_coded_values.py` to detect mismatches.

`model.py` already calls `check_hard_coded_values.py` automatically and will report errors/user feedback if inconsistencies are found.

For details and examples, see [Tutorial: Useful Tools](Tutorial-Useful-Tools).

## 5. Next Options

Once the minimal run works, typical next steps are:

- parameter overrides with JSON files: [Tutorial: Parameter Overriding](Tutorial-Parameter-Overriding)
- biological variants (assay presets and custom logic): [Tutorial: Model Variants](Tutorial-Model-Variants)
- automated calibration with Optuna: [Tutorial: Parameter Optimization](Tutorial-Param-Optimization)
