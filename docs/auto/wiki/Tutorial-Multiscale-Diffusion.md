# Tutorial: Multiscale Diffusion

Use multiscale diffusion when chemical species need smaller timesteps than cell movement, metabolism or other processes. CellFoundry advances the main model once per `TIME_STEP` and can take several smaller diffusion steps within that interval.

In this tutorial you will:

1. Run a three-species example and view it in ParaView.
2. Choose a diffusion timestep for each species.
3. Add a stationary cell that consumes and secretes species.
4. Enable diagnostic output and optionally add vascular sources.
5. Compare saved concentrations with numerical reference solutions.

An **ECM agent** represents a point in the extracellular matrix grid and stores local species concentrations. A **substep** is one diffusion update. A **submodel** runs a group of species through its substeps before control returns to the main model. Species indices start at zero.

## 1. Before you start

Follow [Tutorial: First Steps](Tutorial-First-Steps) to set up an environment with CUDA-enabled `pyflamegpu`. Run commands from the repository root, where `model.py` is located. These command examples use PowerShell.

Check your active Python environment:

```powershell
python -c "import pyflamegpu; print('pyflamegpu OK')"
```

You will need ParaView to inspect the exported fields. The numerical comparison tools use NumPy, Matplotlib and SciPy.

### Choose a grid

The examples below use these structural settings in `model.py`:

```python
N = 11
N_SPECIES = 3
N_CELL_TYPES = 3
BOUNDARY_COORDS = [50.0, -50.0, 50.0, -50.0, 50.0, -50.0]
```

This gives a 100 × 100 × 100 µm domain containing 11 × 11 × 11 ECM agents, with 10 µm spacing. There is an ECM agent at the origin.

Grid dimensions and species counts affect array sizes in Python and C++. Set structural values in `model.py` before running an example. Grid counts are calculated before JSON overrides are applied; changing `N` only in a JSON file does not rebuild those counts. JSON overrides also do not resize C++ declarations.

After changing the grid, check the corresponding constants:

```powershell
python check_hard_coded_values.py --exclude-variants
```

The checker reads literal settings in `model.py`, reports mismatches and offers to update them. The automatic check during startup scans files directly in the model folder and the selected variant folder, when a variant is active. Its `--fail-on-mismatch` option applies fixes automatically.

If you change `N_SPECIES` or `N_CELL_TYPES`, follow [Tutorial: Useful Tools](Tutorial-Useful-Tools) to resize the related arrays as well.

## 2. Run your first diffusion example

Run the supplied example without cells or a fibre network:

```powershell
python model.py --overrides configs/multiscale_diffusion.json --result-dir result_files/multiscale_demo
```

`--overrides` loads a JSON file of parameter values. `--result-dir` chooses the output folder. See [Tutorial: Parameter Overriding](Tutorial-Parameter-Overriding) for the general configuration workflow.

This example runs **20 main steps of 2 seconds**, for 40 seconds of simulated time, and saves a VTK file after every main step.

| Species | Initial concentration | Boundary conditions | Expected behavior |
|---|---:|---|---|
| 0 | 0 | 1 at +X, 0 at −X; no flux through the other faces | Spreads inward from +X while undergoing degradation |
| 1 | 0 | 0 at +X, 1 at −X; no flux through the other faces | Spreads inward from −X |
| 2 | 1 | No flux through all faces | Remains spatially uniform and decays; its diffusivity is zero |

At startup, the model prints the timestep plan. The first run after enabling a feature can pause while GPU functions compile. Initialization and main-step output follow compilation; subsequent runs can reuse cached kernels.

### View the result in ParaView

1. Open the `ecm_data_t*.vtk` file series in `result_files/multiscale_demo` and click **Apply**.
2. Apply **Threshold** to retain points with `is_corner = 0`, removing the boundary-corner markers from the displayed field.
3. Color by `concentration_species_0`, `concentration_species_1` or `concentration_species_2`.
4. Use a **Slice** through the center, or **Plot Over Line** along X, to inspect the concentration profile.
5. Play the time series. Keep the color range fixed across frames so color changes represent concentration changes.

The filename suffix is the main-step number: `ecm_data_t00001.vtk` is the field after step 1, at 2 seconds in this example. Main-model files contain completed main steps, rather than every diffusion substep. ParaView's automatically assigned series times may be frame indices; multiply the main-step number by `TIME_STEP` to obtain seconds.

## 3. Understand the species clocks

The first example uses:

```json
{
  "INCLUDE_DIFFUSION": true,
  "TIME_STEP": 2.0,
  "TIME_STEP_DIFFUSION": [0.05, 0.3, 2.0],
  "DIFFUSION_COEFF_MULTI": [100.0, 5.0, 0.0],
  "ECM_DEGRADATION_RATE_MULTI": [0.01, 0.0, 0.2],
  "DIFFUSION_MIN_SPACING": null,
  "DIFFUSION_CFL_SAFETY": 0.9,
  "DIFFUSION_MAX_SUBSTEPS": 1000000
}
```

Each entry in `TIME_STEP_DIFFUSION` is a **maximum requested timestep**, in seconds. The solver may choose a smaller value to remain stable and to fit an integer number of substeps into the main interval.

On the 10 µm grid, the plan is:

| Species | Requested maximum | Substeps per main step | Effective timestep | Execution |
|---|---:|---:|---:|---|
| 0 | 0.05 s | 40 | 0.05 s | Submodel |
| 1 | 0.3 s | 7 | 2/7 ≈ 0.285714 s | Submodel |
| 2 | 2 s | 1 | 2 s | Main model |

Seven steps of 0.3 seconds would exceed the 2-second interval. Seven steps of `2/7` seconds cover it exactly.

Species with the same substep count share a diffusion group. A group uses a submodel when it needs more than one substep. Counts and effective timesteps are fixed at startup for the entire run.

Cell metabolism, movement and vascular updates occur once per main step. Choosing a smaller diffusion timestep does not increase their update frequency.

### Parameter reference

| Parameter | Meaning |
|---|---|
| `INCLUDE_DIFFUSION` | Enables species transport. |
| `TIME_STEP` | Main interval in seconds; choose it to resolve metabolism, movement and other main-model processes. |
| `TIME_STEP_DIFFUSION` | One positive maximum timestep per species, each no greater than `TIME_STEP`. JSON `null` selects single-clock diffusion; Python uses `None`. |
| `DIFFUSION_COEFF_MULTI` | Nonnegative species diffusivities in µm²/s. |
| `ECM_DEGRADATION_RATE_MULTI` | Nonnegative first-order degradation rates in s⁻¹; zero disables degradation. |
| `HETEROGENEOUS_DIFFUSION` | Uses the local ECM agent values `D_sp` for spatially varying diffusivity. |
| `DIFFUSION_MIN_SPACING` | Three positive lower bounds on neighbor spacing in µm, or `null` to estimate them from the domain and prescribed boundary movement. |
| `DIFFUSION_CFL_SAFETY` | Safety factor for the explicit diffusion stability limit; strictly between 0 and 1. |
| `DIFFUSION_MAX_SUBSTEPS` | Maximum allowed substeps per group; planning fails if this is exceeded. |

`MULTISCALE_DIFFUSION` is derived from `INCLUDE_DIFFUSION` and `TIME_STEP_DIFFUSION`; you do not need to set it yourself. A species with zero diffusivity can still undergo degradation.

## 4. Set initial and boundary concentrations

`INIT_ECM_CONCENTRATION_VALS` specifies the initial concentration of every species throughout the ECM. Boundary arrays have one row per species and six columns in this order:

```text
+X, −X, +Y, −Y, +Z, −Z
```

For example, this row supplies concentration 1 at +X and concentration 0 at −X:

```json
[1.0, 0.0, -1.0, -1.0, -1.0, -1.0]
```

- A nonnegative entry prescribes a concentration on that face. In particular, **0 fixes the concentration to zero**.
- **−1 leaves the concentration unprescribed**. Missing grid neighbors then give zero flux through that face.
- `BOUNDARY_CONC_FIXED_MULTI` applies throughout the simulation.
- `BOUNDARY_CONC_INIT_MULTI` applies during the first main interval and is then released. Set all its entries to −1 if you only want the initial bulk field and fixed boundaries.
- At the intersection of prescribed faces, the greatest applicable concentration is used.

Boundary values are enforced throughout diffusion subcycling. They take precedence over a vascular concentration floor at the same ECM point.

## 5. Observe metabolism around one stationary cell

Create a copy of the first example:

```powershell
Copy-Item configs/multiscale_diffusion.json configs/multiscale_cell.json
```

In the copy, update or add the following entries, keeping the other settings from the first example:

```json
{
  "STEPS": 50,
  "TIME_STEP": 2.0,
  "TIME_STEP_DIFFUSION": [0.1, 0.5, 2.0],
  "DIFFUSION_COEFF_MULTI": [20.0, 10.0, 1.0],
  "ECM_DEGRADATION_RATE_MULTI": [0.0, 0.0, 0.0],
  "INIT_ECM_CONCENTRATION_VALS": [1.0, 0.0, 0.0],
  "INIT_ECM_SAT_CONCENTRATION_VALS": [1.0, 1.0, 1.0],
  "BOUNDARY_CONC_INIT_MULTI": [
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
  ],
  "BOUNDARY_CONC_FIXED_MULTI": [
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
  ],
  "INCLUDE_CELLS": true,
  "N_CELLS": 1,
  "CELL_RADIUS": [5.0, 5.0, 5.0],
  "CELL_SPEED_REF": [0.0, 0.0, 0.0],
  "BROWNIAN_MOTION_STRENGTH_FACTOR": [0.0, 0.0, 0.0],
  "ROTATIONAL_DIFFUSION_RATE": [0.0, 0.0, 0.0],
  "INCLUDE_CHEMOTAXIS": false,
  "INCLUDE_CHEMOKINESIS": false,
  "INCLUDE_DUROTAXIS": false,
  "INCLUDE_CELL_CELL_INTERACTION": false,
  "INIT_CELL_CONCENTRATION_VALS": [0.0, 0.0, 0.0],
  "INIT_CELL_CONSUMPTION_RATES": [2.0, 0.0, 0.0],
  "INIT_CELL_PRODUCTION_RATES": [0.0, 1.0, 0.1],
  "INIT_CELL_REACTION_RATES": [0.1, 0.0, 0.0],
  "DE_NOVO_PRODUCTION": [0, 1, 1],
  "CELL_CONSUMPTION_MULTIPLIER": [1.0, 1.0, 1.0],
  "CELL_PRODUCTION_MULTIPLIER": [1.0, 1.0, 1.0],
  "CELL_REACTION_MULTIPLIER": [1.0, 1.0, 1.0],
  "CELL_INIT_CONCENTRATION_MULTIPLIER": [1.0, 1.0, 1.0],
  "DEBUG_PRINTING": true,
  "DEBUG_PRINT_INTERVAL": 1
}
```

These entries belong inside the copied JSON object; do not append a second object. Inherited settings keep vascularization, cell division, fibre interactions and lumen disabled, and save every main step.

With `N_CELLS = 1`, the model initializes the cell at `(0, 0, 0)`. Zero motility and disabled mechanical interactions keep it there. The cell consumes species 0 and secretes species 1 and 2. `DE_NOVO_PRODUCTION = 1` allows secretion without drawing down an intracellular reservoir; `INIT_ECM_SAT_CONCENTRATION_VALS` supplies the secretion target concentrations. These settings provide an illustrative source-and-sink experiment.

Run the 100-second example:

```powershell
python model.py --overrides configs/multiscale_cell.json --result-dir result_files/multiscale_cell
```

Species 0 uses 20 substeps of 0.1 seconds, species 1 uses 4 substeps of 0.5 seconds, and species 2 advances once per main step.

Inspect a ParaView slice at Z = 0. Expect a nutrient dip near the cell, a spreading species-1 signal, and a more localized species-2 signal. Useful fixed color ranges are `[0.75, 1]`, `[0, 0.3]` and `[0, 0.5]`, respectively. Open `cells_t*.vtk` to check the cell position. Cell files include anchor markers; `radius > 1` selects the cell body for this example.

For automatically generated `no_anchor_cells_t*.vtk` files, set both `SAVE_PICKLE` and `SAVE_NO_ANCHOR_CELL_FILES` to `true`, keeping `SAVE_DATA_TO_FILE` enabled.

### Read the debug output

With `DEBUG_PRINTING = true` and `DEBUG_PRINT_INTERVAL = 1`, the model reports every submodel call:

```text
[DIFFUSION] main step 1 | t=0..2 s | submodel 0 START | species=[0] | 20 x 0.1 s
...
[DIFFUSION] main step 1 | t=0..2 s | submodel 0 END | species=[0] | 20 x 0.1 s
```

Accompanying lines show `C_sp` at the center ECM point and its +X neighbor. One representative CELL position and intracellular concentration sample is printed per reported main step. START values are sampled after cellular exchange and concentration preparation; END values are sampled after that group's diffusion updates.

Set `DEBUG_PRINT_INTERVAL` to 5 to report step 1 and every fifth main step. Set it to 0 to disable the submodel probes, or set `DEBUG_PRINTING` to `false` to disable them along with verbose GPU startup/progress messages. Probes copy agent data to the host, so reduce logging for performance measurements.

## 6. Add vascular sources

Vascularization requires a network file whose coordinates fit the simulation domain. Generate one for the 100 µm cube:

```powershell
python tools/generate_vascular_network.py --bounds 50 -50 50 -50 50 -50 --diameter 1 --density 0.002 --resolution 4 --branching-probability 0.08 --nucleation-faces 1 1 1 1 1 1 --seed 1 --output vascular_network_demo.pickle
```

In a copy of the stationary-cell configuration, set:

```json
{
  "INCLUDE_VASCULARIZATION": true,
  "VASC_NETWORK_FILE": "vascular_network_demo.pickle",
  "INIT_VASCULARIZATION_CONCENTRATION_VALS": [1.0, 0.0, 0.0],
  "INCLUDE_VASCULAR_CELL_RECRUITMENT": false
}
```

Save it as `configs/multiscale_vascular.json` and run:

```powershell
python model.py --overrides configs/multiscale_vascular.json --result-dir result_files/multiscale_vascular
```

The network supplies nutrient without supplying the two secreted signals. Nearby ECM points receive a minimum concentration, called a **vascular floor**, within `MAX_SEARCH_RADIUS_VASCULARIZATION`. Diffusion reapplies that floor during each substep. Vascular state and source locations update once per main step; vascular agents do not run inside the diffusion submodels.

Open `vasc_data_t*.vtk` alongside the ECM series to locate the sources. Supplying high vascular concentrations for every species can obscure the cell's secretion pattern, so choose source values according to the experiment. Enabling vascularization can require an initial GPU compilation pass.

## 7. Choose stable timesteps and spacing bounds

For explicit diffusion, smaller spacing and larger diffusivity require smaller timesteps. For species `s`, the planner uses:

```text
stability_limit[s] = safety / (2 * D_max[s] * sum(1 / h_min[axis]^2))
substeps[s] = ceil(TIME_STEP / min(requested_dt[s], stability_limit[s]))
effective_dt[s] = TIME_STEP / substeps[s]
```

`D_max` is an upper bound on diffusivity; `h_min` contains the minimum expected neighbor spacing along each grid direction. Zero diffusivity removes the diffusion stability restriction. If lumen is enabled, the bound includes `LUMEN_DIFFUSION_COEFF_MULTI` as well as the base coefficients.

### Static and moving domains

With `DIFFUSION_MIN_SPACING = null`, the planner estimates spacing from domain dimensions, main-step duration and prescribed normal boundary speeds. For each axis:

```text
duration = STEPS * TIME_STEP
final_width = initial_width + (positive_face_speed - negative_face_speed) * duration
minimum_spacing = min(initial_width, final_width) / (nodes_on_axis - 1)
```

A static domain uses its initial spacing. Uniform compression uses the smaller expected spacing at the end of the run. Predicted boundary crossing is rejected.

For local deformation that could bring neighbors closer together, supply a conservative bound:

```json
{
  "DIFFUSION_MIN_SPACING": [5.0, 5.0, 5.0]
}
```

Entries must be positive and cannot exceed the initial spacing. In the first diffusion example, reducing the bound from 10 to 5 µm changes the planned counts from `[40, 7, 1]` to `[54, 7, 1]`.

The movement-based estimate assumes uniform compression. Local mechanics can violate that assumption, so the solver checks actual exchange rates during updates. If a fixed timestep becomes unsafe, the run stops with a diagnostic. Investigate the deformation and restart with a suitable spacing bound or timestep; the solver does not adjust its clock during a run.

Stability alone does not establish accuracy. Compare results at smaller diffusion timesteps. If cell exchange or movement changes rapidly, also reduce `TIME_STEP`, because those processes remain on the main clock.

## 8. Run numerical validation

Start with CPU clock and model-wiring checks:

```powershell
python -m unittest manual_tests.test_multiscale_diffusion manual_tests.test_model_initialization_and_layers
python tools/validate_multiscale_diffusion.py
```

Without `--gpu`, the validation script runs NumPy reference and clock checks. To exercise the GPU kernels and export their small test grids:

```powershell
python tools/validate_multiscale_diffusion.py --gpu --save-data-to-file --results-dir result_files/multiscale_validation --output result_files/multiscale_validation/report.json
```

Successful execution finishes with a JSON report. Failed numerical checks raise an error. The isolated grids use production diffusion functions with array dimensions adapted in memory; they do not resize the repository's C++ files.

| Case directory | What it checks |
|---|---|
| `three_species_cosine` | Independent species clocks and a known smooth diffusion/decay solution |
| `heterogeneous_no_flux` | Conservation with spatially varying diffusivity and an impermeable sheet |
| `fnode_crowding` | Fibre-node messages reduce local diffusivity and affect transport |
| `dirichlet_linear` | A fixed linear concentration profile stays stationary |
| `fixed_clock_contracted_grid` | A spacing bound controls substeps while source exchange occurs once per main step |
| `small_main_step` | Fast-species agreement when the same diffusion updates run on a smaller main clock |
| `vascular_floor` | A supplied central concentration is maintained during diffusion and degradation |
| `initial_boundary_release` | Initial boundary concentrations are released after the first main interval |
| `temporal_dt_*`, `spatial_nodes_*` | Error decreases as timestep or grid spacing is refined |

Each exported case contains `diffusion.pvd`, `parameters.json` and `ecm_data_t*.vti`. The VTI series includes the initial field at step 0. Cosine and heterogeneous cases also include `comparison.vti` with reference fields and signed errors.

Open `diffusion.pvd` in ParaView for physical time in seconds. Numerical-test files use names `C_sp_0`, `C_sp_1`, etc., and `D_sp_0`, etc. Reference/error fields use names such as `reference_C_sp_0` and `error_C_sp_0`.

### Generate comparison plots

The plotting tool compares the planar example with an analytical solution, and numerical cases with analytical or matrix-exponential references. Create all required inputs in one directory tree:

```powershell
python model.py --overrides configs/multiscale_diffusion.json --result-dir result_files/multiscale_diffusion_review/full_model/diffusion
Copy-Item configs/multiscale_diffusion.json result_files/multiscale_diffusion_review/full_model/diffusion/parameters.json
python tools/validate_multiscale_diffusion.py --gpu --save-data-to-file --results-dir result_files/multiscale_diffusion_review/numerical --output result_files/multiscale_diffusion_review/report.json
python tools/plot_multiscale_diffusion_validation.py --results-dir result_files/multiscale_diffusion_review
```

Use a fresh result directory when changing the case or its number of steps, so old frames are not mixed into the comparison. The copied `parameters.json` must describe the planar run being plotted. Its analytical reference assumes the unmodified `configs/multiscale_diffusion.json` concentration settings: uniform initial fields, fixed opposing X faces, and no cell or vascular sources.

The `plots/` folder contains PNG/PDF comparisons, signed-error plots, sampled-value CSVs and `comparison_report.json`. Full-model probes are selected near the +X face, domain center and an interior point; their actual coordinates are recorded in the report. Plot titles use domain dimensions read from the VTK output.

Interpret comparisons according to their reference:

- **Continuum analytical solution:** differences include spatial and temporal discretization error.
- **Matrix exponential:** integrates the same spatial diffusion graph in time, allowing you to inspect temporal error on that grid.
- **Exact discrete update:** checks agreement with the expected numerical algorithm.

For the smooth validation cases, halving the timestep approximately halves temporal error. Halving grid spacing while reducing the timestep proportionally to spacing squared approximately quarters spatial error. Read your run's report for measured values; these checks do not determine the accuracy of an arbitrary biological configuration.

## 9. How the solver fits into the model

This section is useful when reading the code or adding diffusion to a custom variant. For variant setup, see [Tutorial: Model Variants](Tutorial-Model-Variants).

Within each main step, the model prepares source and boundary values, performs cellular exchange, and updates local diffusivity. It then advances each diffusion group, commits the resulting concentrations, and completes movement and output. Geometry and source constraints remain fixed during each submodel call.

| Source file | Role |
|---|---|
| `multiscale_ecm_diffusion_prepare.cpp` | Caches prescribed concentrations and applies boundary/source constraints before transport |
| `multiscale_ecm_diffusion_output.cpp` | Publishes the concentration and diffusivity snapshot used by the next substep |
| `multiscale_ecm_diffusion_step.cpp` | Advances selected species and checks the diffusion stability limit |
| `multiscale_ecm_diffusion_commit.cpp` | Copies final agent concentrations into the shared buffer used by other processes |
| `multiscale_ecm_velocity_output.cpp` | Publishes ECM velocity for vascular movement when moving boundaries and vascularization are enabled |

`model.py` defines groups, submodels and layer order. `helper_module.py` validates parameters and plans timesteps. See [C++ Function Reference](Function-Reference) for individual function inputs and outputs.

Each update exchanges concentration with six axial grid neighbors. A connection uses the harmonic mean of the two local diffusivities and the squared distance between points. Missing neighbors contribute zero flux. First-order degradation is applied with an exponential factor, followed by vascular floors and prescribed boundary concentrations.

The combined scheme is first order in time. It uses equal reference voxel volumes; it does not model dilution from changing voxel volumes or provide a general finite-volume treatment of a deformed mesh. Strong deformation needs additional spatial-model validation. Separate species clocks apply to independent diffusion/decay equations; intracellular reactions run through cell metabolism on the main clock.

For a custom layer schedule, call `g['_add_multiscale_diffusion_layers']()` after concentration and diffusivity preparation and before movement, when `g['MULTISCALE_DIFFUSION']` is true. Follow the default schedule's source and message preparation order.

## 10. Troubleshooting

| Symptom | What to check |
|---|---|
| A pause after the Python setup message | GPU compilation/cache loading happens before agent initialization. Enable `DEBUG_PRINTING` for startup and progress messages. |
| No diffusion submodel runs | Enable `INCLUDE_DIFFUSION` and supply a `TIME_STEP_DIFFUSION` list. A group with one substep runs in the main model. |
| No submodel probe output | Set `DEBUG_PRINTING = true` and `DEBUG_PRINT_INTERVAL` to a positive integer. |
| Predicted boundaries cross | Check boundary speeds and total duration; every domain width must stay positive. |
| Fixed diffusion stability bound exceeded | Investigate compression or increased diffusivity, then restart with a smaller timestep or spacing bound. |
| Invalid distance, coefficient or concentration | Inspect coincident ECM points, negative diffusivities and non-finite inputs. |
| Array-size errors after changing the grid/species count | Update structural settings in `model.py`, resize relevant arrays and run the checker. |
| Unexpectedly many substeps | Inspect diffusivity, minimum spacing and requested timesteps; the most restrictive value controls the plan. |
| No visible cell secretion | Check production rates, nonzero saturation targets, and either an intracellular reservoir or `DE_NOVO_PRODUCTION`. |
| Cell effects disappear after enabling vessels | Check vascular concentrations and source coverage; vascular floors can dominate local concentrations. |
| No VTK files | Enable `SAVE_DATA_TO_FILE`, check the save interval and result folder. Numerical-test exports also require `--gpu --save-data-to-file`. |
| Anchor-free cell files are missing | Enable `SAVE_PICKLE` as well as `SAVE_DATA_TO_FILE` and `SAVE_NO_ANCHOR_CELL_FILES`. |
