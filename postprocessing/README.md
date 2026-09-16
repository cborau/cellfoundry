# Diffusion diagnostics

Use `plot_diffusion_results.py` to inspect concentration changes in saved
CellFoundry VTK files. It needs NumPy, pandas and Matplotlib from the project's
Python environment; the simulation and GPU do not need to be running.

## Start with a run

Run from the repository root, replacing `my_run` with your results folder:

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --show
```

The script discovers `ecm_data_t*.vtk`, `cells_t*.vtk` and `vasc_data_t*.vtk`,
and all `concentration_species_*` scalars. It samples the nearest physical point
to `(0, 0, 0)` in each dataset at each saved step, then plots:

- **Traces:** concentration at the selected probe over time, grouped by species.
- **Summary:** mean concentration over each population, with its minimum-to-maximum
  range shaded. These are unweighted point statistics, not total chemical mass.

Figures and CSVs are saved in `result_files/my_run/diffusion_plots/`. Omit `--show`
to save without opening windows. Use `--outdir` to choose another destination and
`--formats png pdf` to export both formats. Repeating a command updates the files
for its selected plots.

ECM corner markers and cell anchor markers are excluded automatically. All actual
agents, including those marked dead, remain in the population statistics.
Each VTK snapshot is read separately so memory does not grow with the full
point population across all saved steps.

Inspect available datasets, saved steps and scalar names without writing files:

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --list
```

The results folder must contain the VTK files directly. With no `--results-dir`,
the default is the repository's `result_files/`, even when launched elsewhere.
Explicit relative paths are resolved from your working directory.

## Choose species, populations and probes

For nutrient depletion near a cell, compare the ECM at the center and an offset:

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --datasets ecm --species 0 1 --probe 0 0 0 --probe 20 0 0 --show
```

- `--datasets ecm cells vascular` selects populations; omitted populations need
  no corresponding files. By default, all available populations are included.
- `--species 0 2` selects species by zero-based index. Alternatively,
  `--variables concentration_species_0 damage` selects named point scalars.
  Each selected variable must exist in at least one selected dataset.
- Repeat `--probe X Y Z` to compare locations. Sampling uses the nearest point
  without interpolation. The trace CSV records its actual coordinates, ID and
  distance from the requested probe. Two probes may select the same point.
- `--subplot-mode by_dataset` puts each population in a separate panel, with a
  line per selected scalar/probe. This helps when intracellular and extracellular
  concentrations have very different ranges.
- `--plots traces` or `--plots summary` selects one plot type. Summary CSVs are
  always exported and include counts of negative and non-finite values.

### Follow a location or an agent

`--nearest-mode per_time` (the default) selects the nearest point independently
at each saved step. This samples a fixed location; the selected agent can change.

`--nearest-mode fixed_id` selects the nearest point once and follows it. Cell and
vascular outputs use their exported `id`, even if the VTK point order changes.
For ECM outputs without an `id`, it follows the original VTK point index, which
requires stable ordering. The reference is the first selected saved step;
`--reference-step 10` selects another saved step.

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --datasets cells --probe 0 0 0 --nearest-mode fixed_id --show
```

To select a known agent directly, use `--datasets cells --agent-id 1332`, replacing
1332 with an ID from your output. `--point-id 0` instead selects the original
zero-based VTK point index. These selectors are alternatives to `--probe`.
Missing tracked agents produce gaps, without switching to a different agent.

## Compare spatial profiles

Profiles show how the ECM concentration varies along a grid line:

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --datasets ecm --plots profiles --profile-axis x --probe 0 0 0 --show
```

By default, the first and last selected saved steps are compared. Use
`--profile-steps 1 10 50` to choose specific saved steps, or
`--plots traces summary profiles` to generate all plots together.

For an X profile, the nearest actual Y/Z grid line to the probe is selected.
The legend gives the sampled Y/Z coordinates; the entire X extent is plotted.
Use `--profile-axis y` or `z` for another direction. Repeated probes selecting
the same line produce a single profile. This mode requires an axis-aligned ECM
grid. For arbitrary line segments, see `extract_scalar_profiles_vtk.py`.

## View concentration maps over time

Add `--plots maps` for 2D ECM concentration maps:

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --plots maps --show
```

By default, maps include **all species, X=0/Y=0/Z=0, and all saved steps**.
There is one figure per species, with **planes in rows and saved steps in
columns**. Each species keeps the same color scale across all planes and times,
using the ECM minimum and maximum over the selected saved steps.

Choose species, planes and exact saved steps:

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --plots maps --species 0 2 --planes z=0 x=20 --steps 1 10 50 --time-step 2 --show
```

Plane coordinates use the simulation's length units. Specify at most one plane
per direction. For example, `--planes z=0` draws only the XY plane at Z=0;
`--planes x=20 y=-10 z=0` draws three planes. If a requested coordinate lies
between saved grid planes, the nearest plane is used at each step and the
subplot title reports its actual coordinate. Out-of-domain requests are errors.
Maps require a complete, uniformly spaced, axis-aligned grid in each plane.

To sample every fifth available saved file, replace `--steps ...` with `--every 5`:

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --plots maps --planes z=0 --every 5 --smooth --show
```

- Smoothing is **off by default**: every ECM agent in the plane supplies one
  pixel of the map, displayed as a solid block without interpolation.
- `--smooth` uses bilinear display interpolation between neighboring pixels.
  It does not alter concentrations or the summary CSV. `--no-smooth` explicitly
  selects the original grid display. Non-finite values appear gray.
- Long runs are split into pages of at most **six time columns**, with every
  selected step included. Use `--map-columns 10` to change this; setting it to
  the number of selected steps places all times on one figure per species.
- `--map-layout by_plane` is an alternative: one figure per plane, species in
  rows and time in columns.
- Combine modes with `--plots traces summary profiles maps`. Explicit `--steps`
  and the existing range/`--every` options apply to all selected plot types.

Map figures are saved alongside the other plots. `diffusion_maps.json` records
the selected species, steps, requested/actual planes, shared color limits and
page filenames. It is an output report; no input JSON config is required.

## Time and larger runs

The filename suffix is the simulation **main step**, not a sequential output
index. For example, with `TIME_STEP = 2`, `ecm_data_t00010.vtk` represents 20 s,
regardless of how many steps were skipped between saves.

```powershell
python postprocessing/plot_diffusion_results.py --results-dir result_files/my_run --config configs/my_run.json --start-step 10 --end-step 50 --every 2
```

`--config` reads `TIME_STEP` from your run's JSON overrides. If omitted, the script
checks for `parameters.json` inside the results folder. `--time-step 2` supplies
the value directly and takes precedence over JSON. Without either source, the
horizontal axis shows main steps. The first saved frame is not necessarily the
initial condition.

`--start-step` and `--end-step` bound the saved steps inclusively; `--every 2`
then reads every second available file in that range, separately per dataset.
Alternatively, `--steps 1 10 50` reads exactly those saved main steps in time order.
`--steps` and `--every` are mutually exclusive.
Reference/profile steps must be present in this selection. Summaries and reported
negative/non-finite counts apply only to the files read. Those counts are
point-samples across time, so the same point can contribute at several steps.
Non-finite values are excluded from min/mean/max and remain visible in the counts.

See `python postprocessing/plot_diffusion_results.py --help` for all options.
