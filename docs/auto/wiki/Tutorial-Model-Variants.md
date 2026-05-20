# Tutorial: Model Variants

This tutorial explains how to use the **variant system** to simulate different biological problems within the same CellFoundry codebase, without branching the repository or duplicating files.

---

## Motivation

CellFoundry is a general-purpose agent-based simulation platform.  Different biological problems (organoid growth, tumour invasion, fibre remodelling, wound healing, …) share most of the simulation infrastructure but differ in:

- Parameter values (cell speed, adhesion strength, domain size, etc.)
- Agent function logic (e.g. a custom `cell_cycle.cpp` with asymmetric division)
- Layer execution order (e.g. insert a differentiation step between L3 and L7)
- Variant-specific environment properties registered on the FLAMEGPU2 model

The variant system lets you encode all of these differences in a single Python package (`variants/<name>/`) plus any modified `.cpp` files, leaving `model.py` untouched.

---

## Directory layout

```
cellfoundry/
├── model.py                          ← base model (never edited per-variant)
├── cell_cycle.cpp                    ← base agent functions
├── cell_move.cpp
│   ...
├── variants/
│   └── organoid/
│       ├── __init__.py               ← organoid variant module
│       └── cell_cycle.cpp            ← organoid-specific override
└── optimizer/
    ├── optuna_config_organoid_variant.yaml
    └── ...
```

Each variant lives in its own subfolder `variants/<name>/`.  The entry point is `variants/<name>/__init__.py`.  The same folder holds only the `.cpp` files that differ from the base model.

> **Path note**: `.cpp` file paths in the `FILES` dict are always relative to the project root (the folder containing `model.py`), e.g. `"variants/organoid/cell_cycle.cpp"`.  

---

## Anatomy of a variant module

A variant module (`__init__.py`) exports up to four objects.  All are optional.

```python
# variants/my_variant/__init__.py

PARAMS: dict        # parameter overrides  (applied before JSON --overrides)
FILES:  dict        # *_file variable redirections (applied after all parameters)

def configure_globals(g: dict) -> None: ...   # inject new global flags
def configure_layers(model, g: dict) -> None: # full layer sequence for this variant
```

### `PARAMS`

A plain dict mapping parameter names to values.  Any key that exists as a global variable in `model.py` can be overridden.  Scalars are broadcast to lists automatically (same behaviour as `--overrides` JSON).

```python
PARAMS = {
    "ORGANOID_ASSAY": True,
    "N_CELLS": 13,
    "CELL_RADIUS": [20.0, 20.0, 20.0],     # explicit list
    "CELL_SPEED_REF": 0.006,                # scalar → broadcast to all types
    "CYCLE_PHASE_G1_DURATION": [12000.0, 24000.0, 36000.0],
}
```

**Priority**: `--overrides` JSON always wins over `PARAMS`.  This means the optimizer can tune any variant parameter without editing the variant file.

### `FILES`

A dict mapping the `*_file` variable names in `model.py` to variant-specific `.cpp` paths (relative to the project root).

```python
FILES = {
    "cell_cycle_file": "variants/organoid/cell_cycle.cpp",
    "cell_move_file":  "variants/organoid/cell_move.cpp",
}
```

Only list files that actually differ from the base model.  Unmentioned functions use the base `.cpp` as usual.

### `configure_globals(g)`

Called after `PARAMS` and `FILES` are applied but **before** `model.py` builds the FLAMEGPU2 `ModelDescription`.  Use it to inject global flags that don't exist in the base model and therefore cannot go into `PARAMS`.

```python
def configure_globals(g: dict) -> None:
    # Inject a new flag that the custom cell_cycle.cpp relies on.
    g["CONTACT_INHIBIT_SIGMA"] = 1.5   # [kPa]
    g["CONTACT_INHIBIT_FACTOR"] = 3.0
```

Values set here are available as `g["KEY"]` inside `configure_layers`.

### `configure_layers(model, g)`

**This function owns the complete layer sequence** when defined.  If it is present in the variant, `model.py` does *not* call `_build_default_layers()` itself — the variant is responsible for the full L0-L8 stack plus any additions.

Call `g['_build_default_layers']()` to include the standard sequence, then add any variant-specific layers after it.  To insert layers *between* default layers, copy the relevant portion of `_build_default_layers` inline and add your layers at the correct position.

```python
def configure_layers(model, g: dict) -> None:
    # --- Register variant-specific environment properties ---
    _env = g.get("env")
    if _env is not None:
        try:
            _env.newPropertyFloat("CONTACT_INHIBIT_SIGMA",
                                  g.get("CONTACT_INHIBIT_SIGMA", 1.5))
            _env.newPropertyFloat("CONTACT_INHIBIT_FACTOR",
                                  g.get("CONTACT_INHIBIT_FACTOR", 3.0))
        except Exception:
            pass  # already registered (safe guard for re-runs)

    # --- Build the full layer sequence ---
    # Option A: use defaults unchanged (append-style — layers added after L8
    # are visible to model.py's L7 result only from the next step).
    g['_build_default_layers']()

    # Option B: insert a layer between L3 and L4 (see section below).
```

If `configure_layers` is **not** defined, `model.py` runs `_build_default_layers()` automatically.

---

## Running a variant

```bash
# Basic run
python model.py --variant organoid

# With additional parameter overrides (JSON wins over variant PARAMS)
python model.py --variant organoid --overrides configs/organoid_paper.json

# Specify result directory
python model.py --variant organoid --result-dir result_files/organoid_run_01
```

If the variant name is not found in `variants/`, the model exits with a clear error listing available variants.

---

## Using variants with the optimizer

Add `model.variant: <name>` to any Optuna YAML config.  The optimizer forwards `--variant <name>` to every trial subprocess automatically.

```yaml
# optimizer/optuna_config_organoid_variant.yaml

model:
  variant: organoid          # ← loads variants/organoid/__init__.py for every trial
  extra_overrides:
    DEBUG_PRINTING: false
  timeout: 0
  cleanup_trials: false

parameters:
  CELL_SPEED_REF:
    type: float
    low: 0.0001
    high: 0.02
    log: true
  # ...
```

The `parameters:` block is the search space; it overrides only the specific values being tuned while the variant provides all other calibrated defaults.

```bash
python -m optimizer.optimize --config optimizer/optuna_config_organoid_variant.yaml
```

---

## Inserting layers between default layers — worked example

**Motivation:** A radial-glia variant needs to run `cell_rg_differentiation` between L3 (metabolism) and L7 (cell–cell interaction) so that updated `epithelialization_level` values are visible to the adhesion function within the same step.

Because FLAMEGPU2 layers are executed in registration order, the only way to insert between existing layers is to register the default layers yourself, with your new layers added at the right positions.  Copy the relevant portion of `_build_default_layers` and expand it:

```python
# variants/radial_glia/__init__.py

def configure_layers(model, g: dict) -> None:
    # Register variant env properties first.
    _env = g.get("env")
    if _env is not None:
        try:
            _env.newPropertyFloat("RG_COMMIT_THRESHOLD", g.get("RG_COMMIT_THRESHOLD", 0.5))
            # ... other RG properties ...
        except Exception:
            pass

    # --- Replicate _build_default_layers with RG layers inserted ---
    # L0 (VASC) — unchanged, call the default block up to here or copy it:
    # (copy the L0 block from _build_default_layers as-is)

    # L1: Agent Locations
    model.newLayer("L1_Agent_Locations").addAgentFunction("BCORNER", "bcorner_output_location_data")
    if g["INCLUDE_DIFFUSION"]:
        model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_grid_location_data")
    if g["INCLUDE_CELLS"]:
        model.Layer("L1_Agent_Locations").addAgentFunction("CELL", "cell_spatial_location_data")
    # ... (copy remaining L1 lines) ...

    # L2 — copy from _build_default_layers unchanged

    # L3: Metabolism & Cell Cycle — copy from _build_default_layers unchanged

    # *** INSERT: RG differentiation reads L1 spatial message, updates
    #             epithelialization_level before L7 cell–cell interaction. ***
    if g["INCLUDE_CELLS"]:
        model.newLayer("L3b_RG_Differentiation").addAgentFunction("CELL", "cell_rg_differentiation")

    # L4-L6: Diffusion — copy from _build_default_layers unchanged

    # *** INSERT: RG polarity update reads updated diffusion field. ***
    if g["INCLUDE_CELLS"]:
        model.newLayer("L6b_RG_Polarity_Update").addAgentFunction("CELL", "cell_rg_polarity_update")

    # L7-L8: Mechanics and Movement — copy from _build_default_layers unchanged
```

The complete `_build_default_layers` source is in `model.py` (search for `def _build_default_layers`).  Copy only the portions you need to replicate; omit subsystems your variant disables.

> **Maintenance note**: if `_build_default_layers` in `model.py` ever gains a new layer, variants that replicate it manually will not pick up the change automatically.  

---

## Adding variant-specific environment properties

Register them inside `configure_layers` using the live `env` object from globals:

```python
def configure_layers(model, g: dict) -> None:
    _env = g.get("env")
    if _env is not None:
        try:
            _env.newPropertyFloat("MY_PARAM", g.get("MY_PARAM", 1.0))
        except Exception:
            pass  # already registered (safe guard for re-runs)
    g['_build_default_layers']()
```

The `try/except` guard prevents errors if the model is restarted or called multiple times in the same process.

---

## Variant-gated agent variables

Some variants introduce per-cell biological state (e.g. RG differentiation progress, apical polarity vector) that is irrelevant — and wastes GPU register pressure, shared memory, and VTK file size — for every unrelated assay running on the same codebase.

The solution is to guard the new variables behind a boolean flag that is `True` only when the appropriate variant is loaded.

### Pattern

**Step 1 — model.py: define the gate flag**

Add one list of variant names and one derived boolean near the other `INCLUDE_*` flags:

```python
# model.py  (~line 235, near other INCLUDE_* flags)
VARIANTS_WITH_RG_VARIABLES = ["radial_glia"]        # extend if another variant reuses them
INCLUDE_RG_VARIABLES = (_VARIANT_NAME in VARIANTS_WITH_RG_VARIABLES)
```

`_VARIANT_NAME` is the string passed via `--variant` (empty string / `None` when no variant is active), so `INCLUDE_RG_VARIABLES` is `False` for all base runs and all non-RG variants at zero cost.

**Step 2 — model.py: gate CELL agent variable declarations**

Wrap the new `newVariable*` calls behind the flag.  FLAMEGPU2 allocates GPU memory per-variable per-agent at compile time, so any variable registered here occupies memory for every cell in every run:

```python
# model.py — in the CELL agent variable section
if INCLUDE_RG_VARIABLES:
    CELL_agent.newVariableFloat("rg_commit_level",         0.0)  # [-] logistic commit state
    CELL_agent.newVariableFloat("epithelialization_level", 0.0)  # [-] junction coverage
    CELL_agent.newVariableFloat("rosette_maturity",        0.0)  # [-] rosette formation index
    CELL_agent.newVariableFloat("apx", 0.0)  # apical vector x
    CELL_agent.newVariableFloat("apy", 0.0)  # apical vector y
    CELL_agent.newVariableFloat("apz", 0.0)  # apical vector z
    CELL_agent.newVariableFloat("rg_neighbour_density", 0.0)  # local RG count (normalised)
    CELL_agent.newVariableFloat("ecm_macro_sp2",        0.0)  # cached ECM morphogen sample
    CELL_agent.newVariableInt("rg_committed", 0)              # 0/1 irreversible commit flag
```

**Step 3 — model.py: gate spatial-message variable declarations**

Spatial messages are broadcast to every cell within the search radius on every step.  Each extra float adds 4 bytes to every message packet:

```python
# model.py — in the cell_spatial_location_message variable block
if INCLUDE_RG_VARIABLES:
    CELL_spatial_location_message.newVariableFloat("rg_commit_level")
    CELL_spatial_location_message.newVariableFloat("epithelialization_level")
    CELL_spatial_location_message.newVariableFloat("apx")
    CELL_spatial_location_message.newVariableFloat("apy")
    CELL_spatial_location_message.newVariableFloat("apz")
```

**Step 4 — model.py: gate host-init values**

Inside the CELL host-init loop, set the extra variables only when the flag is active:

```python
# model.py — inside the "for i in range(N_CELLS):" loop
if INCLUDE_RG_VARIABLES:
    _ap_angle = np.random.uniform(0.0, 2.0 * np.pi)  # random in-plane apical direction
    instance.setVariableFloat("apx", float(np.cos(_ap_angle)))
    instance.setVariableFloat("apy", float(np.sin(_ap_angle)))
    instance.setVariableFloat("apz", 0.0)
    instance.setVariableFloat("rg_commit_level",         0.0)
    instance.setVariableFloat("epithelialization_level", 0.0)
    instance.setVariableFloat("rosette_maturity",        0.0)
    instance.setVariableFloat("rg_neighbour_density",    0.0)
    instance.setVariableFloat("ecm_macro_sp2",           0.0)
    instance.setVariableInt("rg_committed", 0)
```

**Step 5 — model.py: build the extra-VTK lists**

Define two lists in `model.py` (where `SaveDataToFile.run` can pick them up via globals) that describe which extra per-cell scalars and vectors to write:

```python
# model.py — near the SaveDataToFile class or just before the simulation run
CELL_VTK_EXTRA_SCALARS = []   # list of (vtk_name, agent_variable_name, dtype_str)
CELL_VTK_EXTRA_VECTORS = []   # list of (vtk_name, vx_var, vy_var, vz_var)
if INCLUDE_RG_VARIABLES:
    CELL_VTK_EXTRA_SCALARS = [
        ("rg_commit_level",         "rg_commit_level",         "float"),
        ("epithelialization_level", "epithelialization_level", "float"),
        ("rosette_maturity",        "rosette_maturity",        "float"),
        ("rg_neighbour_density",    "rg_neighbour_density",    "float"),
        ("ecm_macro_sp2",           "ecm_macro_sp2",           "float"),
        ("rg_committed",            "rg_committed",            "int"),
    ]
    CELL_VTK_EXTRA_VECTORS = [
        ("apical_vector", "apx", "apy", "apz"),
    ]
```

Pass them through the config dict in `SaveDataToFile.run()`:
```python
config={
    # ... existing keys ...
    "CELL_VTK_EXTRA_SCALARS": CELL_VTK_EXTRA_SCALARS,
    "CELL_VTK_EXTRA_VECTORS": CELL_VTK_EXTRA_VECTORS,
}
```

**Step 6 — helper_module.py: write the extra fields**

In `save_data_to_file_step`, after the fixed orientation VECTORS block and still inside the `with open(...)` block, add:

```python
# Extra per-cell scalars injected by variants
for vtk_name, var_name, dtype in config.get("CELL_VTK_EXTRA_SCALARS", []):
    fmt = "{:.4f} \n" if dtype == "float" else "{} \n"
    file.write(f"SCALARS {vtk_name} {dtype} 1\n")
    file.write("LOOKUP_TABLE default\n")
    extra_data = []
    for ai in av:
        val = ai.getVariableFloat(var_name) if dtype == "float" else ai.getVariableInt(var_name)
        extra_data.append(val)
    for val in extra_data:
        file.write(fmt.format(val))
    for i in range(num_cells):
        for _ in range(num_anchor_points):
            file.write(fmt.format(extra_data[i]))

# Extra per-cell vectors injected by variants
for vtk_name, vx_var, vy_var, vz_var in config.get("CELL_VTK_EXTRA_VECTORS", []):
    file.write(f"VECTORS {vtk_name} float\n")
    for ai in av:
        vx = ai.getVariableFloat(vx_var)
        vy = ai.getVariableFloat(vy_var)
        vz = ai.getVariableFloat(vz_var)
        file.write(f"{vx:.4f} {vy:.4f} {vz:.4f} \n")
    for _ in range(num_total_anchor_points):
        file.write("0.0 0.0 0.0 \n")
```

### Why this pattern works

| Scenario | `INCLUDE_RG_VARIABLES` | GPU memory | VTK output |
|---|---|---|---|
| Base run / any non-RG variant | `False` | no extra | no extra |
| `--variant radial_glia` | `True` | +9 floats + 1 int per cell | RG fields appended |

The lists `CELL_VTK_EXTRA_SCALARS` / `CELL_VTK_EXTRA_VECTORS` are the only coupling point between `model.py` and `helper_module.py`; no other changes to `helper_module.py` are needed when new variables are added.

### Generalising to other variants

Follow the same naming convention for any variant group that adds biologically-specific per-agent state:

```python
VARIANTS_WITH_INVASION_VARIABLES = ["tumour_invasion", "wound_healing"]
INCLUDE_INVASION_VARIABLES = (_VARIANT_NAME in VARIANTS_WITH_INVASION_VARIABLES)
```

---

## Creating a new variant — step-by-step checklist

1. **Create `variants/<name>/`** with an `__init__.py` containing `PARAMS`, `FILES`, and (optionally) `configure_globals` / `configure_layers`.
2. **Add only the `.cpp` files that differ** from the base model to the same folder.  Copy the relevant base `.cpp`, apply your changes, and document the diff clearly.
3. **Test the direct run**:
   ```bash
   python model.py --variant <name>
   ```
4. If optimizing, **create `optimizer/optuna_config_<name>_variant.yaml`** with `model.variant: <name>` and the search space.
5. **Do not edit `model.py`** unless you need a new feature flag pre-defined with a sensible default (add it near the other `INCLUDE_*` flags and override it via `PARAMS`).  For new agent variables or message variables there is no alternative — add them to `model.py` behind an `INCLUDE_*` guard.

---

## What variants cannot do (and what to do instead)

| Need | Solution |
|---|---|
| Add a brand-new agent type with its own message list | Add it to `model.py` behind an `INCLUDE_*` flag; enable it via `PARAMS` |
| Add new agent variables or message variables | Add them to `model.py`; they default to zero and are harmless in unrelated assays |
| Override a parameter that doesn't yet exist in `model.py` | Add it to `model.py` with a sensible default, then override in `PARAMS` |

The principle is: **variants encode biological deltas; structural additions go into the main model**.  This keeps `model.py` as the single source of truth for what the simulator is capable of.

---

## Reference: execution flow with a variant loaded

```
python model.py --variant organoid --overrides configs/extra.json

  1. model.py default parameters defined
  2. --variant organoid parsed from _ORIGINAL_ARGV
  3. variants/organoid/__init__.py loaded via importlib
  4. variant.PARAMS applied  →  organoid baseline set
  5. --overrides extra.json parsed and applied  →  JSON wins over PARAMS
  6. _file variables assigned (e.g. cell_cycle_file = "cell_cycle.cpp")
  7. variant.FILES applied   →  cell_cycle_file = "variants/organoid/cell_cycle.cpp"
  8. variant.configure_globals(globals()) called  →  new flags injected
  9. FLAMEGPU2 ModelDescription built (agents, messages, env properties)
 10. variant.configure_layers(model, globals()) called
       → variant calls g['_build_default_layers']() for the standard L0-L8 stack
       → variant registers extra env properties
       (if configure_layers is absent, model.py calls _build_default_layers() directly)
 11. Logging, simulation run, output saved
```

---

## Directory layout

```
cellfoundry/
├── model.py                          ← base model (never edited per-variant)
├── cell_cycle.cpp                    ← base agent functions
├── cell_move.cpp
│   ...
├── variants/
│   ├── organoid.py                   ← organoid variant module
│   └── organoid/
│       └── cell_cycle.cpp            ← organoid-specific override
└── optimizer/
    ├── optuna_config_organoid_variant.yaml
    └── ...
```

Each variant is a **single Python file** in `variants/`.  The subfolder `variants/<name>/` holds only the `.cpp` files that differ from the base model.

---

## Anatomy of a variant module

A variant module exports up to four objects.  All are optional.

```python
# variants/my_variant.py

PARAMS: dict        # parameter overrides  (applied before JSON --overrides)
FILES:  dict        # *_file variable redirections (applied after all parameters)

def configure_globals(g: dict) -> None: ...   # inject new global flags
def configure_layers(model, g: dict) -> None: # inject / reorder layers
```

### `PARAMS`

A plain dict mapping parameter names to values.  Any key that exists as a global variable in `model.py` can be overridden.  Scalars are broadcast to lists automatically (same behaviour as `--overrides` JSON).

```python
PARAMS = {
    "ORGANOID_ASSAY": True,
    "N_CELLS": 13,
    "CELL_RADIUS": [20.0, 20.0, 20.0],     # explicit list
    "CELL_SPEED_REF": 0.006,                # scalar → broadcast to all types
    "CYCLE_PHASE_G1_DURATION": [12000.0, 24000.0, 36000.0],
}
```

**Priority**: `--overrides` JSON always wins over `PARAMS`.  This means the optimizer can tune any variant parameter without editing the variant file.

### `FILES`

A dict mapping the `*_file` variable names in `model.py` to variant-specific `.cpp` paths (relative to the project root, i.e. the directory containing `model.py`).

```python
FILES = {
    "cell_cycle_file": "variants/organoid/cell_cycle.cpp",
    "cell_move_file":  "variants/organoid/cell_move.cpp",
}
```

Only list files that actually differ from the base model.  Unmentioned functions use the base `.cpp` as usual.

### `configure_globals(g)`

Called after `PARAMS` and `FILES` are applied but **before** `model.py` builds the FLAMEGPU2 `ModelDescription`.  Use it to inject global flags that don't exist in the base model and therefore cannot go into `PARAMS`.

```python
def configure_globals(g: dict) -> None:
    # Register a new feature flag that the custom cell_cycle.cpp relies on.
    g["MY_CUSTOM_FLAG"] = True
    g["CONTACT_INHIBIT_SIGMA"] = 1.5   # [kPa]
```

The layer section in `model.py` can then check `globals().get("MY_CUSTOM_FLAG", False)` safely even when no variant is loaded.

### `configure_layers(model, g)`

Called after **all default layers** have been added.  Use it to:

1. **Append entirely new layers** (the most common use case).
2. **Register variant-specific environment properties** on `g["env"]`.

Layer reordering (moving an existing layer to a different position) is the one case that requires a small surgical change to `model.py` — see the section below.

```python
def configure_layers(model, g: dict) -> None:
    # Append a new statistics layer (requires a matching RTC function
    # "cell_organoid_stats" registered on the CELL agent in model.py).
    if g.get("INCLUDE_ORGANOID_STATS", False):
        model.newLayer("L9_Organoid_Stats").addAgentFunction("CELL", "cell_organoid_stats")
```

---

## Running a variant

```bash
# Basic run
python model.py --variant organoid

# With additional parameter overrides (JSON wins over variant PARAMS)
python model.py --variant organoid --overrides configs/organoid_paper.json

# Specify result directory
python model.py --variant organoid --result-dir result_files/organoid_run_01
```

If the variant name is not found in `variants/`, the model exits with a clear error listing available variants.

---

## Using variants with the optimizer

Add `model.variant: <name>` to any Optuna YAML config.  The optimizer forwards `--variant <name>` to every trial subprocess automatically.

```yaml
# optimizer/optuna_config_organoid_variant.yaml

model:
  variant: organoid          # ← loads variants/organoid.py for every trial
  extra_overrides:
    DEBUG_PRINTING: false
  timeout: 0
  cleanup_trials: false

parameters:
  CELL_SPEED_REF:
    type: float
    low: 0.0001
    high: 0.02
    log: true
  # ...
```

The `parameters:` block is the search space; it overrides only the specific values being tuned while the variant provides all other calibrated defaults.

```bash
python -m optimizer.optimize --config optimizer/optuna_config_organoid_variant.yaml
```

---

## Layer reordering — complete worked example

This is the **only case where you need to touch `model.py`**.  Layer reordering cannot be done purely from a variant module because the default layers are registered with sequential imperative calls; the only way to skip one is to add a guard at that specific call-site.

The change is small and backward-compatible: non-variant runs are unaffected because the guard condition will always be False when no variant is loaded.

**Motivation:** In a dense organoid, the cell-cycle block runs at **L3** (before movement).  A newborn daughter is placed inside its parent and sits there for one full step before repulsion can push it away.  Moving the block to **L9** (after movement) means daughters are placed at the very end of the step; repulsion resolves on the next full step with no stiff-overlap transient.

### Step 1 — Add a guard in model.py

Add a new boolean flag to your variant's `PARAMS` (any name you choose):

```python
# variants/organoid.py
PARAMS = {
    # ... other params ...
    "ORGANOID_LATE_CYCLE": True,   # move cell-cycle to after movement
}
```

This flag must first be pre-defined in `model.py` so that `apply_param_overrides` can find it:

```python
# model.py — add once, near the other feature flags (e.g. line ~450)
ORGANOID_LATE_CYCLE: bool = False
```

Then wrap the L3 cell-cycle block with the guard:

```python
# model.py — layer section (excerpt)
if INCLUDE_CELLS and INCLUDE_CELL_CYCLE and not ORGANOID_LATE_CYCLE:
    model.newLayer("L3_Cell_MaxID_Update").addAgentFunction("CELL", "cell_MaxID_update")
    model.newLayer("L3_Cell_Cycle").addAgentFunction("CELL", "cell_cycle")
    if INCLUDE_FOCAL_ADHESIONS:
        model.newLayer("L3_Cell_Bucket_PostCycle").addAgentFunction("CELL", "cell_bucket_location_data")
        model.newLayer("L3_FOCAD_PostCycle_Update").addAgentFunction("FOCAD", "focad_post_cycle_update")
```

### Step 2 — Re-add the block at the new position via configure_layers

```python
# variants/organoid.py
def configure_layers(model, g: dict) -> None:
    if (g.get("ORGANOID_LATE_CYCLE", False)
            and g.get("INCLUDE_CELLS", False)
            and g.get("INCLUDE_CELL_CYCLE", False)):
        # Appended after L8 — daughters are placed at end of each step.
        model.newLayer("L9_ORGANOID_Cell_MaxID_Update").addAgentFunction("CELL", "cell_MaxID_update")
        model.newLayer("L9_ORGANOID_Cell_Cycle").addAgentFunction("CELL", "cell_cycle")
        if g.get("INCLUDE_FOCAL_ADHESIONS", False):
            model.newLayer("L9_ORGANOID_Cell_Bucket_PostCycle").addAgentFunction("CELL", "cell_bucket_location_data")
            model.newLayer("L9_ORGANOID_FOCAD_PostCycle_Update").addAgentFunction("FOCAD", "focad_post_cycle_update")
```

### Step 3 — Resulting execution order

| Layer | Function | Notes |
|---|---|---|
| L1_Agent_Locations | BCORNER, ECM, CELL | spatial broadcast |
| L7_CELL_Cell_Interaction | CELL | repulsion / adhesion |
| **L8_CELL_Movement** | CELL | cells move |
| *(L3_Cell_Cycle skipped)* | — | guarded by `ORGANOID_LATE_CYCLE` |
| **L9_ORGANOID_Cell_Cycle** | CELL | divide / die *after* movement |

### Design note

This is the deliberate trade-off of the variant system: `model.py` remains the single source of truth for **what the simulator is capable of**, while variant modules encode only the **biological delta**.  A layer reorder is an architectural capability decision — it belongs in `model.py`, documented with a clear flag name and default value.  A reader of `model.py` who has never heard of variants will still understand the guard: `if INCLUDE_CELL_CYCLE and not ORGANOID_LATE_CYCLE:`.

---

## Adding variant-specific environment properties

If your custom `.cpp` reads environment properties that don't exist in the base model, register them in `configure_layers` using the live `env` object from globals:

```python
def configure_layers(model, g: dict) -> None:
    _env = g.get("env")
    if _env is not None:
        try:
            _env.newPropertyFloat("CONTACT_INHIBIT_SIGMA", 1.5)   # [kPa]
            _env.newPropertyFloat("CONTACT_INHIBIT_FACTOR", 3.0)
        except Exception:
            pass  # already registered (safe guard for re-runs)
```

The `try/except` guard prevents errors if the model is restarted or called multiple times in the same process.

> **Important**: The environment must be fully configured *before* `model.py` registers the agent populations and runs the simulation.  Since `configure_layers` is called right before the logging / simulation setup section, this timing is correct.

---

## Creating a new variant — step-by-step checklist

1. **Create `variants/<name>.py`** with `PARAMS`, `FILES`, and (optionally) `configure_globals` / `configure_layers`.
2. **Create `variants/<name>/`** with only the `.cpp` files that differ.  Copy the relevant base `.cpp` and apply your changes; document the diff clearly.
3. **Test the direct run**:
   ```bash
   python model.py --variant <name>
   ```
4. If optimizing, **create `optimizer/optuna_config_<name>_variant.yaml`** with `model.variant: <name>` and the search space.
5. **Do not edit `model.py`** unless you need a layer reorder (add a boolean flag + guard, see section above) or a new feature flag pre-defined with a sensible default.  Both are small, backward-compatible changes.

---

## What variants cannot do (and what to do instead)

| Need | Solution |
|---|---|
| Add a brand-new agent type with its own message list | Add it to the base `model.py` behind an `INCLUDE_*` flag; enable it via `PARAMS` |
| Change agent variable definitions | Add the variable to the base `model.py` agent definition (no variant needed); initialize it conditionally |
| Structural reorder of more than 2–3 layers | Consider a feature-branch if the delta is large enough to justify it |
| Override a parameter that doesn't yet exist in `model.py` | Add it to `model.py` with a sensible default, then override in `PARAMS` |

The principle is: **variants encode biological deltas; structural additions go into the main model**.  This keeps `model.py` as the single source of truth for what the simulator is capable of.

---

## Reference: execution flow with a variant loaded

```
python model.py --variant organoid --overrides configs/extra.json

  1. model.py default parameters defined (lines ~35–446)
  2. --variant organoid parsed from _ORIGINAL_ARGV
  4. variants/organoid.py loaded via importlib
  3. variants/organoid.py loaded via importlib
  4. variant.PARAMS applied  →  organoid baseline set
  5. --overrides extra.json parsed and applied  →  JSON wins over PARAMS
  6. _file variables assigned (e.g. cell_cycle_file = "cell_cycle.cpp")
  7. variant.FILES applied   →  cell_cycle_file = "variants/organoid/cell_cycle.cpp"
  8. variant.configure_globals(globals()) called  →  new flags injected
  9. FLAMEGPU2 ModelDescription built (agents, messages, env properties)
 10. Default layers added (layer-reorder guards in model.py honoured if present)
 11. variant.configure_layers(model, globals()) called  →  new layers appended
 12. Logging, simulation run, output saved
```
