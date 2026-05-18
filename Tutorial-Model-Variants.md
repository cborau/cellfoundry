# Tutorial: Model Variants

This tutorial explains how to use the **variant system** to simulate different biological problems within the same CellFoundry codebase, without branching the repository or duplicating files.

---

## Motivation

CellFoundry is a general-purpose agent-based simulation platform.  Different biological problems (organoid growth, tumour invasion, fibre remodelling, wound healing, …) share most of the simulation infrastructure but differ in:

- Parameter values (cell speed, adhesion strength, domain size, etc.)
- Agent function logic (e.g. a custom `cell_cycle.cpp` with asymmetric division)
- Layer execution order (e.g. run cell cycle *after* movement for fast-proliferating organoids)
- Variant-specific environment properties registered on the FLAMEGPU2 model

The variant system lets you encode all of these differences in a single Python file (`variants/<name>.py`) plus any modified `.cpp` files (`variants/<name>/`), leaving `model.py` untouched.

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
