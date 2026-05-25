# Tutorial: Model Variants

This tutorial explains the variant system using one concrete example: `radial_glia`.

The goal is to understand what a variant can change and how those changes are loaded.

## What a Variant Is

A variant is a folder under `variants/<name>/` with an `__init__.py` file and optional overridden `.cpp` files.

Example:

```text
variants/
  radial_glia/
    __init__.py
    cell_move.cpp
    cell_cell_interaction.cpp
    cell_spatial_location_data.cpp
    cell_rg_differentiation.cpp
    cell_rg_polarity_update.cpp
```

## What `__init__.py` Can Export

The variant module may define:

- `PARAMS`: parameter overrides applied to globals
- `FILES`: remap of `*_file` variables to variant-specific `.cpp`
- `configure_globals(g)`: add or adjust Python globals before model construction
- `configure_layers(model, g)`: control the full layer sequence

All are optional.

## Load and Precedence Order

When running with `--variant radial_glia`:

1. `model.py` defaults are loaded
2. `radial_glia.PARAMS` is applied
3. JSON from `--overrides` is applied (if provided)

So JSON overrides always win.

For a complete CLI-overrides walkthrough, see [Tutorial: Parameter Overriding](Tutorial-Parameter-Overriding).

## Running the Variant

```bash
python model.py --variant radial_glia
```

With extra overrides:

```bash
python model.py --variant radial_glia --overrides configs/organoid_test.json
```

## Why `configure_layers(model, g)` Exists

In `model.py`, the default layers are built by `_build_default_layers()`.

If a variant defines `configure_layers`, `model.py` delegates full control of layer creation to that function. This is necessary for `radial_glia` because it inserts custom layers between default ones:

- `L3b_RG_Differentiation`
- `L6b_RG_Polarity_Update`

Those insertions cannot be achieved by only appending layers after calling default layers.

## Why `g.get("env")` Is Needed in `configure_layers`

`configure_globals(g)` writes plain Python globals into `g`, but that does not automatically register FLAMEGPU environment properties.

Agent RTC/C++ code reads model environment properties, so they must exist on the FLAMEGPU environment object:

```python
_env = g.get("env")
if _env is not None:
    _register_rg_env_properties(_env, g)
```

This is why `radial_glia` calls `_register_rg_env_properties` inside `configure_layers`:

- `configure_globals` sets values in Python space
- `_register_rg_env_properties` publishes them into FLAMEGPU environment space

Both steps are needed when variant-specific RTC/C++ functions access these properties.

In other words, `configure_globals(g)` is useful for injecting values like `RG_COMMIT_RATE`, `RG_ADHESION_MATRIX`, etc., but it does not call `env.newProperty*`.

Without `env.newProperty*`, those keys do not become FLAMEGPU environment properties and C++/RTC agent functions cannot safely read them.

So in this variant:

- `configure_globals(g)` defines the values
- `_register_rg_env_properties(env, g)` registers the values for simulation runtime

## Minimal Anatomy (Radial-Glia)

Conceptually, `radial_glia/__init__.py` does three main things:

1. Sets baseline parameters with `PARAMS`
2. Redirects selected C++ files with `FILES`
3. Builds a custom layer order in `configure_layers`, including:
   - registration of variant env properties
   - registration of two new RTC functions
   - insertion of new layers in the L3/L6 regions

## Variants and Optimization

Variants can be used directly in Optuna configs through:

```yaml
model:
  variant: radial_glia
```

Then Optuna trial parameters are still passed through overrides, so study parameters can tune on top of the variant defaults.

See [Tutorial: Parameter Optimization](Tutorial-Param-Optimization).

## Practical Workflow

1. Put stable assay defaults in variant `PARAMS`.
2. Keep run-specific tweaks in JSON `--overrides` files.
3. Use `configure_layers` only when the variant needs custom ordering/insertion.
4. Register variant env properties explicitly when C++/RTC functions depend on them.

## Related Tutorials

- [Tutorial: Parameter Overriding](Tutorial-Parameter-Overriding)
- [Tutorial: Parameter Optimization](Tutorial-Param-Optimization)
- [Tutorial: First Steps](Tutorial-First-Steps)
