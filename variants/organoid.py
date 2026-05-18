"""
Variant: organoid
=================
Configures CellFoundry to reproduce the organoid growth assay described in
organoid_paper.json.  Cells start as a compact cluster and expand radially
while undergoing cell-cycle driven proliferation.

Usage
-----
    # Direct run:
    python model.py --variant organoid

    # With additional JSON overrides (JSON wins over variant PARAMS):
    python model.py --variant organoid --overrides configs/my_overrides.json

    # Optimizer (via YAML model.variant key — see Tutorial-Model-Variants.md):
    python -m optimizer.optimize --config optimizer/optuna_config_organoid_variant.yaml

Structure of this file
-----------------------
    PARAMS  — scalar / list parameter overrides applied to model.py globals.
              Uses the same keys as the JSON override system.  Scalars are
              broadcast to all cell types automatically.
    FILES   — maps *_file variable names to variant-specific .cpp paths.
              Paths are relative to the project root (CURR_PATH).
    configure_layers(model, g)
            — optional callback injected into the layer-building section of
              model.py.  Receives the pyflamegpu.ModelDescription object and
              the current module globals dict.  Use it to:
                (a) insert new layers at a named injection point, or
                (b) re-add a suppressed default layer at a different position
                    (suppress it via a boolean in PARAMS, see tutorial).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# PARAMS — parameter overrides
# ---------------------------------------------------------------------------
# All keys must exist as globals in model.py.  Scalars are broadcast to lists
# of the correct length by apply_param_overrides().  Explicitly override here
# only the values that differ from the base model defaults.
PARAMS: dict = {
    # --- Simulation control -------------------------------------------------
    "STEPS": 2400,
    "TIME_STEP": 180,           # [s] 3-minute steps — 2400 × 180 s = 120 h
    "SAVE_EVERY_N_STEPS": 12,

    # --- Feature flags ------------------------------------------------------
    "INCLUDE_CELLS": True,
    "INCLUDE_CELL_CELL_INTERACTION": True,
    "INCLUDE_CELL_CYCLE": True,
    "INCLUDE_FOCAL_ADHESIONS": False,
    "DEAD_CELLS_DISAPPEAR": False,
    "PERIODIC_BOUNDARIES_FOR_CELLS": False,
    "INCLUDE_CELL_FNODE_REPULSION": False,
    "INCLUDE_FIBRE_NETWORK": False,
    "INCLUDE_NETWORK_REMODELING": False,
    "INCLUDE_DIFFUSION": False,
    "INCLUDE_CHEMOTAXIS": False,
    "INCLUDE_CHEMOKINESIS": False,
    "INCLUDE_LUMEN": False,
    "INCLUDE_VASCULARIZATION": False,

    # --- Organoid initialisation --------------------------------------------
    "ORGANOID_ASSAY": True,
    "ORGANOID_INIT_RADIUS": 20.0,           # [um] tight initial cluster
    "ORGANOID_ORIENTATION_NOISE": 0.3,      # [rad] mild radial jitter

    # --- Output & visualisation ---------------------------------------------
    "VISUALISATION": False,
    "SHOW_PLOTS": False,
    "SAVE_DATA_TO_FILE": True,
    "SAVE_PICKLE": True,

    # --- Cell population ----------------------------------------------------
    "N_CELLS": 13,
    "CELL_RADIUS": [20.0, 20.0, 20.0],     # [um] large cells for organoid

    # --- Cell migration (calibrated from organoid_paper) --------------------
    # Scalars are broadcast to all N_CELL_TYPES automatically.
    "CELL_SPEED_REF": 0.006197015748809144,          # [um/s]
    "ROTATIONAL_DIFFUSION_RATE": 0.0004325207525386532,  # [rad^2/s]

    # --- Cell–cell mechanics (calibrated) -----------------------------------
    "CELL_CELL_DV_MAX": 0.000285673742984719,        # [um/s] — scalar broadcast
    "CELL_CELL_ADHESION_K": 9.857444189237748,       # [nN/um]
    "CELL_CELL_REPULSION_K": 60.793730704953695,     # [nN/um]

    # --- Cell cycle timing --------------------------------------------------
    # Non-uniform G1 durations give three distinct proliferation rates.
    "DIVISION_RATE_MULTIPLIER": [1.0, 1.0, 1.0],
    "CYCLE_PHASE_G1_DURATION": [12000.0, 24000.0, 36000.0],  # [s]

    # --- Damage / death (disabled — no diffusion in this assay) ------------
    "CELL_HYPOXIA_DAMAGE_RATE": [0.0, 0.0, 0.0],
    "CELL_NUTRIENT_DAMAGE_RATE": [0.0, 0.0, 0.0],
    "CELL_STRESS_DAMAGE_RATE": [0.0, 0.0, 0.0],
    "CELL_BASAL_DAMAGE_REPAIR_RATE": [0.0, 0.0, 0.0],
}


# ---------------------------------------------------------------------------
# FILES — agent function file overrides
# ---------------------------------------------------------------------------
# Keys are the *_file variable names set in model.py (e.g. cell_cycle_file).
# Values are paths relative to the project root (CURR_PATH in model.py).
# Only include files that actually differ from the base model.
FILES: dict = {
    # Custom cell-cycle logic: apical cells (type 0) undergo asymmetric
    # division; luminal cells (type 1) exit cycle at high density.
    "cell_cycle_file": "variants/organoid/cell_cycle.cpp",
}


# ---------------------------------------------------------------------------
# configure_globals — inject new global flags before the model is built
# ---------------------------------------------------------------------------
def configure_globals(g: dict) -> None:
    """Set arbitrary global variables that have no counterpart in the base model.

    Called after PARAMS and FILES are applied, but before model.py builds the
    FLAMEGPU2 ModelDescription.  ``g`` is ``globals()`` from model.py.

    Use this for feature flags that are brand-new to this variant and therefore
    cannot be pre-listed in PARAMS (which only overrides *existing* globals).

    For the organoid variant, no new flags are required by default.  The
    contact-inhibition environment properties needed by the custom
    cell_cycle.cpp are registered in configure_layers() below.
    """
    # Example (not needed unless VARIANT_SUPPRESS_LAYERS is active):
    g["ORGANOID_CONTACT_INHIBIT_SIGMA"] = 1.5  # [kPa]
    g["ORGANOID_CONTACT_INHIBIT_FACTOR"] = 3.0
    pass


# ---------------------------------------------------------------------------
# configure_layers — layer injection callback
# ---------------------------------------------------------------------------
def configure_layers(model, g: dict) -> None:
    """Inject variant-specific simulation layers and environment properties.

    Called from model.py immediately after all default layers have been added.
    ``model`` is the live ``pyflamegpu.ModelDescription`` instance.
    ``g`` is ``globals()`` from model.py at the time of the call.

    New layers added here appear AFTER all default layers.  Layer reordering
    (moving an existing layer to a different position) is not supported via
    this hook — it requires a surgical guard in model.py for that specific
    layer.  See Tutorial-Model-Variants.md for details.

    New environment properties
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Variant-specific environment properties read by custom .cpp files must be
    registered here on ``g["env"]`` (the ModelDescription's environment
    object, already stored in model.py globals).
    """
    # Register contact-inhibition parameters read by the custom cell_cycle.cpp.
    # Guard with try/except so re-running in the same Python process is safe.
    _env = g.get("env")
    if _env is not None:
        try:
            _env.newPropertyFloat("ORGANOID_CONTACT_INHIBIT_SIGMA", 1.5)
            _env.newPropertyFloat("ORGANOID_CONTACT_INHIBIT_FACTOR", 3.0)
        except Exception:
            pass  # property already registered (safe to ignore)

    # Example: add a custom stats layer (requires a matching RTC function).
    # if g.get("INCLUDE_ORGANOID_STATS", False):
    #     model.newLayer("L9_Organoid_Stats").addAgentFunction("CELL", "cell_organoid_stats")

