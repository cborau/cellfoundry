"""
variants/radial_glia/__init__.py

Radial-glia (RG) differentiation variant.

Starting from a monolayer of iPSC (cell_type=0), cells progressively
differentiate to NPC (cell_type=1) and then RG (cell_type=2) in response
to a locally secreted morphogen (ECM species 2).  Differentiated cells
develop apical-basal polarity and form rosette-like structures.

Key additions versus the base model:
  - cell_rg_differentiation.cpp  — commitment + cell-type switching ODE
  - cell_rg_polarity_update.cpp  — apical-vector relaxation toward sp-2 gradient
  - cell_spatial_location_data.cpp override — broadcasts RG variables in messages
  - cell_cell_interaction.cpp override     — type-pair adhesion matrix + epi boost
  - cell_move.cpp override                 — substrate spring + apical bias

Layer sequence (full, with two inserted layers):
  L1   cell_spatial_location_data   ← OVERRIDE
  L2   ECM boundary conditions
  L3   cell_ecm_interaction_metabolism
  L3b  cell_rg_differentiation      ← NEW
  L4   ECM Csp update
  L5   ECM-ECM diffusion
  L6   ECM boundary (second call)
  L6b  cell_rg_polarity_update      ← NEW
  L7   cell_cell_interaction        ← OVERRIDE
       cell_stress_state_update
  L8   cell_move                    ← OVERRIDE
"""

from pathlib import Path

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# PARAMS — override / extend default model.py globals
# ---------------------------------------------------------------------------
PARAMS = {
    # --- Assay type ---
    "MONOLAYER_ASSAY":            True,
    "ORGANOID_ASSAY":             False,
    "MONOLAYER_CELL_TYPE_RATIOS": [1, 0, 0],   # all iPSC at start
    # Seed cells in a ~60 µm disc at the centre of the ECM domain.
    # Interaction radius = 3 × cell_radius = 15 µm; for N=60 cells packed into a
    # 60 µm disc the mean nearest-neighbour distance is ~12 µm, so cells are
    # just within interaction range from t=0 while the full ±500 µm ECM domain
    # is preserved for gradient development.
    "MONOLAYER_CLUSTER_RADIUS":   60.0,         # [µm]

    # --- Agents ---
    "INCLUDE_CELLS":              True,
    "INCLUDE_CELL_CELL_INTERACTION": True,
    "INCLUDE_CELL_CYCLE":         False,
    "INCLUDE_DIFFUSION":          True,
    "INCLUDE_VASCULARIZATION":    False,
    "INCLUDE_FIBRE_NETWORK":      False,
    "INCLUDE_FOCAL_ADHESIONS":    False,
    "INCLUDE_LUMEN":              False,

    # --- Cell geometry ---
    "N_CELLS":                    60,
    "CELL_RADIUS":                [5.0, 5.0, 5.0],          # [µm]

    # --- Cell motility ---
    "CELL_SPEED_REF":             [5e-4, 3e-4, 1e-4],       # [µm/s]
    "ROTATIONAL_DIFFUSION_RATE":  [2e-3, 1e-3, 2e-4],       # [rad²/s]

    # --- Cell-cell mechanics — scalars disabled; matrices override in configure_layers ---
    "CELL_CELL_ADHESION_K":       [0.0, 0.0, 0.0],          # [nN/µm] (replaced by matrix)
    "CELL_CELL_REPULSION_K":      [4.0, 4.0, 4.0],          # [nN/µm] (fallback; matrix used)

    # --- Diffusion ---
    "DIFFUSION_COEFF_MULTI":      [5.0, 5.0, 2.0],          # [µm²/s]
    "BOUNDARY_CONC_INIT_MULTI":   [[2.5]*6, [2.5]*6, [0.0]*6],
    "BOUNDARY_CONC_FIXED_MULTI":  [[2.5]*6, [2.5]*6, [0.0]*6],
    "INIT_CELL_CONCENTRATION_VALS": [2.5, 2.5, 0.0],        # species 2 starts at 0

    # --- Metabolism / secretion ---
    # sp0 and sp1 consumed by all types; sp2 secreted only by committed cells
    "INIT_CELL_PRODUCTION_RATES": [0.0, 0.0, 5e-4],         # [1/s] base for fully committed RG
    "CELL_PRODUCTION_MULTIPLIER": [0.0, 0.5, 1.0],          # type 0: silent, 1: half, 2: full

    # --- Simulation timing ---
    # TIME_STEP=60 s keeps explicit-diffusion Von Neumann number at ~0.72 (stable).
    # 7 days (168h) covers iPSC reseeding (day 3) through established rosettes (day 7+).
    "TIME_STEP":                  60.0,                      # [s]  ← was 10 s
    "STEPS":                      10080,                     # 7 days: 168 h × 3600 s/h ÷ 60 s/step
    "SAVE_EVERY_N_STEPS":         240,                       # save every 4h 
}

# ---------------------------------------------------------------------------
# FILES — redirect *_file global variables to variant-specific .cpp paths
# ---------------------------------------------------------------------------
FILES = {
    "cell_spatial_location_data_file": str(_HERE / "cell_spatial_location_data.cpp"),
    "cell_cell_interaction_file":      str(_HERE / "cell_cell_interaction.cpp"),
    "cell_move_file":                  str(_HERE / "cell_move.cpp"),
    # cell_rg_differentiation and cell_rg_polarity_update are new functions
    # registered programmatically in configure_layers below.
}


# ---------------------------------------------------------------------------
# configure_globals — inject global scalars needed before env build
# ---------------------------------------------------------------------------
def configure_globals(g: dict) -> None:
    """Inject RG-specific scalars into model.py globals before the model build.

    These values are read back in configure_layers to register env properties.
    They are set as globals so downstream code (e.g. logging) can reference them.
    """
    g["RG_COMMIT_RATE"]                = 5e-6    # [1/s]  basal commit rate
                                                          #        τ = 1/k ≈ 55.6 h; first NPC cells
                                                          #        appear ~24 h after reseeding
                                                          #        (1−exp(−5e−6×86400) ≈ 0.35 = NPC threshold)
    g["RG_COMMIT_AUTOCRINE_RATE"]      = 5e-4    # [1/(s·a.u.)] morphogen-driven amplification;
                                                          #        kicks in once the first NPC cells secrete
    g["RG_COMMIT_PARACRINE_RATE"]      = 1e-4    # [1/s]  RG neighbour-driven rate
    g["RG_COMMIT_THRESHOLD_NPC"]       = 0.35    # [-]
    g["RG_COMMIT_THRESHOLD_RG"]        = 0.70    # [-]
    g["RG_EPITHELIAL_RATE"]            = 1e-5    # [1/s]  was 1e-4; τ ≈ 28 h → polarity
                                                          #        develops gradually over days
    g["RG_POLARITY_TAU"]               = 3600.0  # [s]    apical-vector relaxation time (1 h)
                                                          #        was 600 s (10 min); slower makes
                                                          #        alignment accumulate over hours
    g["RG_POLARITY_GRADIENT_THRESHOLD"]= 1e-6    # [a.u./µm]
    g["RG_SUBSTRATE_K"]                = 0.5     # [nN/µm]
    g["RG_SUBSTRATE_Z0"]               = 0.0     # [µm above COORD_BOUNDARY_Z_NEG]
    g["RG_APICAL_BIAS_RG"]             = 2e-3    # [µm/s] for RG type
    g["RG_APICAL_BIAS_NPC"]            = 5e-4    # [µm/s] for NPC type
    # Adhesion matrix: 3×3 flattened row-major, indexed as [self_type * 3 + nb_type]  [nN/µm]
    # Rows = self cell type; columns = neighbour cell type.
    # Biological rationale:
    #   iPSC (0) — few adhesion molecules; weak attraction to all types
    #   NPC (1)  — N-cadherin begins to be expressed; moderate homotypic adhesion
    #   RG  (2)  — N-cadherin / NCAM rich junctions; strong homotypic adhesion drives
    #              rosette assembly; further boosted by epithelialization_level (×RG_EPITHELIAL_ADHESION_BOOST)
    #
    #              nb:  iPSC  NPC   RG
    #   self: iPSC  [  0.4,  0.4,  0.2 ]
    #         NPC   [  0.4,  0.8,  0.6 ]
    #         RG    [  0.2,  0.6,  2.0 ]
    g["RG_ADHESION_MATRIX"]  = [0.4, 0.4, 0.2,
                                 0.4, 0.8, 0.6,
                                 0.2, 0.6, 2.0]
    g["RG_REPULSION_MATRIX"] = [4.0] * 9        # [nN/µm]
    g["RG_EPITHELIAL_ADHESION_BOOST"] = 3.0     # [-]


# ---------------------------------------------------------------------------
# configure_layers — full layer sequence with two inserted layers
# ---------------------------------------------------------------------------
def configure_layers(model, g: dict) -> None:
    """Build the full simulation layer sequence for the RG variant.

    This variant CANNOT call g['_build_default_layers']() because it must
    insert two new layers (L3b, L6b) between default layers.  The relevant
    portions of the default sequence are replicated inline here.

    Any booleans from PARAMS that affect which default layers are created are
    read from g (which contains the merged model.py globals).
    """
    _env = g.get("env")

    # --- Register new RG env properties -------------------------------------------
    if _env is not None:
        _register_rg_env_properties(_env, g)

    # --- Register new RTC agent functions on the CELL agent -----------------------
    CELL_agent = model.Agent("CELL")

    rg_diff_code = (_HERE / "cell_rg_differentiation.cpp").read_text()
    rg_diff_fn = CELL_agent.newRTCFunction("cell_rg_differentiation", rg_diff_code)
    rg_diff_fn.setMessageInput("cell_spatial_location_message")

    rg_pol_code = (_HERE / "cell_rg_polarity_update.cpp").read_text()
    CELL_agent.newRTCFunction("cell_rg_polarity_update", rg_pol_code)

    # --- Convenience flags (read from merged globals) -----------------------------
    INCLUDE_DIFFUSION           = g.get("INCLUDE_DIFFUSION",          True)
    INCLUDE_CELLS               = g.get("INCLUDE_CELLS",              True)
    INCLUDE_CELL_CELL_INTERACTION = g.get("INCLUDE_CELL_CELL_INTERACTION", True)
    INCLUDE_CELL_CYCLE          = g.get("INCLUDE_CELL_CYCLE",         False)
    INCLUDE_VASCULARIZATION     = g.get("INCLUDE_VASCULARIZATION",    False)
    INCLUDE_FIBRE_NETWORK       = g.get("INCLUDE_FIBRE_NETWORK",      False)
    INCLUDE_FOCAL_ADHESIONS     = g.get("INCLUDE_FOCAL_ADHESIONS",    False)
    INCLUDE_LUMEN               = g.get("INCLUDE_LUMEN",              False)
    ORGANOID_ASSAY              = g.get("ORGANOID_ASSAY",             False)
    MOVING_BOUNDARIES           = g.get("MOVING_BOUNDARIES",          False)
    INCLUDE_VASCULAR_CELL_RECRUITMENT = g.get("INCLUDE_VASCULAR_CELL_RECRUITMENT", False)
    INCLUDE_CELL_FNODE_REPULSION = g.get("INCLUDE_CELL_FNODE_REPULSION", False)
    INCLUDE_NETWORK_REMODELING  = g.get("INCLUDE_NETWORK_REMODELING", False)
    HETEROGENEOUS_DIFFUSION     = g.get("HETEROGENEOUS_DIFFUSION",    False)

    # --- L0: VASC (skipped — INCLUDE_VASCULARIZATION = False) ---

    # --- L1: Agent locations ---
    model.newLayer("L1_Agent_Locations").addAgentFunction("BCORNER", "bcorner_output_location_data")
    if INCLUDE_DIFFUSION:
        model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_grid_location_data")
    if INCLUDE_CELLS:
        model.Layer("L1_Agent_Locations").addAgentFunction("CELL", "cell_spatial_location_data")

    # --- L2: Boundary interactions ---
    if INCLUDE_DIFFUSION:
        model.newLayer("L2_ECM_Boundary_Interactions").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")

    # --- L3: Metabolism (no cell cycle in RG variant) ---
    if INCLUDE_CELLS and INCLUDE_DIFFUSION:
        model.newLayer("L3_Metabolism").addAgentFunction("CELL", "cell_ecm_interaction_metabolism")

    # --- L3b: RG differentiation (NEW) ---
    if INCLUDE_CELLS:
        model.newLayer("L3b_RG_Differentiation").addAgentFunction("CELL", "cell_rg_differentiation")

    # --- L4-L6: ECM Csp update, diffusion, boundary ---
    if INCLUDE_DIFFUSION:
        model.newLayer("L4_ECM_Csp_Update").addAgentFunction("ECM", "ecm_Csp_update")
        if HETEROGENEOUS_DIFFUSION and INCLUDE_FIBRE_NETWORK:
            model.newLayer("L4_ECM_Dsp_Update").addAgentFunction("ECM", "ecm_Dsp_update")
        model.newLayer("L5_Diffusion").addAgentFunction("ECM", "ecm_ecm_interaction")
        model.newLayer("L6_Diffusion_Boundary").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")

    # --- L6b: RG polarity update (NEW) ---
    if INCLUDE_CELLS:
        model.newLayer("L6b_RG_Polarity_Update").addAgentFunction("CELL", "cell_rg_polarity_update")

    # --- L7: Cell-cell interaction and stress ---
    if INCLUDE_CELLS and INCLUDE_CELL_CELL_INTERACTION:
        model.newLayer("L7_CELL_CELL_Interaction").addAgentFunction("CELL", "cell_cell_interaction")
    if INCLUDE_CELLS:
        model.newLayer("L7_CELL_Stress_State_Update").addAgentFunction("CELL", "cell_stress_state_update")

    # --- L8: Movement ---
    if INCLUDE_CELLS:
        model.newLayer("L8_CELL_Movement").addAgentFunction("CELL", "cell_move")
    if MOVING_BOUNDARIES:
        model.newLayer("L8_BCORNER_Movement").addAgentFunction("BCORNER", "bcorner_move")
        model.newLayer("L8_ECM_Movement").addAgentFunction("ECM", "ecm_move")


# ---------------------------------------------------------------------------
# Internal helper — register RG-specific env properties
# ---------------------------------------------------------------------------
def _register_rg_env_properties(env, g: dict) -> None:
    """Register all RG-specific environment properties.

    Guards each registration with a try/except so re-running the same Python
    process (e.g. during interactive testing) is safe.
    """
    def _safe(register_fn):
        try:
            register_fn()
        except Exception:
            pass  # property already registered

    _safe(lambda: env.newPropertyFloat("RG_COMMIT_RATE",
                                        g.get("RG_COMMIT_RATE", 2e-5)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_AUTOCRINE_RATE",
                                        g.get("RG_COMMIT_AUTOCRINE_RATE", 5e-4)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_PARACRINE_RATE",
                                        g.get("RG_COMMIT_PARACRINE_RATE", 1e-4)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_THRESHOLD_NPC",
                                        g.get("RG_COMMIT_THRESHOLD_NPC", 0.35)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_THRESHOLD_RG",
                                        g.get("RG_COMMIT_THRESHOLD_RG", 0.70)))
    _safe(lambda: env.newPropertyFloat("RG_EPITHELIAL_RATE",
                                        g.get("RG_EPITHELIAL_RATE", 1e-4)))
    _safe(lambda: env.newPropertyFloat("RG_POLARITY_TAU",
                                        g.get("RG_POLARITY_TAU", 600.0)))
    _safe(lambda: env.newPropertyFloat("RG_POLARITY_GRADIENT_THRESHOLD",
                                        g.get("RG_POLARITY_GRADIENT_THRESHOLD", 1e-6)))
    _safe(lambda: env.newPropertyFloat("RG_SUBSTRATE_K",
                                        g.get("RG_SUBSTRATE_K", 0.5)))
    _safe(lambda: env.newPropertyFloat("RG_SUBSTRATE_Z0",
                                        g.get("RG_SUBSTRATE_Z0", 0.0)))
    _safe(lambda: env.newPropertyFloat("RG_APICAL_BIAS_RG",
                                        g.get("RG_APICAL_BIAS_RG", 2e-3)))
    _safe(lambda: env.newPropertyFloat("RG_APICAL_BIAS_NPC",
                                        g.get("RG_APICAL_BIAS_NPC", 5e-4)))
    _safe(lambda: env.newPropertyArrayFloat("RG_ADHESION_MATRIX",
                                             g.get("RG_ADHESION_MATRIX",
                                                   [0.4, 0.4, 0.2,
                                                    0.4, 0.8, 0.6,
                                                    0.2, 0.6, 2.0])))
    _safe(lambda: env.newPropertyArrayFloat("RG_REPULSION_MATRIX",
                                             g.get("RG_REPULSION_MATRIX",
                                                   [4.0] * 9)))
    _safe(lambda: env.newPropertyFloat("RG_EPITHELIAL_ADHESION_BOOST",
                                        g.get("RG_EPITHELIAL_ADHESION_BOOST", 3.0)))
