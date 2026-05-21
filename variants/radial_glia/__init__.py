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
    # ±150 µm domain (set in model.py) → dx = 15 µm/voxel (N=21 → 9261 voxels).
    # 240 cells in a 50 µm disc → A_per_cell ~ pi* R_disc²/N_CELLS -> dnn ~ sqrt(A_per_cell) = R_disc * sqrt(pi/N_CELLS) µm mean NN distance; just within 15 µm interaction radius.
    # 50 µm cluster spans ~13 voxels per axis, giving real per-cell sp2 gradient heterogeneity.
    "MONOLAYER_CLUSTER_RADIUS":   120.0,         # [µm]  for a known/desired dnn -> R_disc = dnn * sqrt(N_CELLS/pi) = 12 * sqrt(240/pi) ≈ 100 µm
    "MONOLAYER_Z":                -150.0,          # [µm]  0 = domain centre
    # Wrap cells in x and y to avoid border pile-up in the finite domain.
    # The override cell_move.cpp already implements wrapf() for all axes.
    "PERIODIC_BOUNDARIES_FOR_CELLS": True,

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
    "N_CELLS":                    500,
    "CELL_RADIUS":                [5.0, 5.0, 5.0],          # [µm]

    # --- Cell motility ---
    "CELL_SPEED_REF":             [5e-4, 3e-4, 1e-4],       # [µm/s]
    "ROTATIONAL_DIFFUSION_RATE":  [2e-3, 1e-3, 2e-4],       # [rad²/s]

    # --- Cell-cell mechanics — scalars disabled; matrices override in configure_layers ---
    "CELL_CELL_ADHESION_K":       [0.0, 0.0, 0.0],          # [nN/µm] (replaced by matrix)
    "CELL_CELL_REPULSION_K":      [4.0, 4.0, 4.0],          # [nN/µm] (fallback; matrix used)
    # DV_MAX caps the cell-cell velocity contribution.  Default = 0.5×CELL_SPEED_REF = 2.5e-4 µm/s,
    # which is so small that adhesion is outcompeted by random motility and cells cannot cluster.
    # Raise to 10×CELL_SPEED_REF so adhesive/repulsive forces dominate self-propulsion.
    "CELL_CELL_DV_MAX":           [5e-3, 3e-3, 1e-3],       # [µm/s]  10× CELL_SPEED_REF

    # --- Diffusion ---
    # sp0/sp1 unchanged at D=0.1 µm²/s.  sp2 also stays at 0.1 µm²/s but now has
    # first-order degradation (ECM_DEGRADATION_RATE_MULTI[2]=4e-5 /s, half-life 4.8h).
    # Diffusion length: L = sqrt(D/lambda) = sqrt(0.1/4e-5) = 50 µm.  Only cells
    # within ~50 µm of a secreting cluster see significant sp2 -> prevents the
    # domain-wide cascade that drove all 240 cells to RG.
    # Stability: Fi = 3*0.1*60/225 = 0.08 < 0.5;  lambda*dt = 4e-5*60 = 0.0024 < 1.
    "DIFFUSION_COEFF_MULTI":      [0.1, 0.1, 0.1],          # [µm²/s]
    "ECM_DEGRADATION_RATE_MULTI": [0.0, 0.0, 4e-5],         # [1/s]  sp2 half-life 4.8h
    # sp2: ALL six faces zero-flux (-1.0).  With first-order degradation the steady-state
    # sp2 profile is set by the local balance of secretion and degradation; no boundary
    # sink is needed.  The gradient points away from secreting cells (toward periphery)
    # and is heterogeneous — driven by the spatial distribution of the committed cluster.
    # Index convention: [0]=+X, [1]=-X, [2]=+Y, [3]=-Y, [4]=+Z, [5]=-Z (substrate).
    "BOUNDARY_CONC_INIT_MULTI":   [[2.5]*6, [2.5]*6, [-1.0]*6],
    "BOUNDARY_CONC_FIXED_MULTI":  [[2.5]*6, [2.5]*6, [-1.0]*6],
    # Cell intracellular starting concentrations: sp2=0 so cells start with no internal morphogen
    "INIT_CELL_CONCENTRATION_VALS": [2.5, 2.5, 0.0],        # species 2 starts at 0
    # ECM voxel starting concentrations: sp2 MUST also be 0 (default is 2.5 for all species).
    # Without this override the ECM grid is pre-loaded with sp2=2.5, the autocrine term fires
    # at full strength from step 0, and all cells commit in ~12 minutes instead of ~24 h.
    "INIT_ECM_CONCENTRATION_VALS":  [2.5, 2.5, 0.0],        # species 2 starts at 0
    # C_sp_sat is the equilibrium ECM sp2 that NPC/RG cells drive their local voxel toward.
    # Without this override C_sp_sat[2]=0, so c1=dt*alpha*k_prod*0=0 and NO sp2 is ever secreted.
    # Caps the local sp2 peak so the autocrine drive near the
    # cluster centre is at most k_auto * C_sat (NPC->RG in ~54h from NPC formation).
    "INIT_ECM_SAT_CONCENTRATION_VALS": [0.0, 0.0, 0.5],     # sp2 secretion target [µM]

    # --- Metabolism / secretion ---
    # sp0 and sp1 consumed by all types; sp2 secreted only by committed cells.
    # DE_NOVO_PRODUCTION[2]=1: RG morphogen (sp2) is synthesised de-novo from cellular
    # metabolism — no intracellular pool is depleted when secreting (PhysiCell-style source
    # term).  sp0 and sp1 are nutrients transported from ECM → cell, so they stay at 0
    # (mass-conserved uptake; no de-novo production by cells).
    "INIT_CELL_PRODUCTION_RATES": [0.0, 0.0, 5e-4],         # [1/s] base for fully committed RG
    "CELL_PRODUCTION_MULTIPLIER": [0.0, 0.5, 1.0],          # type 0: silent, 1: half, 2: full
    "DE_NOVO_PRODUCTION":         [0, 0, 1],                 # 1 = de-novo (sp2), 0 = mass-conserved (sp0, sp1)

    # --- Simulation timing ---
    # TIME_STEP=60 s. With dx=7.5 µm, D=0.1 µm²/s: Fi=0.32 < 0.5, explicit stable.
    # 7 days (168h) covers iPSC reseeding (day 3) through established rosettes (day 7+).
    "TIME_STEP":                  60.0,                      # [s]
    "STEPS":                      10080,                     # 7 days: 168 h × 3600 s/h ÷ 60 s/step
    "SAVE_EVERY_N_STEPS":         60,                        # save every 1h; fine enough to capture type transitions
    "DEBUG_PRINT_INTERVAL":       60,                        # print live stats every 1h of sim time
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
    g["RG_COMMIT_RATE"]                = 5e-6    # [1/s]  basal commit rate.
                                                          #        tau = 1/(5e-6+2.5e-6) = 37h; first NPC at ~28h.
                                                          #        x_eq_basal = 5e-6/7.5e-6 = 0.667 < RG threshold;
                                                          #        sp2 autocrine required to cross into RG territory.
    g["RG_COMMIT_AUTOCRINE_RATE"]      = 2e-5    # [1/(s·µM)]  morphogen-driven amplification.
                                                          #        Slows commitment so the sp2 gradient can
                                                          #        concentrate around the first nucleation site
                                                          #        before neighbours commit.
                                                          #        RG threshold (x_eq=0.70) requires sp2>0.175µM.
                                                          #        With max sp2~0.21µM: RG zone is narrow, biasing
                                                          #        toward 1-2 central rosettes.
    g["RG_COMMIT_INHIBIT_RATE"]        = 1e-6    # [1/s]  Notch-Delta lateral inhibition rate.
                                                          #        Allows each nucleation site to grow to ~10
                                                          #        cells before the inhibitory fence is strong
                                                          #        enough to suppress the next ring.
                                                          #        With k_basal=2e-6, the fence actually holds:
                                                          #        Equilibrium (1 mature RG neighbor, delta~0.12):
                                                          #          effective_decay = 1.5e-6 + 8e-6*0.12 = 2.46e-6
                                                          #          No sp2: x_eq = 2e-6/4.46e-6 = 0.45 -> NPC
                                                          #          sp2=0.20µM: x_eq = 6e-6/8.46e-6 = 0.71 -> RG
    g["RG_COMMIT_THRESHOLD_NPC"]       = 0.35    # [-]
    g["RG_COMMIT_THRESHOLD_RG"]        = 0.67    # [-]
    g["RG_EPITHELIAL_RATE"]            = 2e-6    # [1/s]  τ ≈ 140 h → polarity develops
                                                          #        only for fully committed RG in 7 days;
                                                          #        also gated on cell_type==2 so NPC
                                                          #        cells never accumulate epi>0
    g["RG_POLARITY_SP2_THRESHOLD"]     = 0.1     # [uM]  sp2 gate for z-bias; at L=50 um: C(75 um)~0.09 < threshold
    g["RG_SUBSTRATE_K"]                = 5e-5    # [nN/µm / (nN·s/µm)] → effective 1/s
                                                          #        stability: λ = K/D·dt = 5e-5/0.4×60 ≈ 0.0075 (< 1, stable)
                                                          #        equilibrium height: z_eq = bias·D/K
                                                          #          RG (bias=2e-3): z_eq ≈ 16 µm (rosette migration)
                                                          #          NPC (bias=5e-4): z_eq ≈  4 µm
                                                          #          iPSC (bias=0):   z_eq =  0 µm (stays on substrate)
    g["RG_SUBSTRATE_Z0"]               = 0.0    # [µm above COORD_BOUNDARY_Z_NEG]
                                                          #        z_rest = COORD_BOUNDARY_Z_NEG + RG_SUBSTRATE_Z0
                                                          #        = -75 + 75 = 0 µm (domain centre),
                                                          #        consistent with MONOLAYER_Z = 0.0
    g["RG_APICAL_BIAS_RG"]             = 8e-4    # [µm/s] for RG type; prevents rapid z-detachment
    g["RG_APICAL_BIAS_NPC"]            = 5e-4    # [µm/s] for NPC type
    # Adhesion matrix: 3×3 flattened row-major, indexed as [self_type * 3 + nb_type]  [nN/µm]
    # Rows = self cell type; columns = neighbour cell type.
    # Biological rationale:
    #   iPSC (0) — few adhesion molecules; weak attraction to all types
    #   NPC (1)  — N-cadherin begins to be expressed; moderate homotypic adhesion
    #   RG  (2)  — N-cadherin / NCAM rich junctions; strong homotypic adhesion drives
    #              rosette assembly; further boosted by epithelialization_level (×RG_EPITHELIAL_ADHESION_BOOST)
    #              nb:  iPSC  NPC   RG
    #   self: iPSC  [  0.4,  0.4,  0.2 ]
    #         NPC   [  0.4,  0.8,  0.6 ]
    #         RG    [  0.2,  0.6,  2.2 ]   <- max with boost = 2.2 x 2.5 = 5.5 > repulsion 4.0
    #                                         at epi=0.5: K_adh = 2.2*1.75 = 3.85 < 4.0 (gap maintained)
    g["RG_ADHESION_MATRIX"]  = [0.4, 0.4, 0.2,
                                 0.4, 0.8, 0.6,
                                 0.2, 0.6, 2.2]
    g["RG_REPULSION_MATRIX"] = [4.0, 4.0, 4.0,
                               4.0, 4.0, 4.0,
                               4.0, 4.0, 4.0]  # [nN/µm] RG-RG repulsion = same as others
    g["RG_EPITHELIAL_ADHESION_BOOST"] = 2.5     # [-]  max RG-RG = 1.2×2.5=3.0
    g["RG_COMMIT_NOISE"]     = 7e-5    # [1/s / sqrt(s)]  Ito noise on commitment ODE
                                       # sigma_total(24h) = 1e-4 * sqrt(86400) ~ 0.029
                                       # NPC timing spread ~ +/-0.029 / (5e-6 * 0.65) ~ +/-2.5h
    g["RG_INTRINSIC_APICAL_Z"] = 2e-3  # [-/step]  blend alpha toward (0,0,1) per step
                                       # For RG cells alpha = 2e-3 * commit (~1.4e-3 at commit=0.7)
                                       # -> half-life ~8h: aligns quickly after RG commitment.
                                       # For NPC cells alpha = 2e-3 * epi * commit (much slower;
                                       # at epi=0.1, commit=0.5: half-life ~115h)
    g["RG_LUMEN_BIAS_STRENGTH"] = 4e-3  # [-/step]  XY blend toward local RG-centroid lumen cue
    g["RG_LUMEN_SEARCH_RADIUS"] = 42.0  # [um]       neighbour radius for local lumen centroid
    g["RG_LUMEN_MIN_NEIGHBOURS"] = 2.0  # [-]        minimum alive RG neighbours to enable lumen cue
    g["RG_APICAL_NOISE_AMP"]   = 1e-3  # [-/step]  xy noise std dev for cells outside the morphogen gate
    g["RG_XY_SPRING_K"]    = 5e-6    # [1/s]  xy substrate spring stiffness for NPC/RG;
                                       #        anchor slips at 20 µm so cells can aggregate
    g["RG_XY_BOND_BREAK"]  = 20.0   # [µm]   bond-rupture distance: anchor slips to current
                                       #        position when cell has moved >20 µm from it
    g["RG_COMMIT_DECAY_RATE"] = 1.5e-6  # [1/s]  first-order decay of rg_commit_level;
                                       #        x_eq_basal = 5e-6/(5e-6+2.5e-6) = 0.667 < RG_threshold.
                                       #        With k_inhibit=8e-6, 1 mature RG neighbour (delta~0.12):
                                       #          effective_decay = 2.5e-6 + 0.96e-6 = 3.46e-6
                                       #          No sp2:        x_eq = 5e-6/8.46e-6 = 0.591 -> fence holds
                                       #          sp2=0.14uM:    x_eq = 7.8e-6/11.26e-6 = 0.693 -> NPC
                                       #          sp2=0.17uM:    x_eq = 8.4e-6/11.86e-6 = 0.708 -> RG
                                       #        Fence holds at mean sp2; rosette grows where sp2 peaks.



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

    rg_diff_fn = CELL_agent.newRTCFunctionFile("cell_rg_differentiation", str(_HERE / "cell_rg_differentiation.cpp"))
    rg_diff_fn.setMessageInput("cell_spatial_location_message")

    rg_polarity_fn = CELL_agent.newRTCFunctionFile("cell_rg_polarity_update", str(_HERE / "cell_rg_polarity_update.cpp"))
    rg_polarity_fn.setMessageInput("cell_spatial_location_message")

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
                                        g.get("RG_COMMIT_RATE", 5e-6)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_AUTOCRINE_RATE",
                                        g.get("RG_COMMIT_AUTOCRINE_RATE", 3e-6)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_INHIBIT_RATE",
                                        g.get("RG_COMMIT_INHIBIT_RATE", 1e-6)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_THRESHOLD_NPC",
                                        g.get("RG_COMMIT_THRESHOLD_NPC", 0.35)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_THRESHOLD_RG",
                                        g.get("RG_COMMIT_THRESHOLD_RG", 0.67)))
    _safe(lambda: env.newPropertyFloat("RG_EPITHELIAL_RATE",
                                        g.get("RG_EPITHELIAL_RATE", 1e-5)))
    _safe(lambda: env.newPropertyFloat("RG_POLARITY_SP2_THRESHOLD",
                                        g.get("RG_POLARITY_SP2_THRESHOLD", 0.0)))
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
                                        g.get("RG_EPITHELIAL_ADHESION_BOOST", 2.5)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_NOISE",
                                        g.get("RG_COMMIT_NOISE", 7e-5)))
    _safe(lambda: env.newPropertyFloat("RG_INTRINSIC_APICAL_Z",
                                        g.get("RG_INTRINSIC_APICAL_Z", 0.5)))
    _safe(lambda: env.newPropertyFloat("RG_LUMEN_BIAS_STRENGTH",
                                        g.get("RG_LUMEN_BIAS_STRENGTH", 4e-3)))
    _safe(lambda: env.newPropertyFloat("RG_LUMEN_SEARCH_RADIUS",
                                        g.get("RG_LUMEN_SEARCH_RADIUS", 42.0)))
    _safe(lambda: env.newPropertyFloat("RG_LUMEN_MIN_NEIGHBOURS",
                                        g.get("RG_LUMEN_MIN_NEIGHBOURS", 2.0)))
    _safe(lambda: env.newPropertyFloat("RG_APICAL_NOISE_AMP",
                                        g.get("RG_APICAL_NOISE_AMP", 0.0)))
    _safe(lambda: env.newPropertyFloat("RG_XY_SPRING_K",
                                        g.get("RG_XY_SPRING_K", 1e-5)))
    _safe(lambda: env.newPropertyFloat("RG_XY_BOND_BREAK",
                                        g.get("RG_XY_BOND_BREAK", 20.0)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_DECAY_RATE",
                                        g.get("RG_COMMIT_DECAY_RATE", 1.5e-6)))
