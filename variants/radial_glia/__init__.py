"""
variants/radial_glia/__init__.py

Radial-glia (RG) differentiation variant.

Starting from a monolayer of iPSC (cell_type=0), cells progressively
differentiate to NEP (neuroepithelial progenitor, cell_type=1) and then
RG (cell_type=2) in response to a locally secreted morphogen (ECM
species 2).  Differentiated cells develop apical-basal polarity and form
rosette-like structures.

Key additions versus the base model:
  - cell_cycle.cpp override                — proliferation with asymmetric RG division
  - cell_rg_differentiation.cpp  — commitment + cell-type switching ODE
  - cell_rg_polarity_update.cpp  — apical-vector relaxation toward sp-2 gradient
  - cell_spatial_location_data.cpp override — broadcasts RG variables in messages
  - cell_cell_interaction.cpp override     — type-pair adhesion matrix + epi boost
  - cell_move.cpp override                 — substrate spring + apical bias

Layer sequence (full, with inserted layers):
  L1   cell_spatial_location_data   ← OVERRIDE
  L2   ECM boundary conditions
  L2b  cell_cycle                   ← OVERRIDE (RG variant)
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
    # --- Domain ---
    # 1000×1000×50 µm domain. Z is only 50 µm (monolayer: cells never stack more than
    # 2-3 diameters). With N=6 the shortest side gives dx=10 µm (2.5× finer than
    # default 25 µm) at 101×101×6 = 61,206 ECM agents — similar to the default count.
    "BOUNDARY_COORDS":            [500.0, -500.0, 500.0, -500.0, 25.0, -25.0],
    "N":                          6,

    # --- Assay type ---
    "MONOLAYER_ASSAY":            True,
    "ORGANOID_ASSAY":             False,
    "MONOLAYER_CELL_TYPE_RATIOS": [1, 0, 0],   # all iPSC at start
    # Biological seeding density: 10,000 cells/cm² on day 3 post-replating.
    # 1000×1000 µm domain = 0.01 cm² → 100 cells match protocol density exactly.
    # Near-domain-wide radius gives near-uniform placement across the xy-plane.
    "MONOLAYER_CLUSTER_RADIUS":   490.0,         # [µm]  near-uniform seeding across ±500 µm domain
    "MONOLAYER_Z":                -15.0,          # [µm]  substrate sits at z=-25; place cells 10 µm above floor
    # Wrap cells in x and y to avoid border pile-up in the finite domain.
    # The override cell_move.cpp already implements wrapf() for all axes.
    "PERIODIC_BOUNDARIES_FOR_CELLS": True,

    # --- Agents ---
    "INCLUDE_CELLS":              True,
    "INCLUDE_CELL_CELL_INTERACTION": True,
    "INCLUDE_CELL_CYCLE":         True,
    # Cell-cycle timing — biological: ~42h for iPSC and NEP; RG ~52h.
    # Phases scale from the base-model 24h reference (G1:S:G2:M = 10:8:4:2 h → same proportions).
    "CYCLE_PHASE_G1_DURATION":    [75600.0, 75600.0, 93600.0],   # [s]  iPSC/NEP 21h, RG 26h
    "CYCLE_PHASE_S_DURATION":     [36000.0, 36000.0, 43200.0],   # [s]  iPSC/NEP 10h, RG 12h
    "CYCLE_PHASE_G2_DURATION":    [18000.0, 18000.0, 21600.0],   # [s]  iPSC/NEP  5h, RG  6h
    "CYCLE_PHASE_M_DURATION":     [21600.0, 21600.0, 28800.0],   # [s]  iPSC/NEP  6h, RG  8h  → totals 42h/42h/52h
    # RG division-rate multiplier slightly suppressed (0.7) to reflect the specialised
    # Notch-mediated cell cycle; the longer cycle duration already handles most of the delay.
    "DIVISION_RATE_MULTIPLIER":   [1.0, 1.0, 0.7],               # [-]  per cell type
    # Fraction of RG divisions that are symmetric (both daughters stay RG).
    # The remaining fraction are asymmetric (one RG + one NEP progenitor).
    "RG_SYMMETRIC_DIVISION_PROB": 0.25,                          # [-]  probability of symmetric RG division
    "INCLUDE_DIFFUSION":          True,
    "INCLUDE_VASCULARIZATION":    False,
    "INCLUDE_FIBRE_NETWORK":      False,
    "INCLUDE_FOCAL_ADHESIONS":    False,
    "INCLUDE_LUMEN":              False,

    # --- Cell geometry ---
    "MIN_ROSETTE_SIZE":           12,                        # minimum RG cells per cluster to count as a genuine rosette
    "N_CELLS":                    300,
    "CELL_RADIUS":                [10.0, 10.0, 10.0],       # [µm]

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
    # sp0/sp1 unchanged at D=0.1 µm²/s.  sp2 reduced to D=0.03 µm²/s to sharpen
    # the local RG signal relative to the background from distributed NEP cells.
    # Signal-to-noise analysis (2-D Green's function, steady state):
    #   Background from ~600 NEP cells: C_bg ≈ N*m_NEP*q / (A*λ) = 1.5 q_RG  (with m_NEP=0.1)
    #   Single-RG local signal at r=10 µm: C_local ≈ q/(2πD)*K0(R/L) ≈ 6.5 q_RG
    #   Ratio C_local / C_bg ≈ 4.3:1  (vs 0.37:1 with old D=0.1, m_NEP=0.5)
    # Diffusion length: L = sqrt(D/λ) = sqrt(0.03/4e-5) = 27.4 µm (~2.7 cell diameters).
    # Only cells within ~27 µm of a committed RG cluster see a significant sp2 boost.
    # Stability: Fi = 3*0.03*60/100 = 0.054 < 0.5;  λ*dt = 4e-5*60 = 0.0024 < 1.
    "DIFFUSION_COEFF_MULTI":      [0.1, 0.1, 0.03],         # [µm²/s]  sp2 tightened to L=27 µm
    "ECM_DEGRADATION_RATE_MULTI": [0.0, 0.0, 4e-5],         # [1/s]  sp2 half-life 4.8h
    # sp2: ALL six faces zero-flux (-1.0).  With first-order degradation the steady-state
    # sp2 profile is set by the local balance of secretion and degradation; no boundary
    # sink is needed.  With D=0.03 µm²/s and L=27 µm the gradient is steeper than before,
    # giving the polarity update a stronger directional cue near the cluster.
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
    "INIT_ECM_SAT_CONCENTRATION_VALS": [0.0, 0.0, 0.5],     # sp2 secretion target [µM]  — raised from 0.5: lets ECM accumulate more sp2
                                                              # so the drive at high-density zones builds faster.
                                                              # SNR unchanged: q_RG cancels in C_local/C_bg ratio.

    # --- Metabolism / secretion ---
    # sp0 and sp1 consumed by all types; sp2 secreted only by committed cells.
    # DE_NOVO_PRODUCTION[2]=1: RG morphogen (sp2) is synthesised de-novo from cellular
    # metabolism — no intracellular pool is depleted when secreting (PhysiCell-style source
    # term).  sp0 and sp1 are nutrients transported from ECM → cell, so they stay at 0
    # (mass-conserved uptake; no de-novo production by cells).
    "INIT_CELL_PRODUCTION_RATES": [0.0, 0.0, 5e-4],         # [1/s] base for fully committed RG
    "CELL_PRODUCTION_MULTIPLIER": [0.0, 0.1, 1.0],          # type 0: silent, 1: low background (0.1×), 2: full
                                                              # NEP reduced from 0.5→0.1: background C_bg drops 5×
                                                              # while local RG signal (C_local, set by D and λ) is
                                                              # unchanged → C_local/C_bg ≈ 4.3:1 vs old 0.37:1.
                                                              # NEP still contributes enough background to allow
                                                              # first-RG nucleation within the 120h window.
    "DE_NOVO_PRODUCTION":         [0, 0, 1],                 # 1 = de-novo (sp2), 0 = mass-conserved (sp0, sp1)

    # --- Simulation timing ---
    # TIME_STEP=60 s. With dx=10 µm (N=6, 1000×1000×50 domain), D=0.03 µm²/s for sp2:
    # Fi = 3×0.03×60/100 = 0.054 << 0.5 (stable).
    # Simulate day 3 (post-replating) → day 8 (rosette analysis): 5 days = 7200 steps.
    "TIME_STEP":                  60.0,                      # [s]
    "STEPS":                      7200,                      # 5 days: 120 h × 3600 s/h ÷ 60 s/step
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
    "cell_cycle_file":                 str(_HERE / "cell_cycle.cpp"),
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
    g["RG_COMMIT_RATE"]                = 5e-6    # [1/s]  basal commit rate (iPSC only; community-gated).
                                                          #        This encodes the "community effect": a
                                                          #        minimum local cell density is required before
                                                          #        isolated cells can initiate differentiation.
                                                          #        tau = 1/(5e-6+2.5e-6) = 37h; first NEP at ~28h.
                                                          #        x_eq_basal = 5e-6/7.5e-6 = 0.667 < RG threshold;
                                                          #        sp2 autocrine required to cross into RG territory.
    g["RG_COMMIT_AUTOCRINE_RATE"]      = 3.95e-5    # [1/(s·µM)]  morphogen-driven amplification.
                                                          #        x_eq = drive/(drive+k_decay); drive=k_auto*sp2
                                                          #        τ = 1/(k_auto*sp2 + k_decay);
                                                          #          at sp2=0.14µM: τ = 1/(8.4e-6+1e-6) = 29h  (was 62h at 2.5e-5)
                                                          #          at sp2=0.30µM: τ = 1/(18e-6+1e-6) = 14.6h
                                                          #        x_eq at sp2=0.14: 8.4e-6/9.4e-6 = 0.89 >> 0.67
                                                          #        First RG expected ~65h (NEP at ~22h + 43h drive at rising sp2).
    g["RG_COMMIT_INHIBIT_RATE"]        = 2e-5    # [1/s]  Notch-Delta lateral inhibition rate.
                                                          #        x_eq = drive/(drive+k_decay+k_inhibit*delta)
                                                          #        1 RG neighbor (delta~0.08), sp2=0.22: x_eq=0.679 -> borderline
                                                          #        2 RG neighbors (delta~0.16), sp2=0.22: x_eq=0.567 -> inhibited ✓
                                                          #        Boundary at ~2 RG neighbors -> small 7-cell rosettes.
    g["RG_COMMIT_THRESHOLD_NEP"]       = 0.35    # [-]
    g["RG_COMMIT_THRESHOLD_RG"]        = 0.67    # [-]
    g["RG_EPITHELIAL_RATE"]            = 2e-6    # [1/s]  τ ≈ 140 h → polarity develops
                                                          #        only for fully committed RG in 7 days;
                                                          #        also gated on cell_type==2 so NEP
                                                          #        cells never accumulate epi>0
    g["RG_POLARITY_SP2_THRESHOLD"]     = 0.1     # [uM]  sp2 gate for z-bias; at L=50 um: C(75 um)~0.09 < threshold
    g["RG_SUBSTRATE_K"]                = 1e-4    # [nN/µm / (nN·s/µm)] → effective 1/s
                                                          #        stability: λ = K/D·dt = 1e-4/0.4×60 ≈ 0.015 (< 1, stable)
                                                          #        equilibrium height: z_eq = bias·D/K
                                                          #          RG  (bias=8e-4): z_eq ≈  8 µm (rosette elevation)
                                                          #          NEP (bias=5e-4): z_eq ≈  5 µm
                                                          #          iPSC (bias=0):   z_eq =  0 µm (stays on substrate)
    g["RG_SUBSTRATE_Z0"]               = 0.0    # [µm above COORD_BOUNDARY_Z_NEG]
                                                          #        z_rest = COORD_BOUNDARY_Z_NEG + RG_SUBSTRATE_Z0
                                                          #        = -75 + 75 = 0 µm (domain centre),
                                                          #        consistent with MONOLAYER_Z = 0.0
    g["RG_APICAL_BIAS_RG"]             = 8e-4    # [µm/s] for RG type; prevents rapid z-detachment
    g["RG_APICAL_BIAS_NEP"]            = 5e-4    # [µm/s] for NEP (neuroepithelial progenitor) type
    # Adhesion matrix: 3×3 flattened row-major, indexed as [self_type * 3 + nb_type]  [nN/µm]
    # Rows = self cell type; columns = neighbour cell type.
    # Biological rationale:
    #   iPSC (0) — few adhesion molecules; weak attraction to all types
    #   NEP (1)  — N-cadherin begins to be expressed; moderate homotypic adhesion
    #   RG  (2)  — N-cadherin / NCAM rich junctions; strong homotypic adhesion drives
    #              rosette assembly; further boosted by epithelialization_level (×RG_EPITHELIAL_ADHESION_BOOST)
    #              nb:  iPSC  NEP   RG
    #   self: iPSC  [  0.4,  0.4,  0.2 ]
    #         NEP   [  0.4,  0.8,  0.6 ]
    #         RG    [  0.2,  0.6,  1.5 ]   <- max with boost = 1.5 × 2.5 = 3.75 < repulsion 4.0
    #                                         gap maintained at all epithelialization levels
    g["RG_ADHESION_MATRIX"]  = [0.4, 0.4, 0.2,
                                 0.4, 0.8, 0.6,
                                 0.2, 0.6, 1.5]
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
    g["RG_LUMEN_SEARCH_RADIUS"] = 84.0  # [um]       neighbour radius for local lumen centroid (scaled 2× for r=10 µm)
    g["RG_LUMEN_MIN_NEIGHBOURS"] = 2.0  # [-]        minimum alive RG neighbours to enable lumen cue
    g["RG_APICAL_NOISE_AMP"]   = 1e-3  # [-/step]  xy noise std dev for cells outside the morphogen gate
    g["RG_XY_SPRING_K"]    = 5e-6    # [1/s]  xy substrate spring stiffness for NPC/RG;
                                       #        anchor slips at 20 µm so cells can aggregate
    g["RG_XY_BOND_BREAK"]  = 40.0   # [µm]   bond-rupture distance: anchor slips to current
                                       #        position when cell has moved >40 µm from it
    g["RG_COMMIT_DECAY_RATE"] = 1.0e-6  # [1/s]  first-order decay of rg_commit_level.
                                       #          effective_decay = 2.5e-6 + 0.96e-6 = 3.46e-6
                                       #          No sp2:        x_eq = 5e-6/8.46e-6 = 0.591 -> fence holds
                                       #          sp2=0.14uM:    x_eq = 7.8e-6/11.26e-6 = 0.693 -> NPC
                                       #          sp2=0.17uM:    x_eq = 8.4e-6/11.86e-6 = 0.708 -> RG
                                       #        Fence holds at mean sp2; rosette grows where sp2 peaks.
    g["RG_COMMUNITY_MIN_DENSITY"]  = 4.0   # [-]  minimum live spatial neighbours to fully gate
                                       #        the basal commit drive (community effect).
                                       #        community_gate = min(1, n_nbrs / RG_COMMUNITY_MIN_DENSITY)
                                       #        Isolated cells (< 4 neighbours within the
                                       #        spatial search radius) have proportionally
                                       #        reduced basal differentiation drive.



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
    # ECM messages carry mechanical state as well as concentrations.
    if INCLUDE_DIFFUSION or MOVING_BOUNDARIES:
        model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_grid_location_data")
    if INCLUDE_CELLS:
        model.Layer("L1_Agent_Locations").addAgentFunction("CELL", "cell_spatial_location_data")

    # --- L2: Boundary interactions ---
    if INCLUDE_DIFFUSION:
        model.newLayer("L2_ECM_Boundary_Interactions").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")

    # --- L2b: Cell cycle (override enabled; runs before metabolism so newly born
    #          cells are metabolised in the same step they are created) ---
    if INCLUDE_CELLS and INCLUDE_CELL_CYCLE:
        model.newLayer("L2b_Cell_Cycle").addAgentFunction("CELL", "cell_cycle")

    # --- L3: Metabolism ---
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
    if INCLUDE_DIFFUSION or MOVING_BOUNDARIES:
        model.newLayer("L5_Diffusion").addAgentFunction("ECM", "ecm_ecm_interaction")
    if INCLUDE_DIFFUSION:
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
                                        g.get("RG_COMMIT_AUTOCRINE_RATE", 2.5e-5)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_INHIBIT_RATE",
                                        g.get("RG_COMMIT_INHIBIT_RATE", 2e-5)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_THRESHOLD_NEP",
                                        g.get("RG_COMMIT_THRESHOLD_NEP", 0.35)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_THRESHOLD_RG",
                                        g.get("RG_COMMIT_THRESHOLD_RG", 0.67)))
    _safe(lambda: env.newPropertyFloat("RG_EPITHELIAL_RATE",
                                        g.get("RG_EPITHELIAL_RATE", 1e-5)))
    _safe(lambda: env.newPropertyFloat("RG_POLARITY_SP2_THRESHOLD",
                                        g.get("RG_POLARITY_SP2_THRESHOLD", 0.0)))
    _safe(lambda: env.newPropertyFloat("RG_SUBSTRATE_K",
                                        g.get("RG_SUBSTRATE_K", 1e-4)))
    _safe(lambda: env.newPropertyFloat("RG_SUBSTRATE_Z0",
                                        g.get("RG_SUBSTRATE_Z0", 0.0)))
    _safe(lambda: env.newPropertyFloat("RG_APICAL_BIAS_RG",
                                        g.get("RG_APICAL_BIAS_RG", 2e-3)))
    _safe(lambda: env.newPropertyFloat("RG_APICAL_BIAS_NEP",
                                        g.get("RG_APICAL_BIAS_NEP", 5e-4)))
    _safe(lambda: env.newPropertyArrayFloat("RG_ADHESION_MATRIX",
                                             g.get("RG_ADHESION_MATRIX",
                                                   [0.4, 0.4, 0.2,
                                                    0.4, 0.8, 0.6,
                                                    0.2, 0.6, 1.5])))
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
                                        g.get("RG_LUMEN_SEARCH_RADIUS", 84.0)))
    _safe(lambda: env.newPropertyFloat("RG_LUMEN_MIN_NEIGHBOURS",
                                        g.get("RG_LUMEN_MIN_NEIGHBOURS", 2.0)))
    _safe(lambda: env.newPropertyFloat("RG_APICAL_NOISE_AMP",
                                        g.get("RG_APICAL_NOISE_AMP", 0.0)))
    _safe(lambda: env.newPropertyFloat("RG_XY_SPRING_K",
                                        g.get("RG_XY_SPRING_K", 1e-5)))
    _safe(lambda: env.newPropertyFloat("RG_XY_BOND_BREAK",
                                        g.get("RG_XY_BOND_BREAK", 40.0)))
    _safe(lambda: env.newPropertyFloat("RG_COMMIT_DECAY_RATE",
                                        g.get("RG_COMMIT_DECAY_RATE", 1.0e-6)))
    _safe(lambda: env.newPropertyFloat("RG_SYMMETRIC_DIVISION_PROB",
                                        g.get("RG_SYMMETRIC_DIVISION_PROB", 0.25)))
    _safe(lambda: env.newPropertyFloat("RG_COMMUNITY_MIN_DENSITY",
                                        g.get("RG_COMMUNITY_MIN_DENSITY", 4.0)))
