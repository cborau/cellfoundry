"""End-to-end teaching variant: CELL creates short-lived MARKER agents.

The complete generic schedule is written below, followed by four marker layers.
Markers are inert observations: they do not supply chemical mass or forces.
See docs/auto/wiki/Tutorial-Variant-Cell-Markers.md for the walkthrough.
"""
import math
from numbers import Integral


PARAM_DEFAULTS = {
    "MARKER_CAPACITY": 16,          # Total IDs available over the run, not live count.
    "MARKER_LIFETIME_STEPS": 2,     # Includes the birth step's observation.
    "MARKER_DETECTION_RADIUS": 15.0,  # um, Euclidean distance to an ECM node.
}
PARAMS = {
    "N_CELLS": 4, "STEPS": 4, "TIME_STEP": 1.0,
    "INCLUDE_CELLS": True, "INCLUDE_CELL_CELL_INTERACTION": True,
    "INCLUDE_CELL_CYCLE": False, "INCLUDE_DIFFUSION": False,
    "INCLUDE_FIBRE_NETWORK": False, "INCLUDE_FOCAL_ADHESIONS": False,
    "INCLUDE_NETWORK_REMODELING": False, "INCLUDE_CELL_FNODE_REPULSION": False,
    "INCLUDE_VASCULARIZATION": False, "INCLUDE_VASCULAR_CELL_RECRUITMENT": False,
    "INCLUDE_LUMEN": False, "ORGANOID_ASSAY": False, "MONOLAYER_ASSAY": False,
    "VISUALISATION": False, "SHOW_PLOTS": False,
    "SAVE_PICKLE": True, "SAVE_DATA_TO_FILE": True, "SAVE_EVERY_N_STEPS": 1,
}


def validate_config(config):
    if not config["INCLUDE_CELLS"]:
        raise ValueError("cell_markers requires INCLUDE_CELLS=True")
    for name in ("INCLUDE_CELL_CYCLE", "INCLUDE_VASCULAR_CELL_RECRUITMENT"):
        if config[name]:
            raise ValueError(f"cell_markers requires {name}=False: this example uses a fixed CELL population")
    for name in ("MARKER_CAPACITY", "MARKER_LIFETIME_STEPS"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value < 2**31 - 1:
            raise ValueError(f"{name} must be a positive Int")
    if config["MARKER_CAPACITY"] < config["N_CELLS"]:
        raise ValueError("MARKER_CAPACITY must be at least N_CELLS: each CELL emits one marker")
    radius = config["MARKER_DETECTION_RADIUS"]
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("MARKER_DETECTION_RADIUS must be finite and positive")


def declare_model(ctx):
    cell = ctx.agents["CELL"]
    cell.newVariableInt("marker_emitted", 0)
    cell.newVariableInt("own_marker_count", 0)
    ecm = ctx.agents["ECM"]
    ecm.newVariableInt("marker_count", 0)
    ecm.newVariableFloat("marker_exposure", 0.0)  # accumulated marker-seconds

    marker = ctx.model.newAgent("MARKER")
    marker.newState("active")
    marker.newVariableInt("id")
    marker.newVariableInt("owner_cell_id", -1)
    for coordinate in ("x", "y", "z"):
        marker.newVariableFloat(coordinate)
    marker.newVariableInt("age_steps", 0)
    ctx.agents["MARKER"] = marker
    ctx.add_population("MARKER", count=0, capacity=ctx.config["MARKER_CAPACITY"], state="active")

    # Four markers by default: brute force makes the interaction easy to inspect.
    # One publisher feeds two consumers; no existing message schema is changed.
    message = ctx.model.newMessageBruteForce("marker_report")
    message.newVariableInt("id")
    message.newVariableInt("owner_cell_id")
    for coordinate in ("x", "y", "z"):
        message.newVariableFloat(coordinate)
    ctx.messages["marker_report"] = message
    ctx.env.newPropertyInt("MARKER_LIFETIME_STEPS", ctx.config["MARKER_LIFETIME_STEPS"])
    ctx.env.newPropertyFloat("MARKER_DETECTION_RADIUS", ctx.config["MARKER_DETECTION_RADIUS"])


def register_functions(ctx):
    directory = ctx.root / "variants" / ctx.name
    # Prefix the absolute shared-header path so RTC does not depend on cwd.
    source = '#include "' + (ctx.root / "variant_ids.cuh").as_posix() + '"\n'
    source += (directory / "cell_emit_marker.cpp").read_text(encoding="utf-8")
    birth = ctx.agents["CELL"].newRTCFunction("cell_emit_marker", source)
    birth.setAgentOutput("MARKER", "active")

    publish = ctx.agents["MARKER"].newRTCFunctionFile("marker_publish", str(directory / "marker_publish.cpp"))
    publish.setInitialState("active")
    publish.setEndState("active")
    publish.setMessageOutput("marker_report")

    for agent_name, function_name in (("CELL", "cell_read_markers"), ("ECM", "ecm_read_markers")):
        function = ctx.agents[agent_name].newRTCFunctionFile(function_name, str(directory / (function_name + ".cpp")))
        function.setMessageInput("marker_report")

    age = ctx.agents["MARKER"].newRTCFunctionFile("marker_age", str(directory / "marker_age.cpp"))
    age.setInitialState("active")
    age.setEndState("active")
    age.setAllowAgentDeath(True)


def register_runtime(ctx):
    from .runtime import initialize_cell, Metrics
    ctx.add_agent_initializer("CELL", initialize_cell)
    ctx.cell_vtk_scalars.extend([
        ("marker_emitted", "marker_emitted", "int"),
        ("own_marker_count", "own_marker_count", "int"),
    ])
    if ctx.config["SAVE_PICKLE"]:
        metrics = Metrics(ctx)
        ctx.add_init_function(metrics.initialize)
        ctx.add_step_function(metrics.step)
        ctx.add_exit_function(metrics.finish)


def configure_layers(ctx):
    """Complete generic schedule, followed by the entire marker lifecycle."""
    model, config = ctx.model, ctx.config
    HETEROGENEOUS_DIFFUSION = config["HETEROGENEOUS_DIFFUSION"]
    INCLUDE_CELLS = config["INCLUDE_CELLS"]
    INCLUDE_CELL_CELL_INTERACTION = config["INCLUDE_CELL_CELL_INTERACTION"]
    INCLUDE_CELL_CYCLE = config["INCLUDE_CELL_CYCLE"]
    INCLUDE_CELL_FNODE_REPULSION = config["INCLUDE_CELL_FNODE_REPULSION"]
    INCLUDE_DIFFUSION = config["INCLUDE_DIFFUSION"]
    INCLUDE_FIBRE_NETWORK = config["INCLUDE_FIBRE_NETWORK"]
    INCLUDE_FOCAL_ADHESIONS = config["INCLUDE_FOCAL_ADHESIONS"]
    INCLUDE_LUMEN = config["INCLUDE_LUMEN"]
    INCLUDE_NETWORK_REMODELING = config["INCLUDE_NETWORK_REMODELING"]
    INCLUDE_VASCULARIZATION = config["INCLUDE_VASCULARIZATION"]
    INCLUDE_VASCULAR_CELL_RECRUITMENT = config["INCLUDE_VASCULAR_CELL_RECRUITMENT"]
    MONOLAYER_ASSAY = config["MONOLAYER_ASSAY"]
    MOVING_BOUNDARIES = config["MOVING_BOUNDARIES"]
    MULTISCALE_DIFFUSION = config["MULTISCALE_DIFFUSION"]
    ORGANOID_ASSAY = config["ORGANOID_ASSAY"]

    # L0: VASC concentration update — runs BEFORE L1 so the ECM grid message
    # (broadcast in L1) already reflects the VASC-imposed concentration floor.
    if INCLUDE_VASCULARIZATION:
        model.newLayer("L0_VASC_Bucket_Locations").addAgentFunction("VASC", "vasc_bucket_location_data")
        model.newLayer("L0_VASC_Csp_Update").addAgentFunction("VASC", "vasc_Csp_update")
        model.newLayer("L0_VASC_Spatial_Locations").addAgentFunction("VASC", "vasc_spatial_location_data")
        model.newLayer("L0_ECM_VASC_Csp_Update").addAgentFunction("ECM", "ecm_vasc_Csp_update")
        if INCLUDE_CELLS and INCLUDE_VASCULAR_CELL_RECRUITMENT:
            model.newLayer("L0_VASC_Cell_Spawn").addAgentFunction("VASC", "vasc_ecm_cell_spawn")

    if MULTISCALE_DIFFUSION:
        # Array3D producers append within a parent step, so pin before the one
        # L1 broadcast instead of attempting to overwrite the same message.
        model.newLayer("L0_ECM_Diffusion_Boundary").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
    # L1: Agent_Locations
    model.newLayer("L1_Agent_Locations").addAgentFunction("BCORNER", "bcorner_output_location_data")
    # The ECM message carries both concentration data and mechanical state.
    # Moving-boundary ECM mechanics (and VASC advection) need it even when
    # soluble-factor diffusion is disabled.
    if INCLUDE_DIFFUSION or MOVING_BOUNDARIES:
        model.Layer("L1_Agent_Locations").addAgentFunction("ECM", "ecm_grid_location_data")
    if INCLUDE_CELLS:
        model.Layer("L1_Agent_Locations").addAgentFunction("CELL", "cell_spatial_location_data")
        if INCLUDE_FOCAL_ADHESIONS:
            model.newLayer("L1_CELL_Locations_2").addAgentFunction("CELL", "cell_bucket_location_data")  # these functions share data of the same agent, so must be in separate layers
    if INCLUDE_CELLS and (ORGANOID_ASSAY or MONOLAYER_ASSAY) and INCLUDE_LUMEN:
        model.newLayer("L1_LUMEN_Locations").addAgentFunction("LUMEN", "lumen_spatial_location_data")
    if INCLUDE_FIBRE_NETWORK:
        model.newLayer("L1_FNODE_Locations_1").addAgentFunction("FNODE", "fnode_spatial_location_data")
        # These functions share data of the same agent, so must be in separate layers
        model.newLayer("L1_FNODE_Locations_2").addAgentFunction("FNODE", "fnode_bucket_location_data")

    # L2: Boundary_Interactions
    if INCLUDE_DIFFUSION and not MULTISCALE_DIFFUSION:
        model.newLayer("L2_ECM_Boundary_Interactions").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
    if INCLUDE_FIBRE_NETWORK:
        model.newLayer("L2_FNODE_Boundary_Interactions").addAgentFunction("FNODE", "fnode_boundary_interaction")
    if INCLUDE_FIBRE_NETWORK and INCLUDE_CELLS and INCLUDE_NETWORK_REMODELING:
        model.newLayer("L2_CELL_FNODE_Remodel").addAgentFunction("CELL", "cell_fnode_remodel")
        model.newLayer("L2_FNODE_Remodel").addAgentFunction("FNODE", "fnode_remodel")
        model.newLayer("L2_FNODE_Remodel_Apply").addAgentFunction("FNODE", "fnode_apply_remodel_updates")
        model.newLayer("L2_FNODE_Update_Links").addAgentFunction("FNODE", "fnode_update_links")

    # L3: Metabolism & Cell Cycle
    if INCLUDE_CELLS and INCLUDE_DIFFUSION:
        model.newLayer("L3_Metabolism").addAgentFunction("CELL", "cell_ecm_interaction_metabolism")
    if INCLUDE_CELLS and INCLUDE_CELL_CYCLE:
        model.newLayer("L3_Cell_MaxID_Update").addAgentFunction("CELL", "cell_MaxID_update")
        model.newLayer("L3_Cell_Cycle").addAgentFunction("CELL", "cell_cycle")
        if INCLUDE_FOCAL_ADHESIONS:
            model.newLayer("L3_Cell_Bucket_PostCycle").addAgentFunction("CELL", "cell_bucket_location_data")
            model.newLayer("L3_FOCAD_PostCycle_Update").addAgentFunction("FOCAD", "focad_post_cycle_update")
    if INCLUDE_DIFFUSION:
        # L4_ECM_Csp_Update
        model.newLayer("L4_ECM_Csp_Update").addAgentFunction("ECM", "ecm_Csp_update")
        if HETEROGENEOUS_DIFFUSION and INCLUDE_FIBRE_NETWORK:
            model.newLayer("L4_ECM_Dsp_Update").addAgentFunction("ECM", "ecm_Dsp_update")
        if INCLUDE_CELLS and (ORGANOID_ASSAY or MONOLAYER_ASSAY) and INCLUDE_LUMEN and HETEROGENEOUS_DIFFUSION:
            model.newLayer("L4_ECM_Dsp_Lumen_Update").addAgentFunction("ECM", "ecm_Dsp_lumen_update")
    # ECM spring/damping forces run once on the parent clock. In compatibility
    # mode this function also performs the historical concentration update.
    if INCLUDE_DIFFUSION or MOVING_BOUNDARIES:
        model.newLayer("L5_Diffusion").addAgentFunction("ECM", "ecm_ecm_interaction")
    if MULTISCALE_DIFFUSION:
        ctx.add_multiscale_diffusion_layers()
    if INCLUDE_DIFFUSION:
        # L6_Diffusion_Boundary (called twice to ensure concentration at boundaries is properly shown visually)
        model.newLayer("L6_Diffusion_Boundary").addAgentFunction("ECM", "ecm_boundary_concentration_conditions")
    if INCLUDE_FIBRE_NETWORK:
        # L7_Fibre_Network Mechanical interactions
        model.newLayer("L7_FNODE_Repulsion").addAgentFunction("FNODE", "fnode_fnode_spatial_interaction")
        model.newLayer("L7_FNODE_Network_Mechanics").addAgentFunction("FNODE", "fnode_fnode_bucket_interaction")
        if INCLUDE_FOCAL_ADHESIONS:
            model.newLayer("L7_FOCAD_Mechanics").addAgentFunction("FOCAD", "focad_fnode_interaction")
            # These FOCAD location functions are placed here because they require updated force information to be broadcasted to  FNODE and CELL update functions
            model.newLayer("L7_FOCAD_Locations_1").addAgentFunction("FOCAD", "focad_spatial_location_data")
            model.newLayer("L7_FOCAD_Locations_2").addAgentFunction("FOCAD", "focad_bucket_location_data")
            model.newLayer("L7_FNODE_Force_Update").addAgentFunction("FNODE", "fnode_focad_interaction")
            model.newLayer("L7_CELL_Stress_Update").addAgentFunction("CELL", "cell_focad_update")

    if INCLUDE_CELLS and INCLUDE_CELL_CELL_INTERACTION:
        model.newLayer("L7_CELL_CELL_Interaction").addAgentFunction("CELL", "cell_cell_interaction")
    if INCLUDE_CELLS and INCLUDE_FIBRE_NETWORK and INCLUDE_CELL_FNODE_REPULSION:
        model.newLayer("L7_CELL_FNODE_Repulsion").addAgentFunction("CELL", "cell_fnode_repulsion")
        model.newLayer("L7_FNODE_CELL_Repulsion").addAgentFunction("FNODE", "fnode_cell_repulsion")
    if INCLUDE_CELLS and ORGANOID_ASSAY and INCLUDE_LUMEN:
        model.newLayer("L7_LUMEN_LUMEN_Interaction").addAgentFunction("LUMEN", "lumen_lumen_interaction")
        model.newLayer("L7_LUMEN_CELL_Interaction").addAgentFunction("LUMEN", "lumen_cell_interaction")
        model.newLayer("L7_CELL_LUMEN_Interaction").addAgentFunction("CELL", "cell_lumen_interaction")
    # Unified nucleus stress finalization
    if INCLUDE_CELLS:
        model.newLayer("L7_CELL_Stress_State_Update").addAgentFunction("CELL", "cell_stress_state_update")

    # L8_Agent_Movement
    if INCLUDE_CELLS:
        model.newLayer("L8_CELL_Movement").addAgentFunction("CELL", "cell_move")
        if INCLUDE_FOCAL_ADHESIONS:
            # Re-broadcast CELL bucket after movement so FOCAD anchor update
            # reads post-move anchor positions instead of stale L1 data.
            model.newLayer("L8_CELL_Bucket_Post_Move").addAgentFunction("CELL", "cell_bucket_location_data")
    if INCLUDE_CELLS and ORGANOID_ASSAY and INCLUDE_LUMEN:
        model.newLayer("L8_LUMEN_Movement").addAgentFunction("LUMEN", "lumen_move")
        model.newLayer("L8_CELL_LUMEN_Secretion").addAgentFunction("CELL", "cell_lumen_secretion")
    if INCLUDE_FIBRE_NETWORK:
        model.newLayer("L8_FNODE_Movement").addAgentFunction("FNODE", "fnode_move")
        if INCLUDE_FOCAL_ADHESIONS:
            # Broadcast FNODE post-move positions into dedicated message list
            # so focad_move reads current-step coordinates.
            model.newLayer("L8_FNODE_Locations_Post_Move").addAgentFunction("FNODE", "fnode_bucket_location_data_postmove")
            # Sync FOCAD anchor (x_i) with post-move cell position before focad_move,
            # so the ori vector (x_i - x) is consistent at step end.
            model.newLayer("L8_FOCAD_Anchor_Post_Move").addAgentFunction("FOCAD", "focad_anchor_update")
            model.newLayer("L8_FOCAD_Movement").addAgentFunction("FOCAD", "focad_move")
    # If boundaries are not moving, the ECM grid does not need to be updated
    if MOVING_BOUNDARIES:
        model.newLayer("L8_BCORNER_Movement").addAgentFunction("BCORNER", "bcorner_move")
        model.newLayer("L8_ECM_Movement").addAgentFunction("ECM", "ecm_move")
        if INCLUDE_VASCULARIZATION:
            # Refresh the ECM Array3D after ecm_move.  Otherwise vasc_move
            # consumes the pre-force/pre-move velocity broadcast in L1.
            model.newLayer("L8_ECM_Locations_Post_Move").addAgentFunction(
                "ECM", "multiscale_ecm_velocity_output" if MULTISCALE_DIFFUSION else "ecm_grid_location_data")
            model.newLayer("L8_VASC_Movement").addAgentFunction("VASC", "vasc_move")

    # The custom sequence uses post-move positions and is fully visible here.
    model.newLayer("M1_CELL_Emit").addAgentFunction("CELL", "cell_emit_marker")
    model.newLayer("M2_MARKER_Publish").addAgentFunction("MARKER", "marker_publish")
    model.newLayer("M3_Read_Markers").addAgentFunction("CELL", "cell_read_markers")
    model.Layer("M3_Read_Markers").addAgentFunction("ECM", "ecm_read_markers")
    model.newLayer("M4_MARKER_Age").addAgentFunction("MARKER", "marker_age")
