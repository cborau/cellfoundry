"""Copy this directory to variants/<your_name>/ and fill in the relevant hooks.

Run unchanged: python model.py --variant variant_template
This uses core parameter defaults and the complete generic schedule below.

Only configure_layers(ctx) is REQUIRED. The other four functions and three
dictionaries below are the complete set of OPTIONAL variant entry points.
Python helpers can have any name, but the framework only calls recognized hooks.
Do not import model.py: the construction context supplies the native objects.

Guide: docs/auto/wiki/Tutorial-Model-Variants.md
Worked example: docs/auto/wiki/Tutorial-Variant-Cell-Markers.md
"""

# OPTIONAL: introduce new parameter names, available to JSON/optimizer overrides.
# Do not redeclare core names (N, N_SPECIES, TIME_STEP, etc.) here.
PARAM_DEFAULTS = {
    # "MY_RATE": 0.1,
    # "CUSTOM_CAPACITY": 100,
}

# OPTIONAL: change core parameters or defaults introduced above.
# Core/default values -> PARAMS -> user JSON (highest priority).
# Structural grid/array/kernel settings remain core-controlled; see the guide.
PARAMS = {
    # "SAVE_PICKLE": True,
    # "MY_RATE": 0.2,
}

# OPTIONAL: replace a core RTC file BEFORE its function is registered.
# Keep the kernel's existing function name and compatible message signature.
# Adding a new function uses register_functions(), not this dictionary.
FILES = {
    # "cell_cycle_file": "variants/your_name/cell_cycle.cpp",
}


def validate_config(config):
    """OPTIONAL, once during construction: reject unsupported effective settings.

    Receives the resolved configuration, including JSON/optimizer overrides.
    Treat it as read-only. Raise ValueError with an actionable explanation.
    This hook does not construct agents, edit parameters or run every step.
    """
    # if not config["INCLUDE_CELLS"]:
    #     raise ValueError("This variant requires INCLUDE_CELLS=True")
    # if config["MY_RATE"] < 0:
    #     raise ValueError("MY_RATE must be non-negative")
    pass


def declare_model(ctx):
    """OPTIONAL, once during construction: describe variables, types and messages.

    Core agents already exist. No live agent instances exist yet. Define new
    GPU environment properties explicitly; a Python parameter is not one.
    Reserve new custom-ID populations HERE, before message bounds are fixed.
    """
    # Extend an enabled core agent, with a constant default (no initializer needed):
    # ctx.agents["CELL"].newVariableFloat("my_value", 0.0)
    # ctx.env.newPropertyFloat("MY_RATE", ctx.config["MY_RATE"])

    # Create a new type, an initial state, and an initially empty birth reservation:
    # agent = ctx.model.newAgent("CUSTOM")
    # agent.newVariableInt("id")
    # agent.newVariableFloat("value", 0.0)
    # agent.newState("active")
    # ctx.agents["CUSTOM"] = agent
    # ids = ctx.add_population("CUSTOM", count=0,
    #                          capacity=ctx.config["CUSTOM_CAPACITY"], state="active")
    # message = ctx.model.newMessageBucket("custom_report")
    # message.setBounds(ids.begin, ids.end)  # Upper bound is exclusive.
    # message.newVariableFloat("value")
    # ctx.messages["custom_report"] = message
    # For GPU births use variant_ids.cuh with the counter AND exhaustion flag.
    pass


def register_functions(ctx):
    """OPTIONAL, once during construction: register and bind extra GPU functions.

    Declarations are complete. Files must exist and their signatures must match
    their message bindings. Registration alone never schedules execution.
    """
    # directory = ctx.root / "variants" / ctx.name
    # function = ctx.agents["CUSTOM"].newRTCFunctionFile(
    #     "custom_publish", str(directory / "custom_publish.cpp"))
    # function.setInitialState("active")
    # function.setEndState("active")
    # function.setMessageOutput("custom_report")
    # ctx.functions["CUSTOM.custom_publish"] = function  # Optional handle storage.

    # A receiving function must have the matching MessageBucket input signature:
    # receiver.setMessageInput("custom_report")
    # A CELL function creating CUSTOM must explicitly bind its birth target:
    # birth.setAgentOutput("CUSTOM", "active")
    # A function returning flamegpu::DEAD must explicitly allow death:
    # aging.setAllowAgentDeath(True)
    pass


def register_runtime(ctx):
    """OPTIONAL, once during construction: register callbacks for later execution.

    The runtime.py filename is a convenience, not a framework entry point.
    Uncomment only the imports/registrations actually needed by your variant.
    Per-agent callbacks do not run for GPU births; initialize daughters in C++.
    """
    # from .runtime import initialize_cell, initialize_custom, Runtime
    # ctx.add_agent_initializer("CELL", initialize_cell)
    # ctx.add_agent_initializer("CUSTOM", initialize_custom)
    # ctx.cell_vtk_scalars.append(("my_value", "my_value", "float"))
    # ctx.cell_vtk_vectors.append(("my_vector", "my_x", "my_y", "my_z"))

    # if ctx.config["SAVE_PICKLE"]:
    #     runtime = Runtime(ctx)
    #     ctx.add_init_function(runtime.initialize)
    #     ctx.add_step_function(runtime.step)
    #     ctx.add_exit_function(runtime.finish)
    # Keep intermediate GPU/host ordering explicit in configure_layers().
    pass


# REQUIRED hook: edit this complete schedule directly; no core layers are added
# implicitly. Place your own functions after their producers and before consumers.
def configure_layers(ctx):
    """REQUIRED: the entire GPU schedule for all enabled generic features."""
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

    # Example addition (after its producer/consumer ordering has been decided):
    # model.newLayer("Custom_Publication").addAgentFunction("CUSTOM", "custom_publish")
    # model.newLayer("Custom_Response").addAgentFunction("CELL", "cell_custom_response")
