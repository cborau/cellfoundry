"""Small teaching model: stationary cells accumulate a signal and switch on.

No mechanics, diffusion or births run. See Tutorial-Model-Variants.md for the
complete walkthrough. This variant deliberately owns its two-layer schedule.
"""
import math


PARAM_DEFAULTS = {"SIGNAL_RATE": 0.1, "SIGNAL_THRESHOLD": 0.25}
PARAMS = {
    "N_CELLS": 4, "STEPS": 4, "TIME_STEP": 1.0,
    "INCLUDE_CELLS": True,
    "INCLUDE_CELL_CELL_INTERACTION": False, "INCLUDE_CELL_CYCLE": False,
    "INCLUDE_DIFFUSION": False, "INCLUDE_FIBRE_NETWORK": False,
    "INCLUDE_FOCAL_ADHESIONS": False, "INCLUDE_NETWORK_REMODELING": False,
    "INCLUDE_CELL_FNODE_REPULSION": False, "INCLUDE_VASCULARIZATION": False,
    "INCLUDE_VASCULAR_CELL_RECRUITMENT": False, "INCLUDE_LUMEN": False,
    "ORGANOID_ASSAY": False, "MONOLAYER_ASSAY": False,
    "MOVING_BOUNDARIES": False,
    "VISUALISATION": False, "SHOW_PLOTS": False,
    "SAVE_PICKLE": True, "SAVE_DATA_TO_FILE": True, "SAVE_EVERY_N_STEPS": 1,
}


def validate_config(config):
    if not config["INCLUDE_CELLS"]:
        raise ValueError("simple_signal requires INCLUDE_CELLS=True")
    for name, value in PARAMS.items():
        if value is False and (name.startswith("INCLUDE_") or
                               name in ("MOVING_BOUNDARIES", "ORGANOID_ASSAY", "MONOLAYER_ASSAY")):
            if config[name]:
                raise ValueError(f"simple_signal does not schedule {name}; keep it False")
    for name in PARAM_DEFAULTS:
        value = config[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def declare_model(ctx):
    cell = ctx.agents["CELL"]
    cell.newVariableFloat("signal", 0.0)
    cell.newVariableInt("signal_on", 0)
    for name in PARAM_DEFAULTS:
        ctx.env.newPropertyFloat(name, ctx.config[name])


def register_functions(ctx):
    directory = ctx.root / "variants" / ctx.name
    cell = ctx.agents["CELL"]
    cell.newRTCFunctionFile("signal_accumulate", str(directory / "signal_accumulate.cpp"))
    cell.newRTCFunctionFile("signal_switch", str(directory / "signal_switch.cpp"))


def initialize_cell(instance, rng):
    instance.setVariableFloat("signal", 0.0)
    instance.setVariableInt("signal_on", 0)


def register_runtime(ctx):
    ctx.add_agent_initializer("CELL", initialize_cell)
    ctx.cell_vtk_scalars.extend([
        ("signal", "signal", "float"), ("signal_on", "signal_on", "int"),
    ])

    def initialize_results(host):
        ctx.runtime_results(host)["SIGNAL_OVER_TIME"] = []

    def record_signal(host):
        cells = host.agent("CELL").getPopulationData()
        ctx.runtime_results(host)["SIGNAL_OVER_TIME"].append({
            "step": host.getStepCounter() + 1,
            "cells": [{"id": cell.getVariableInt("id"),
                       "signal": cell.getVariableFloat("signal"),
                       "signal_on": cell.getVariableInt("signal_on")} for cell in cells],
        })

    ctx.add_init_function(initialize_results)
    ctx.add_step_function(record_signal)


def configure_layers(ctx):
    ctx.model.newLayer("Signal_Accumulate").addAgentFunction("CELL", "signal_accumulate")
    ctx.model.newLayer("Signal_Switch").addAgentFunction("CELL", "signal_switch")
