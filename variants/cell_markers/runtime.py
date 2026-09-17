"""Optional Python module for host initialization and output; imported explicitly."""
import pandas as pd


def initialize_cell(instance, rng):
    # Defaults already provide these values; explicit initialization illustrates
    # the per-agent callback. GPU birth fields are set in cell_emit_marker.cpp.
    instance.setVariableInt("marker_emitted", 0)
    instance.setVariableInt("own_marker_count", 0)


class Metrics:
    def __init__(self, ctx):
        self.ctx = ctx

    def initialize(self, host):
        results = self.ctx.runtime_results(host)
        results["MARKER_HISTORY"] = []
        results["MARKER_SUMMARY"] = pd.DataFrame(columns=[
            "step", "live_markers", "emitted_total", "owner_reports", "ecm_exposure_total"])
        results["ECM_MARKER_FINAL"] = pd.DataFrame()

    def step(self, host):
        # Intentionally small teaching populations. For larger models prefer
        # reductions, sparse sampling and dedicated spatial/bucket messages.
        results = self.ctx.runtime_results(host)
        step = host.getStepCounter() + 1
        markers = [{"id": m.getVariableInt("id"),
                    "owner_cell_id": m.getVariableInt("owner_cell_id"),
                    "age_steps": m.getVariableInt("age_steps"),
                    **{axis: m.getVariableFloat(axis) for axis in ("x", "y", "z")}}
                   for m in host.agent("MARKER", "active").getPopulationData()]
        cells = [{"id": c.getVariableInt("id"),
                  "marker_emitted": c.getVariableInt("marker_emitted"),
                  "own_marker_count": c.getVariableInt("own_marker_count")}
                 for c in host.agent("CELL").getPopulationData()]
        results["MARKER_HISTORY"].append({"step": step, "markers": markers, "cells": cells})
        frame = results["MARKER_SUMMARY"]
        frame.loc[len(frame)] = [step, len(markers),
                                sum(c["marker_emitted"] for c in cells),
                                sum(c["own_marker_count"] for c in cells),
                                host.agent("ECM").sumFloat("marker_exposure")]

    def finish(self, host):
        # Generic ECM VTK fields are not extended by CELL's VTK registry.
        # Export these variant-specific ECM fields in the merged results pickle.
        self.ctx.runtime_results(host)["ECM_MARKER_FINAL"] = pd.DataFrame([
            {"id": e.getVariableInt("id"), "grid_lin_id": e.getVariableInt("grid_lin_id"),
             **{axis: e.getVariableFloat(axis) for axis in ("x", "y", "z")},
             "marker_count": e.getVariableInt("marker_count"),
             "marker_exposure": e.getVariableFloat("marker_exposure")}
            for e in host.agent("ECM").getPopulationData()])
