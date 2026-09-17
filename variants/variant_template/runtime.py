"""OPTIONAL helper module: nothing here is discovered or registered automatically.

Import the needed names in __init__.py's register_runtime(). You may move these
definitions into __init__.py or rename this module and update the import.
"""


def initialize_cell(instance, rng):
    """Optional initial CELL fields. Core position/type/basic state already exist."""
    # instance.setVariableFloat("my_value", rng.uniform(0.0, 1.0))
    pass


def initialize_custom(instance, rng):
    """Optional initial CUSTOM fields, after the framework assigns its Int id.

    Runs once per managed initial agent; count=0 means there are none. It is not
    called for GPU births. Never modify the allocated id here.
    """
    # instance.setVariableFloat("value", 1.0)
    pass


class Runtime:
    """Example host callbacks, inactive until registered in __init__.py.

    Keep per-run mutable data in ctx.runtime_results(host), not module globals
    or this shared object. This matters when one model serves multiple runs.
    Replace the TEMPLATE_* output keys with unique names for your model.
    """
    def __init__(self, ctx):
        self.ctx = ctx

    def initialize(self, host):
        """Runs once per simulation, after core populations/macros are initialized."""
        self.ctx.runtime_results(host)["TEMPLATE_STEPS"] = []

    def step(self, host):
        """Runs after GPU layers every step; gate expensive work by sampling interval."""
        step = host.getStepCounter() + 1
        interval = self.ctx.config["SAVE_EVERY_N_STEPS"]
        if step % interval == 0:
            self.ctx.runtime_results(host)["TEMPLATE_STEPS"].append({"step": step})
        # A stateful agent must be queried in its state:
        # population = host.agent("CUSTOM", "active").getPopulationData()
        # For large populations prefer reductions instead of copying all agents.

    def finish(self, host):
        """Runs at normal simulation completion; suitable for final snapshots."""
        self.ctx.runtime_results(host)["TEMPLATE_FINISHED"] = True
        # The core merges these keys into its results pickle; collisions fail.
