# Tutorial: a complete variant with CELL-created markers

This walkthrough uses the runnable `variants/cell_markers/` package. It complements the first [model-variants tutorial](Tutorial-Model-Variants.md): the signal example isolates two functions; this example retains the **entire generic CellFoundry schedule** and adds a small complete interaction. Start a new model by copying [`variant_template`](../../../variants/variant_template/__init__.py), or copy this worked example if its structure is closer to your problem.

The example is a programming exercise. Each CELL creates one stationary MARKER at its current position. A marker reports its owner and position for two steps, then dies. CELL counts its own reports; nearby ECM nodes accumulate exposure. Markers exert no forces and do not represent soluble chemical mass. Exposure is an additional observation variable, independent of `C_sp` and diffusion.

It demonstrates new parameters, validation, extensions to two existing agents, a new agent type/state, an initially empty population, GPU births, managed IDs, a new message with two consumers, death, host callbacks, VTK additions and pickle output. It also shows where the generic mechanics/diffusion schedule runs. Optional core features needing external networks are present as conditional schedule branches, but are disabled in the default example. They are not prerequisites for understanding the variant.

## 1. Files and responsibilities

All executable code is included; there are no omitted functions to invent before running:

| File | Responsibility |
| --- | --- |
| [`__init__.py`](../../../variants/cell_markers/__init__.py) | Parameters, validation, native declarations, function bindings, callback registration and the full explicit schedule. |
| [`runtime.py`](../../../variants/cell_markers/runtime.py) | Small Python initializer and per-run analysis callbacks, imported explicitly by `register_runtime()`. |
| [`cell_emit_marker.cpp`](../../../variants/cell_markers/cell_emit_marker.cpp) | CELL creates one MARKER in state `active` and initializes every daughter field. |
| [`marker_publish.cpp`](../../../variants/cell_markers/marker_publish.cpp) | MARKER sends its ID, owner CELL ID and position. |
| [`cell_read_markers.cpp`](../../../variants/cell_markers/cell_read_markers.cpp) | CELL counts messages naming it as owner. |
| [`ecm_read_markers.cpp`](../../../variants/cell_markers/ecm_read_markers.cpp) | ECM counts markers within a distance cutoff and accumulates exposure. |
| [`marker_age.cpp`](../../../variants/cell_markers/marker_age.cpp) | MARKER ages and returns `DEAD` when its lifetime expires. |

`runtime.py` is a normal Python module. It is useful here to keep analysis separate from declarations and scheduling. The framework does not load it by name, nor does placing a function there cause it to run. This is the same organization used by radial glia, whose initialization and rosette analysis are considerably longer. All of these Python definitions could instead be placed in `__init__.py`.

Only `configure_layers(ctx)` is a mandatory variant hook. This example also uses the four optional hooks because it needs their responsibilities: `validate_config`, `declare_model`, `register_functions`, and `register_runtime`. Those hooks run once during model construction; the functions they register run later. Consult the [hook table and lifecycle](Tutorial-Model-Variants.md#variant-interface-and-construction-order) before adding your own helper names.

## 2. Parameters and supported scope

The variant introduces:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `MARKER_CAPACITY` | 16 | Maximum total marker IDs available during the run; dead IDs are not recycled. |
| `MARKER_LIFETIME_STEPS` | 2 | Number of steps in which a marker reports, including its birth step. |
| `MARKER_DETECTION_RADIUS` | 15 µm | Euclidean distance cutoff used by ECM consumers. |

`PARAM_DEFAULTS` introduces these names so JSON and optimizer overrides can change them. `declare_model()` separately creates the GPU properties needed by kernels. Capacity is consumed by `add_population()`, which creates the bound/counter/error properties; there is no need to create a duplicate capacity property on the GPU.

`PARAMS` selects four cells, four steps, a one-second time step, CELL interactions and saved outputs. Diffusion, fibre networks, focal adhesions, vascularization and lumen are off initially. Core grid dimensions, species/type counts and domain size are not overridden. No core Python or C++ file is edited to enable this example.

The example requires an initially fixed CELL population: CELL division and vascular CELL recruitment are rejected by `validate_config()`. CELL **creating MARKER** is a different mechanism from CELL **creating CELL**, so it works while `INCLUDE_CELL_CYCLE=False`. Each CELL creates at most one marker, making the required capacity predictable. Validation rejects a capacity smaller than `N_CELLS`, invalid lifetimes/radii, and disabled CELL agents. This is a useful early check; the generic GPU exhaustion check still protects actual ID allocations.

The full generic schedule includes conditional branches for optional features. Having those branches in the source is not proof that every biological parameter combination is calibrated. The executable tests cover the default case and an additional multiscale-diffusion case; enabling a file-backed feature also requires its normal core inputs and consistency checks.

## 3. Extend existing agents and create a new one

`declare_model(ctx)` adds:

| Agent | Variable | Type/default | Meaning |
| --- | --- | --- | --- |
| CELL | `marker_emitted` | Int / 0 | One after this CELL has created its marker. |
| CELL | `own_marker_count` | Int / 0 | Number of reports received from markers owned by this CELL this step. |
| ECM | `marker_count` | Int / 0 | Markers within the cutoff during the current observation. |
| ECM | `marker_exposure` | Float / 0 | Accumulated marker-seconds at this node. |
| MARKER | `id` | Int | Allocated custom ID in the MARKER reservation. |
| MARKER | `owner_cell_id` | Int / -1 | Typed reference to a CELL, not to another MARKER. |
| MARKER | `x`, `y`, `z` | Float | Fixed position copied from the parent at birth. |
| MARKER | `age_steps` | Int / 0 | Completed observation steps. |

CELL and ECM remain the existing core descriptions. The grid, its geometry and its initialization order stay under core control. Zero defaults initialize ECM's extra fields; **no extra ECM initialization callback is needed**. Extending an agent's variables does not require changing the way its population is created.

The new type is declared and reserved entirely in the variant:

```python
marker = ctx.model.newAgent("MARKER")
marker.newState("active")
marker.newVariableInt("id")
marker.newVariableInt("owner_cell_id", -1)
for coordinate in ("x", "y", "z"):
    marker.newVariableFloat(coordinate)
marker.newVariableInt("age_steps", 0)
ctx.agents["MARKER"] = marker
ctx.add_population("MARKER", count=0,
                   capacity=ctx.config["MARKER_CAPACITY"], state="active")
```

The initial population is empty. The reservation still exists before simulation construction, with bounds and a birth counter. `active` is an explicitly named native FLAMEGPU state; MARKER functions are bound to that state, and births target it. FLAMEGPU also has a default state, but this example puts no markers in it. All state-specific output queries therefore use `host.agent("MARKER", "active")`.

Changing MARKER's capacity does not change the core populations' initial IDs. See the [ID allocation explanation](Tutorial-Model-Variants.md#new-agent-populations-identifiers-and-bucket-messages) for `CURRENT_ID`, per-type counters, bucket keys, dense indices and exhaustion errors.

## 4. Create a message and bind each function

The variant declares a new `MessageBruteForce` named `marker_report`, containing Int `id`, Int `owner_cell_id`, and Float `x/y/z`. It does not extend an existing message, so no core publisher needs replacing.

Brute-force messaging lets every consumer inspect every marker message. For four markers this keeps the example transparent. The ECM kernel applies the actual distance cutoff; the CELL kernel filters by owner ID. This costs roughly `(number of ECM nodes + number of cells) × number of markers` per step. Larger models should use appropriate spatial and/or owner-keyed bucket messages, with the corresponding bounds and radii. Message strategy is a modeling/implementation decision, not something the hook system chooses automatically.

`register_functions(ctx)` creates these native bindings:

| Agent function | Input → output message | Additional binding |
| --- | --- | --- |
| CELL `cell_emit_marker` | None → None | `setAgentOutput("MARKER", "active")` |
| MARKER `marker_publish` | None → BruteForce | `setMessageOutput("marker_report")`; initial/end state `active` |
| CELL `cell_read_markers` | BruteForce → None | `setMessageInput("marker_report")` |
| ECM `ecm_read_markers` | BruteForce → None | `setMessageInput("marker_report")` |
| MARKER `marker_age` | None → None | Initial/end state `active`; `setAllowAgentDeath(True)` |

For example:

```python
publish = ctx.agents["MARKER"].newRTCFunctionFile(
    "marker_publish", str(directory / "marker_publish.cpp"))
publish.setInitialState("active")
publish.setEndState("active")
publish.setMessageOutput("marker_report")
```

The birth function uses `newRTCFunction()` with a source string prefixed by the absolute path to `variant_ids.cuh`; the remaining functions use `newRTCFunctionFile()`. This makes the shared header location explicit. Both approaches register native RTC functions; neither schedules execution by itself.

There is no `FILES` dictionary because this model adds functions without replacing core ones. If you replace a core function in your own model, use `FILES` before core registration and keep its expected function name/signature. If you add a field to an existing message, its publisher must also write that field. The template contains a commented `FILES` example; an unused replacement would add no value to this working model.

## 5. CELL gives birth to MARKER

The complete birth logic is short:

```cpp
FLAMEGPU_AGENT_FUNCTION(cell_emit_marker, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") || FLAMEGPU->getVariable<int>("marker_emitted")) {
        return flamegpu::ALIVE;
    }
    auto last = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_MARKER_LAST_ID");
    auto exhausted = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_MARKER_ID_EXHAUSTED");
    const int end = FLAMEGPU->environment.getProperty<int>("VARIANT_MARKER_ID_END");
    const int id = cellfoundry_claim_variant_id(last, end, exhausted);
    if (id < 0) return flamegpu::ALIVE;

    FLAMEGPU->agent_out.setVariable<int>("id", id);
    FLAMEGPU->agent_out.setVariable<int>("owner_cell_id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->agent_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->agent_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->agent_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->agent_out.setVariable<int>("age_steps", 0);
    FLAMEGPU->setVariable<int>("marker_emitted", 1);
    return flamegpu::ALIVE;
}
```

The header is included by Python registration. Each eligible CELL calls the atomic bounded allocator, writes every required daughter field, and only then records that it has emitted. Writing `agent_out` creates the daughter in the bound target state. The return value applies to the parent CELL, which survives. A failed ID claim writes no daughter and sets a flag causing the framework to reject the run after the GPU layers; it cannot silently produce a successful result with missing markers.

Python per-agent initializers **do not run for GPU births**. The C++ writes above are therefore essential even if a similarly named initializer exists in `runtime.py`. Here there are no initial MARKER agents at all, so there is no MARKER host initializer to register.

The owner reference stores the parent's custom CELL ID. A marker's own ID is from a different type's reservation. Code consuming `owner_cell_id` must look up a CELL; numeric IDs alone do not identify an agent across all types.

## 6. Write and read messages, then expire markers

`marker_publish.cpp` copies the marker's five report fields into `message_out`. It is scheduled after birth, so newborns can publish in their birth step. Both receivers run after publication.

The CELL receiver sets `own_marker_count` to the number of messages with matching `owner_cell_id`. It assigns zero when there are no matches. It does not assume marker and owner remain close; this is an owner relationship, not a distance relationship.

For ECM node position `p`, let `k` be the number of marker positions within `MARKER_DETECTION_RADIUS`, using ordinary Euclidean distance. The ECM receiver applies:

```text
marker_count = k
marker_exposure = marker_exposure + k × TIME_STEP
```

Exposure has units of marker-seconds and persists after the markers expire. Empty message lists reset `marker_count` to zero and leave exposure unchanged. Marker positions stay fixed at their birth locations even if the parent CELL subsequently moves. Markers are born at the parent's post-movement position; there is no additional marker movement or boundary-crossing rule. The cutoff is not periodically wrapped.

Finally `marker_age.cpp` increments `age_steps` and returns `DEAD` once it reaches `MARKER_LIFETIME_STEPS`. `setAllowAgentDeath(True)` makes that return effective. Aging occurs **after** the observations, so lifetime 2 means two reports: in the birth step and in the next step.

## 7. Own the entire schedule

Open `configure_layers()` in [`cell_markers/__init__.py`](../../../variants/cell_markers/__init__.py). It contains the complete generic schedule directly, followed by the four marker layers. It does not import the organoid schedule, call a core schedule builder or apply hidden insertion rules.

| Schedule portion | Purpose and dependency |
| --- | --- |
| L0 | Conditional vascular updates and multiscale boundary preparation before publication. |
| L1 | Required generic agent/message broadcasts for enabled features. |
| L2 | Conditional boundary and fibre-remodeling interactions. |
| L3 | Conditional metabolism and core cell-cycle sequence; CELL division is disabled/rejected in this example. |
| L4–L6 | Conditional concentration/diffusivity updates, ECM interaction, multiscale substeps/commit and diffusion boundaries. |
| L7 | Conditional mechanics and CELL stress finalization. |
| L8 | CELL movement and conditional fibre/adhesion/boundary/ECM/vascular movement. |
| M1 | CELL emits MARKER at its updated position. |
| M2 | MARKER publishes, including newborns. |
| M3 | CELL and ECM consume the same report list. |
| M4 | MARKER ages and may die, after both receivers have finished. |

Only enabled branches create layers. With the default parameters, the exact active GPU sequence is:

```text
L1_Agent_Locations:         BCORNER broadcast + CELL broadcast
L7_CELL_CELL_Interaction:   CELL interactions
L7_CELL_Stress_State_Update:CELL stress finalization
L8_CELL_Movement:          CELL movement
M1_CELL_Emit:              CELL -> MARKER birth
M2_MARKER_Publish:         MARKER -> marker_report
M3_Read_Markers:           CELL reads reports + ECM reads reports
M4_MARKER_Age:             MARKER age/death
```

The custom portion is explicit code:

```python
model.newLayer("M1_CELL_Emit").addAgentFunction("CELL", "cell_emit_marker")
model.newLayer("M2_MARKER_Publish").addAgentFunction("MARKER", "marker_publish")
model.newLayer("M3_Read_Markers").addAgentFunction("CELL", "cell_read_markers")
model.Layer("M3_Read_Markers").addAgentFunction("ECM", "ecm_read_markers")
model.newLayer("M4_MARKER_Age").addAgentFunction("MARKER", "marker_age")
```

The two M3 functions use different agent populations and only read the same message list, so they can share a layer. Birth and publication need different layers, as do observation and death. Putting aging before publication would change the meaning of lifetime; putting birth after publication would delay the first report by a step.

When multiscale diffusion is enabled, this full schedule explicitly invokes `ctx.add_multiscale_diffusion_layers()` after concentration/diffusivity preparation. The helper owns internal solver substeps, while the variant still owns the surrounding order. Marker observation remains on the parent time step; it does not execute once per diffusion substep.

Core initialization and end-of-step output callbacks are separate from this GPU layer list. The capacity guard runs after GPU layers before normal core end-of-step output. Variant analysis callbacks registered in `register_runtime()` also run on the host after the GPU work.

## 8. Register initialization and analysis explicitly

The registration hook contains:

```python
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
```

The CELL callback illustrates initial-agent initialization; its zero assignments match declaration defaults and could be omitted for this deterministic example. It neither creates cells nor allocates their IDs. The ECM additions use defaults directly. These are separate choices for separate agents.

`Metrics.initialize()` creates result buffers for each run. `Metrics.step()` records the post-GPU populations and a small summary. `Metrics.finish()` captures the final ECM fields before the core writes the merged pickle. Buffers live in `ctx.runtime_results(host)`, not module globals, so simulations do not share result rows accidentally. The helper object stores only its context.

The variant adds these output keys:

| Key | Contents |
| --- | --- |
| `MARKER_HISTORY` | Per-step marker positions/ages/owner IDs and CELL emission/report fields. |
| `MARKER_SUMMARY` | Step, live markers, cumulative emissions, owner reports and total ECM exposure. |
| `ECM_MARKER_FINAL` | Final ECM positions, IDs/grid indices, current marker counts and accumulated exposure. |

CELL VTK output includes the two added CELL scalars. The CELL VTK registry does not automatically export MARKER or extra ECM fields: these are intentionally exported through the pickle in this example. Add a dedicated exporter in `runtime.py` when your model needs geometry files for its new type. Expensive population copies are suitable here because the teaching model is tiny; large models should use reductions and less frequent sampling.

## 9. Run and validate

Run from the project root in the configured FLAMEGPU environment:

```sh
python model.py --variant cell_markers --result-dir results/cell_markers
```

With four initially alive CELL agents and lifetime 2:

| Completed step | Markers alive after aging | Cumulative emissions | Owner reports observed this step |
| --- | --- | --- | --- |
| 1 | 4 | 4 | 4 |
| 2 | 0 | 4 | 4 |
| 3 | 0 | 4 | 0 |
| 4 | 0 | 4 | 0 |

Step 2 has four reports even though there are no surviving markers at its end: observation occurs before death. ECM exposure increases in steps 1 and 2, then remains constant. For a fixed ECM node that has `k` markers within range, its final exposure is `2 × k` marker-seconds at the default one-second step. Each marker has one valid owner, IDs are unique within MARKER, and each CELL emits exactly once.

To combine the example with multiscale diffusion, create an overrides file containing:

```json
{
  "INCLUDE_DIFFUSION": true,
  "TIME_STEP_DIFFUSION": [0.25, 0.5, 1.0],
  "N_CELLS": 6,
  "MARKER_LIFETIME_STEPS": 3,
  "MARKER_DETECTION_RADIUS": 12.0
}
```

Then run the same variant with `--overrides <path>`. These diffusion values assume the supplied three-species configuration; changes to structural dimensions require the core consistency procedure. The marker lifetime is still measured in parent steps. Capacity 16 covers the six one-time births.

The repeatable validation command is:

```sh
python tools/validate_cell_markers.py
python -m unittest manual_tests.test_cell_markers
```

GPU validation uses an isolated source copy, runs the default and multiscale cases, checks births/owners/death/empty messages, independently calculates every ECM node's expected exposure from marker positions, and checks VTK/pickle output. CPU checks compare the generic portion of the explicit schedule with the core schedule and test rejected configurations. These tests establish software wiring and arithmetic; they are not a biological validation.

## 10. Adapt the example without hidden steps

- For a different birth rule, change eligibility in `cell_emit_marker.cpp`, choose a lifetime-wide capacity and update its validation. Initialize all new daughter variables in the same birth kernel.
- For moving markers, add a movement kernel and put it explicitly before or after publication according to which positions consumers should see; define boundary behavior too.
- For more efficient communication, declare suitable spatial/bucket messages, bind matching kernel signatures, update publisher fields and keep publication before consumption. An owner-keyed bucket uses CELL IDs, not MARKER IDs.
- For additional CELL births, define how the new CELL variables are inherited/initialized, update the replacement birth kernel(s), permit the feature in validation and revisit total marker capacity. Removing the validation error alone is insufficient.
- For specialized analysis/export, add ordinary Python code to `runtime.py` and register the relevant callback. Merely creating the file or naming a function `step()` has no execution effect.

The full schedule stays in the variant. Each added mechanism requires its declarations, function bindings, initialization/birth rules, explicit execution order and a small expected-result test.
