"""Numerical verification of CellFoundry's production diffusion kernels.

Run: python tools/validate_multiscale_diffusion.py --gpu --output report.json
Without --gpu this runs the scheduler and independent NumPy reference only.
The GPU harness uses the same definitions in model.py and RTC sources as model.py,
specialized in memory: it never changes N, N_SPECIES, or checked-in sources.
"""

import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helper_module import plan_diffusion, validate_diffusion_parameters
from helper_module import reject_trial


def load_production_wiring(model, ns):
    """Load model.py definitions without importing its executable simulation.

    AST extraction selects unmodified production functions and host classes;
    the test does not maintain a second diffusion scheduler.
    """
    import pyflamegpu
    names = {"CheckDiffusionStability", "DiffusionSubmodelDebug", "ExitAfterDiffusionSubsteps",
             "_new_diffusion_message", "_register_multiscale_diffusion",
             "_add_multiscale_diffusion_layers"}
    tree = ast.parse((ROOT / "model.py").read_text(encoding="utf-8"))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(definitions) == len(names)
    namespace = dict(pyflamegpu=pyflamegpu, math=math, model=model, reject_trial=reject_trial)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ROOT / "model.py"), "exec"), namespace)
    # Tests use small independent grids/species counts. Only this test adapter
    # substitutes literal dimensions in memory; production uses ordinary files
    # synchronized by check_hard_coded_values.py.
    def test_rtc_file(agent, name, filename):
        code = specialize_test_source(filename, ns["N_SPECIES"], ns["ECM_POPULATION_SIZE"])
        return agent.newRTCFunction(name, code)

    with patch.object(pyflamegpu.AgentDescription, "newRTCFunctionFile", test_rtc_file):
        engine = namespace["_register_multiscale_diffusion"](model, ns)
    namespace["_DIFFUSION_ENGINE"] = engine
    return namespace["_add_multiscale_diffusion_layers"], engine


def specialize_test_source(filename, species_count, population_size):
    """Adapt checked literal dimensions for isolated tests, without file edits."""
    code = (ROOT / filename).read_text(encoding="utf-8")
    for constant, value in (("N_SPECIES", species_count), ("ECM_POPULATION_SIZE", population_size)):
        code = re.sub(rf"(\b{constant}\s*=\s*)\d+", rf"\g<1>{value}", code)
    if species_count == 1:
        # rc.5 classifies one-element arrays as scalar RTC variables.
        code = re.sub(r'getVariable<float, N_SPECIES>\(("[^"]+"), i\)',
                      r'getVariable<float>(\1)', code)
        code = re.sub(r'setVariable<float, N_SPECIES>\(("[^"]+"), i, ',
                      r'setVariable<float>(\1, ', code)
        code = re.sub(r'getProperty<(float|unsigned int)>\(("(?:DIFFUSION_ACTIVE_SPECIES|TIME_STEP_DIFFUSION|DIFFUSION_COEFF_MULTI|ECM_DEGRADATION_RATE_MULTI)"), i\)',
                      r'getProperty<\1>(\2)', code)
    return code


def write_diffusion_vti(path, field, spacing, coefficients=None, reference=None):
    """Export a Cartesian concentration field with x varying fastest for VTK."""
    shape = field.shape[1:]
    extent = " ".join(str(value) for n in shape for value in (0, n - 1))
    vtk = ET.Element("VTKFile", type="ImageData", version="0.1", byte_order="LittleEndian")
    grid = ET.SubElement(vtk, "ImageData", WholeExtent=extent, Origin="0 0 0",
                         Spacing=" ".join(str(h) for h in spacing))
    piece = ET.SubElement(grid, "Piece", Extent=extent)
    point_data = ET.SubElement(piece, "PointData", Scalars="C_sp_0")
    for species in range(len(field)):
        arrays = [(f"C_sp_{species}", field[species])]
        if coefficients is not None:
            arrays.append((f"D_sp_{species}", coefficients[species]))
        if reference is not None:
            arrays.extend([(f"reference_C_sp_{species}", reference[species]),
                           (f"error_C_sp_{species}", field[species] - reference[species])])
        for name, values in arrays:
            item = ET.SubElement(point_data, "DataArray", type="Float64", Name=name, format="ascii")
            item.text = " ".join(f"{value:.17g}" for value in np.asarray(values).ravel(order="F"))
    ET.SubElement(piece, "CellData")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)


def write_diffusion_series(directory, steps, interval):
    """Create a ParaView collection with physical time in seconds."""
    vtk = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(vtk, "Collection")
    for step in range(steps + 1):
        ET.SubElement(collection, "DataSet", timestep=f"{step * interval:.17g}", group="", part="0",
                      file=f"ecm_data_t{step:05d}.vti")
    ET.ElementTree(vtk).write(directory / "diffusion.pvd", encoding="utf-8", xml_declaration=True)


def reference_step(field, coefficients, spacing, dt, decay):
    """Independent vectorized face-flux reference, all exterior faces no-flux."""
    change = np.zeros_like(field)
    for axis, h in enumerate(spacing, start=1):
        left = [slice(None)] * 4
        right = left.copy()
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        dl, dr = coefficients[left], coefficients[right]
        face = np.divide(2 * dl * dr, dl + dr, out=np.zeros_like(dl), where=dl + dr > 0)
        transfer = dt * face * (field[right] - field[left]) / h**2
        change[left] += transfer
        change[right] -= transfer
    return (field + change) * np.exp(-np.asarray(decay)[:, None, None, None] * dt)


def run_gpu(initial, coefficients, spacing, interval, requested, steps, *,
            decay=None, fixed=None, initial_boundary=None, floors=None,
            heterogeneous=False, actual_spacing=None, exchange=None, legacy=False,
            minimum_spacing=None, save_data_to_file=False, output_dir=None,
            fnode_positions=None, fnode_search_radius=1.0, fnode_average_density=1):
    """Run real FLAMEGPU messages, submodels, boundary preparation and commit."""
    import pyflamegpu as fg

    initial = np.asarray(initial, dtype=float)
    size = initial.shape[0]
    dims = list(initial.shape[1:])
    count = math.prod(dims)
    supplied_d = np.asarray(coefficients, dtype=float)
    if supplied_d.ndim == 1:
        base_d = list(supplied_d)
        local_d = np.broadcast_to(supplied_d[:, None, None, None], initial.shape)
    else:
        base_d = list(np.max(supplied_d, axis=(1, 2, 3)))
        local_d = supplied_d
    decay = list(decay if decay is not None else [0.0] * size)
    fixed = fixed if fixed is not None else [[-1.0] * 6 for _ in range(size)]
    initial_boundary = initial_boundary if initial_boundary is not None else [[-1.0] * 6 for _ in range(size)]
    xyz_spacing = actual_spacing if actual_spacing is not None else spacing
    boundaries = [value for n, h in zip(dims, xyz_spacing) for value in ((n - 1) * h, 0.0)]
    ns = dict(N_SPECIES=size, ECM_POPULATION_SIZE=count, ECM_AGENTS_PER_DIR=dims,
              TIME_STEP=interval, TIME_STEP_DIFFUSION=requested, DIFFUSION_CFL_SAFETY=0.9,
              DIFFUSION_MAX_SUBSTEPS=1000000, ECM_DEGRADATION_RATE_MULTI=decay,
              HETEROGENEOUS_DIFFUSION=heterogeneous, DIFFUSION_COEFF_MULTI=base_d,
              MOVING_BOUNDARIES=actual_spacing is not None,
              DIFFUSION_MIN_SPACING=minimum_spacing if minimum_spacing is not None else xyz_spacing,
              INIT_ECM_CONCENTRATION_VALS=[0.0] * size, INIT_ECM_SAT_CONCENTRATION_VALS=[1.0] * size,
              BOUNDARY_CONC_INIT_MULTI=initial_boundary, BOUNDARY_CONC_FIXED_MULTI=fixed,
              SAVE_DATA_TO_FILE=save_data_to_file, SAVE_EVERY_N_STEPS=1)
    ns.update({f"L0_{axis}": (n - 1) * h for axis, n, h in zip("xyz", dims, spacing)})
    groups = validate_diffusion_parameters(ns)
    ns["DIFFUSION_GROUPS"] = groups
    output_dir = Path(output_dir) if output_dir is not None else None
    if save_data_to_file and output_dir is None:
        raise ValueError("SAVE_DATA_TO_FILE requires an output directory")
    model = fg.ModelDescription("diffusion_validation")
    env = model.Environment()
    env.newPropertyArrayFloat("ECM_DEGRADATION_RATE_MULTI", decay)
    env.newPropertyUInt("HETEROGENEOUS_DIFFUSION", heterogeneous)
    env.newPropertyArrayFloat("DIFFUSION_COEFF_MULTI", base_d)
    env.newPropertyArrayUInt("ECM_AGENTS_PER_DIR", dims)
    env.newPropertyArrayFloat("COORDS_BOUNDARIES", boundaries)
    env.newPropertyFloat("ECM_BOUNDARY_INTERACTION_RADIUS", min(xyz_spacing) * 0.01)
    env.newMacroPropertyFloat("C_SP_MACRO", size, count)
    for prop in ("BOUNDARY_CONC_INIT_MULTI", "BOUNDARY_CONC_FIXED_MULTI"):
        env.newMacroPropertyFloat(prop, size, 6)
    agent = model.newAgent("ECM")
    agent.newVariableInt("grid_lin_id")
    for name in ("x", "y", "z"):
        agent.newVariableFloat(name)
    for name in ("grid_i", "grid_j", "grid_k"):
        agent.newVariableUInt8(name)
    agent.newVariableArrayFloat("C_sp", size)
    agent.newVariableArrayFloat("D_sp", size)
    agent.newVariableArrayFloat("diffusion_boundary", size, [-1.0] * size)
    agent.newVariableArrayFloat("diffusion_vascular_floor", size, [-1.0] * size)
    agent.newVariableUInt("diffusion_error", 0)
    add_layers, engine = load_production_wiring(model, ns)
    if legacy:
        # Exercise the retained, checked-in solver too. Only specialize array
        # extents in memory; this is the same operation as the size checker.
        for name in ("INCLUDE_DIFFUSION", "MULTISCALE_DIFFUSION", "UNSTABLE_DIFFUSION", "DEBUG_PRINTING", "DEBUG_DIFFUSION"):
            env.newPropertyUInt(name, 1 if name == "INCLUDE_DIFFUSION" else 0)
        env.newPropertyFloat("TIME_STEP", interval)
        env.newPropertyFloat("EPSILON", 1e-10)
        env.newPropertyFloat("ECM_ECM_EQUILIBRIUM_DISTANCE", spacing[0])
        agent.newVariableInt("id")
        agent.newVariableArrayFloat("C_sp_sat", size)
        for name in ("vx", "vy", "vz", "fx", "fy", "fz", "k_elast", "d_dumping"):
            agent.newVariableFloat(name)
        message = model.newMessageArray3D("legacy_ecm_message")
        message.setDimensions(*dims)
        for name in ("id", "grid_lin_id"):
            message.newVariableInt(name)
        for name in ("grid_i", "grid_j", "grid_k"):
            message.newVariableUInt8(name)
        for name in ("x", "y", "z", "vx", "vy", "vz", "k_elast", "d_dumping"):
            message.newVariableFloat(name)
        for name in ("D_sp", "C_sp", "C_sp_sat"):
            message.newVariableArrayFloat(name, size)
        for function in ("ecm_grid_location_data", "ecm_ecm_interaction"):
            code = (ROOT / f"{function}.cpp").read_text(encoding="utf-8")
            for constant, value in (("N_SPECIES", size), ("ECM_POPULATION_SIZE", count)):
                code = re.sub(rf"(\b{constant}\s*=\s*)\d+", rf"\g<1>{value}", code)
            fn = agent.newRTCFunction(function, code)
            if function == "ecm_grid_location_data":
                fn.setMessageOutput("legacy_ecm_message")
            else:
                fn.setMessageInput("legacy_ecm_message")

    if fnode_positions is not None:
        assert heterogeneous and not legacy
        agent.newVariableInt("id")
        env.newPropertyFloat("ECM_ECM_EQUILIBRIUM_DISTANCE", fnode_search_radius)
        env.newPropertyUInt("AVG_NETWORK_VOXEL_DENSITY", fnode_average_density)
        fnodes = model.newAgent("FNODE")
        for axis in "xyz":
            fnodes.newVariableFloat(axis)
        message = model.newMessageSpatial3D("fnode_test_message")
        message.setRadius(fnode_search_radius)
        message.setMin(*[-2 * fnode_search_radius] * 3)
        message.setMax(*[(n - 1) * h + 2 * fnode_search_radius for n, h in zip(dims, xyz_spacing)])
        fnodes.newRTCFunction("fnode_test_publish", """FLAMEGPU_AGENT_FUNCTION(fnode_test_publish, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
            FLAMEGPU->message_out.setLocation(FLAMEGPU->getVariable<float>("x"), FLAMEGPU->getVariable<float>("y"), FLAMEGPU->getVariable<float>("z"));
            return flamegpu::ALIVE;
        }""").setMessageOutput("fnode_test_message")
        agent.newRTCFunction("ecm_Dsp_update", specialize_test_source("ecm_Dsp_update.cpp", size, count)).setMessageInput("fnode_test_message")

    class Initialize(fg.HostFunction):
        def run(self, api):
            for prop, vals in (("BOUNDARY_CONC_INIT_MULTI", initial_boundary), ("BOUNDARY_CONC_FIXED_MULTI", fixed)):
                macro = api.environment.getMacroPropertyFloat(prop)
                for s in range(size):
                    for face in range(6):
                        macro[s][face] = float(vals[s][face])

    class ParentExchange(fg.HostFunction):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def run(self, api):
            self.calls += 1
            if exchange is not None:
                population = api.agent("ECM").getPopulationData()
                for item in population:
                    index = np.unravel_index(item.getVariableInt("grid_lin_id"), dims)
                    values = np.asarray(item.getVariableArrayFloat("C_sp"))
                    item.setVariableArrayFloat("C_sp", list(values + np.asarray(exchange)[(slice(None),) + index]))

    class Finish(fg.HostFunction):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.macro_error = 0.0
            self.counts = []

        def run(self, api):
            self.calls += 1
            # Compare the committed macro buffer with bound agent variables.
            macro = api.environment.getMacroPropertyFloat("C_SP_MACRO")
            for item in api.agent("ECM").getPopulationData():
                index = item.getVariableInt("grid_lin_id")
                for s, value in enumerate(item.getVariableArrayFloat("C_sp")):
                    self.macro_error = max(self.macro_error, abs(float(macro[s][index]) - value))
            self.counts = [group.substeps for group in groups]
            if save_data_to_file:
                field = np.empty_like(initial)
                actual_d = np.empty_like(local_d)
                for item in api.agent("ECM").getPopulationData():
                    index = np.unravel_index(item.getVariableInt("grid_lin_id"), dims)
                    field[(slice(None),) + index] = item.getVariableArrayFloat("C_sp")
                    actual_d[(slice(None),) + index] = item.getVariableArrayFloat("D_sp")
                write_diffusion_vti(output_dir / f"ecm_data_t{self.calls:05d}.vti", field, xyz_spacing, actual_d)
            if self.calls == 1:
                init = api.environment.getMacroPropertyFloat("BOUNDARY_CONC_INIT_MULTI")
                for s in range(size):
                    for face in range(6):
                        init[s][face] = -1.0

    init, parent_exchange, finish = Initialize(), ParentExchange(), Finish()
    model.addInitFunction(init)
    model.newLayer("parent_exchange").addHostFunction(parent_exchange)
    if fnode_positions is not None:
        model.newLayer("FNODE_Locations").addAgentFunction("FNODE", "fnode_test_publish")
        model.newLayer("ECM_Dsp_Update").addAgentFunction("ECM", "ecm_Dsp_update")
    if legacy:
        model.newLayer("legacy_publish").addAgentFunction("ECM", "ecm_grid_location_data")
        model.newLayer("legacy_transport").addAgentFunction("ECM", "ecm_ecm_interaction")
    else:
        add_layers()
    model.addStepFunction(finish)
    population = fg.AgentVector(agent, count)
    for linear, index in enumerate(np.ndindex(*dims)):
        item = population[linear]
        item.setVariableInt("grid_lin_id", linear)
        if legacy or fnode_positions is not None:
            item.setVariableInt("id", linear + 1)
        if legacy:
            item.setVariableFloat("k_elast", 1.0)
        for name, coordinate, h in zip("xyz", index, xyz_spacing):
            item.setVariableFloat(name, coordinate * h)
        for name, coordinate in zip(("grid_i", "grid_j", "grid_k"), index):
            item.setVariableUInt8(name, coordinate)
        item.setVariableArrayFloat("C_sp", list(initial[(slice(None),) + index]))
        item.setVariableArrayFloat("D_sp", list(local_d[(slice(None),) + index]))
        if floors is not None:
            item.setVariableArrayFloat("diffusion_vascular_floor", list(floors[(slice(None),) + index]))
    simulation = fg.CUDASimulation(model)
    simulation.SimulationConfig().steps = steps
    simulation.SimulationConfig().random_seed = 42
    simulation.setPopulationData(population)
    if fnode_positions is not None:
        fnode_population = fg.AgentVector(fnodes, len(fnode_positions))
        for item, position in zip(fnode_population, fnode_positions):
            for axis, value in zip("xyz", position):
                item.setVariableFloat(axis, float(value))
        simulation.setPopulationData(fnode_population)
    if save_data_to_file:
        write_diffusion_vti(output_dir / "ecm_data_t00000.vti", initial, xyz_spacing, local_d)
    simulation.simulate()
    if save_data_to_file:
        write_diffusion_series(output_dir, steps, interval)
        parameters = {key: value for key, value in ns.items() if key != "DIFFUSION_GROUPS"}
        parameters.update(actual_spacing=list(xyz_spacing),
                          groups=[dict(species=group.species, substeps=group.substeps, dt=group.dt)
                                  for group in groups], nested=engine["nested"])
        (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    simulation.getPopulationData(population)
    result = np.empty_like(initial)
    for item in population:
        index = np.unravel_index(item.getVariableInt("grid_lin_id"), dims)
        result[(slice(None),) + index] = item.getVariableArrayFloat("C_sp")
    fnode_d_error = 0.0
    if fnode_positions is not None:
        for item in population:
            position = np.array([item.getVariableFloat(axis) for axis in "xyz"])
            neighbors = np.count_nonzero(np.linalg.norm(np.asarray(fnode_positions) - position, axis=1) < fnode_search_radius)
            density = neighbors / max(fnode_average_density, 1)
            multiplier = max(0.05, min(1.0, 1.0 / (1.0 + 4.0 * max(density - 1.0, 0.0))))
            expected = np.asarray(base_d) * multiplier
            fnode_d_error = max(fnode_d_error, float(np.max(abs(np.asarray(item.getVariableArrayFloat("D_sp")) - expected))))
        assert fnode_d_error < 2e-7
    assert parent_exchange.calls == steps and finish.calls == steps
    assert finish.macro_error == 0.0
    return result, dict(parent_steps=finish.calls, substeps=finish.counts,
                        macro_error=finish.macro_error,
                        groups=[list(group.species) for group in groups], nested=engine["nested"],
                        fnode_diffusivity_error=fnode_d_error if fnode_positions is not None else None)




def validate_fnode_diffusion(save_data_to_file=False, output_dir=None):
    """Test production FNODE crowding -> D_sp -> multiscale transport.

    A controlled cluster of FNODE messages fits inside a 4 x 4 x 4 um cube.
    Fibre connectivity/mechanics are not exercised by this density-only test.
    """
    shape = (5, 5, 5)
    positions = np.array([[2 + offset, 2.07, 2.03] for offset in (-0.21, -0.11, 0.01, 0.09, 0.19)])
    base_d = np.array([1.0, 0.1, 0.0])
    initial = 0.2 + np.random.default_rng(12).random((3,) + shape)
    local_d = np.zeros_like(initial)
    for index in np.ndindex(shape):
        count = np.count_nonzero(np.linalg.norm(positions - np.array(index), axis=1) < 1.0)
        multiplier = max(0.05, 1.0 / (1.0 + 4.0 * max(count - 1, 0)))
        local_d[(slice(None),) + index] = base_d * multiplier
    final, metadata = run_gpu(initial, base_d, [1, 1, 1], 0.2, [0.01] * 3, 3,
                              heterogeneous=True, fnode_positions=positions,
                              save_data_to_file=save_data_to_file, output_dir=output_dir)
    reference = initial.copy()
    for _ in range(60):
        reference = reference_step(reference, local_d, [1, 1, 1], 0.01, [0, 0, 0])
    error = float(np.max(abs(final - reference)))
    assert error < 3e-6
    if save_data_to_file:
        write_diffusion_vti(Path(output_dir) / "comparison.vti", final, [1, 1, 1], reference=reference)
        (Path(output_dir) / "fnode_positions.json").write_text(json.dumps(positions.tolist(), indent=2) + "\n", encoding="utf-8")
    return dict(linf=error, **metadata)


def validate_fixed_clock_guards():
    """Exercise direct/submodel failures and the absence of adaptive selection."""
    start = np.ones((1, 5, 5, 5))
    _, metadata = run_gpu(start, [0.1], [1, 1, 1], 0.1, [0.1], 2,
                          heterogeneous=True, actual_spacing=[1, 1, 1])
    assert metadata["nested"] == [False] and metadata["substeps"] == [1]
    rejected = []
    cases = [
        ("unsafe_parent_timestep", [0.1], 0.5, [0.5], dict(actual_spacing=[0.1] * 3), "fixed CFL bound"),
        ("unsafe_submodel_timestep", [1.0], 0.5, [0.13], dict(actual_spacing=[0.25] * 3), "fixed CFL bound"),
        ("coincident_neighbors", [1.0], 0.1, [0.1], dict(actual_spacing=[0, 1, 1]), "neighbor distance"),
    ]
    invalid_d = np.ones_like(start)
    invalid_d[0, 2, 2, 2] = -1.0
    cases.append(("negative_local_diffusivity", invalid_d, 0.1, [0.1], dict(heterogeneous=True), "diffusion coefficient"))
    for name, coefficients, interval, requested, options, expected in cases:
        try:
            run_gpu(start, coefficients, [1, 1, 1], interval, requested, 1,
                    minimum_spacing=[1, 1, 1], **options)
        except Exception as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: unexpected error: {error}") from error
            rejected.append(name)
        else:
            raise AssertionError(f"{name}: unsafe diffusion was not rejected")
    return dict(single_step_with_moving_and_heterogeneous_flags=metadata, rejected_cases=rejected)


def validate(gpu=False, save_data_to_file=False, output_dir=None):
    results = {"backend": "FLAMEGPU" if gpu else "NumPy reference", "checks": {}}
    checks = results["checks"]
    case_directories = []

    def run_case(name, *args, **kwargs):
        directory = Path(output_dir) / name if output_dir is not None else None
        result = run_gpu(*args, save_data_to_file=save_data_to_file, output_dir=directory, **kwargs)
        if save_data_to_file:
            case_directories.append(name)
        return result

    def comparison(name, field, expected, spacing):
        if save_data_to_file:
            write_diffusion_vti(Path(output_dir) / name / "comparison.vti", field, spacing, reference=expected)
    shape = (9, 9, 9)
    spacing = [1.0, 1.0, 1.0]
    indices = np.indices(shape)
    mode = np.prod(np.cos(np.pi * (indices + 0.5) / np.asarray(shape)[:, None, None, None]), axis=0)
    initial = np.array([1 + 0.2 * mode, 1 + 0.3 * mode, 1 + 0.1 * mode])
    ds = [1.0, 0.1, 0.0]
    decay = [0.05, 0.1, 4.0]
    interval, requested, steps = 0.6, [0.04, 0.3, 0.6], 3
    groups = plan_diffusion(interval, requested, ds, spacing)
    expected = np.empty_like(initial)
    discrete_eigenvalue = sum(4 * math.sin(math.pi / (2 * n))**2 / h**2 for n, h in zip(shape, spacing))
    amplitudes = [0.2, 0.3, 0.1]
    for group in groups:
        for s in group.species:
            amplitude = (1 - ds[s] * group.dt * discrete_eigenvalue)**(group.substeps * steps)
            expected[s] = (1 + amplitudes[s] * amplitude * mode) * math.exp(-decay[s] * interval * steps)
    if gpu:
        output, metadata = run_case("three_species_cosine", initial, ds, spacing, interval, requested, steps, decay=decay)
    else:
        output = initial.copy()
        for group in groups:
            selected = list(group.species)
            d = np.broadcast_to(np.array(ds)[selected, None, None, None], output[selected].shape)
            for _ in range(group.substeps * steps):
                output[selected] = reference_step(output[selected], d, spacing, group.dt, np.array(decay)[selected])
        metadata = {"substeps": [g.substeps for g in groups]}
    error = float(np.max(np.abs(output - expected)))
    assert error < 3e-6, error
    checks["three_species_cosine_and_exact_decay"] = {"linf": error, **metadata}
    comparison("three_species_cosine", output, expected, spacing)
    # Continuum comparison includes the known spatial truncation of the stencil.
    continuum_lambda = sum((math.pi / (n * h))**2 for n, h in zip(shape, spacing))
    analytic = np.array([(1 + a * math.exp(-d * continuum_lambda * interval * steps) * mode)
                         * math.exp(-loss * interval * steps) for a, d, loss in zip(amplitudes, ds, decay)])
    checks["continuum_cosine_error"] = {"linf": float(np.max(abs(output - analytic))),
                                           "rms": float(np.sqrt(np.mean((output - analytic)**2)))}
    assert checks["continuum_cosine_error"]["linf"] < 3e-4
    rng = np.random.default_rng(5)
    field = 0.2 + rng.random((2, 6, 5, 4))
    local_d = np.ones_like(field)
    local_d[:, 3:] = 0.03
    local_d[1, 2] = 0  # impermeable internal sheet
    h = [0.7, 1.1, 1.6]
    if gpu:
        final, meta = run_case("heterogeneous_no_flux", field, local_d, h, 0.2, [0.01, 0.01], 4, heterogeneous=True)
    else:
        final = field.copy()
        for _ in range(80):
            final = reference_step(final, local_d, h, 0.01, [0, 0])
        meta = {}
    reference = field.copy()
    for _ in range(80):
        reference = reference_step(reference, local_d, h, 0.01, [0, 0])
    relative_mass = float(np.max(abs(final.sum(axis=(1, 2, 3)) / field.sum(axis=(1, 2, 3)) - 1)))
    reference_error = float(np.max(abs(final - reference)))
    assert relative_mass < 1e-6 and reference_error < 3e-6 and final.min() >= 0
    checks["heterogeneous_no_flux_and_anisotropic_spacing"] = {"relative_mass_error": relative_mass, "reference_linf": reference_error, **meta}
    comparison("heterogeneous_no_flux", final, reference, h)
    if gpu:
        # Fixed opposing x faces give a stationary linear profile; other faces
        # have zero flux. Every substep must see the pinned boundary values.
        linear = np.broadcast_to(np.linspace(0, 1, 7)[None, :, None, None], (1, 7, 5, 4)).copy()
        final, _ = run_case("dirichlet_linear", linear, [1.0], [1, 1, 1], 0.5, [0.02], 3,
                           fixed=[[1, 0, -1, -1, -1, -1]])
        error = float(np.max(abs(final - linear)))
        assert error < 1e-6
        checks["dirichlet_linear_steady_state"] = {"linf": error}
        # Integer-free ratio, actual grid contraction, and a source delta once
        # per parent step exercise time coverage and the coupling contract.
        start = np.ones((1, 5, 5, 5))
        exchange = np.full_like(start, 0.02)
        final, meta = run_case("fixed_clock_contracted_grid", start, [1.0], [1, 1, 1], 0.5, [0.13], 3,
                             actual_spacing=[0.25, 0.25, 0.25], exchange=exchange)
        assert np.max(abs(final - 1.06)) < 2e-6 and meta["substeps"][0] > 4
        checks["fixed_clock_contracted_grid_and_once_per_parent_exchange"] = {"linf": float(np.max(abs(final - 1.06))), **meta}
        # Fast and slow clocks must not change independent species' solutions.
        fine, _ = run_case("small_main_step", initial, ds, spacing, 0.04, [0.04, 0.04, 0.04], 45, decay=decay)
        slow_fine_error = float(np.max(abs(output[0] - fine[0])))
        assert slow_fine_error < 3e-6
        checks["fast_species_matches_small_main_step"] = {"linf": slow_fine_error}
        # A decaying vessel voxel is replenished every substep, and an initial
        # boundary value is released after exactly one parent interval.
        start = np.zeros((1, 5, 5, 5))
        floors = np.full_like(start, -1.0)
        floors[0, 2, 2, 2] = 1.0
        final, _ = run_case("vascular_floor", start, [1.0], [1, 1, 1], 0.4, [0.02], 2,
                           decay=[1.0], floors=floors)
        assert final[0, 2, 2, 2] == 1 and final[0, 1, 2, 2] > 0
        released, _ = run_case("initial_boundary_release", start, [0.0], [1, 1, 1], 0.4, [0.02], 2,
                              decay=[1.0], initial_boundary=[[1, -1, -1, -1, -1, -1]])
        error = abs(float(released[0, -1, 2, 2]) - math.exp(-0.4))
        assert error < 2e-6
        checks["vascular_floor_and_initial_boundary_release"] = {"release_error": error}
        # Lock the stable homogeneous zero-decay legacy equation before further
        # transport work. New boundaries/heterogeneous flux/decay intentionally
        # have improved semantics and are validated separately above.
        legacy_field, _ = run_case("legacy_explicit", initial, ds, spacing, 0.02, [0.02] * 3, 12, legacy=True)
        explicit_field, _ = run_case("multiscale_explicit", initial, ds, spacing, 0.02, [0.02] * 3, 12)
        error = float(np.max(abs(legacy_field - explicit_field)))
        assert error < 2e-6
        checks["stable_legacy_explicit_equivalence"] = {"linf": error}
        # Temporal refinement isolates integration error from spatial error by
        # comparing with the exact exponential of this discrete Fourier mode.
        temporal_errors = []
        exact = 1 + 0.2 * math.exp(-discrete_eigenvalue * 0.6) * mode
        for dt in (0.1, 0.05, 0.025):
            final, _ = run_case(f"temporal_dt_{dt:g}", initial[:1], [1.0], spacing, 0.6, [dt], 1)
            temporal_errors.append(float(np.max(abs(final[0] - exact))))
        assert all(1.8 < a / b < 2.2 for a, b in zip(temporal_errors, temporal_errors[1:]))
        checks["first_order_temporal_convergence"] = {"dt": [0.1, 0.05, 0.025], "linf": temporal_errors}
        # Halving h and reducing dt proportionally to h^2 should give second
        # order convergence to the continuum solution on a fixed 10-unit cube.
        spatial_errors = []
        for n in (5, 10, 20):
            h = 10.0 / n
            grid = np.indices((n, n, n))
            spatial_mode = np.prod(np.cos(np.pi * (grid + 0.5) / n), axis=0)
            field = (1 + 0.2 * spatial_mode)[None]
            # 0.48 is divisible by every requested dt, keeping dt/h^2 fixed.
            final, _ = run_case(f"spatial_nodes_{n}", field, [1.0], [h] * 3, 0.48, [0.04 * h * h], 1)
            exact = 1 + 0.2 * math.exp(-3 * (math.pi / 10)**2 * 0.48) * spatial_mode
            spatial_errors.append(float(np.max(abs(final[0] - exact))))
        # Max norm also samples slightly different points on each mesh.
        assert all(3.0 < a / b < 4.5 for a, b in zip(spatial_errors, spatial_errors[1:])), spatial_errors
        checks["second_order_spatial_convergence"] = {"nodes_per_axis": [5, 10, 20], "linf": spatial_errors}
    if gpu:
        checks["fixed_clock_guards"] = validate_fixed_clock_guards()
        directory = Path(output_dir) / "fnode_crowding" if output_dir is not None else None
        checks["fnode_crowding_and_diffusion"] = validate_fnode_diffusion(save_data_to_file, directory)
        if save_data_to_file:
            case_directories.append("fnode_crowding")
    if save_data_to_file:
        results["paraview_cases"] = case_directories
        index = "# Multiscale diffusion validation outputs\n\n"
        index += "Open a case's `diffusion.pvd` in ParaView and select `C_sp_0`, `C_sp_1`, etc. Times are seconds.\n\n"
        index += "`parameters.json` records SAVE_DATA_TO_FILE=True, the fixed groups and the geometry.\n"
        index += "`comparison.vti` (where present) includes reference concentrations and signed errors.\n\n"
        index += "\n".join(f"- [{name}]({name}/diffusion.pvd)" for name in case_directories) + "\n"
        (Path(output_dir) / "README.md").write_text(index, encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-data-to-file", action="store_true", help="SAVE_DATA_TO_FILE=True: export every main step to ParaView VTI/PVD")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "result_files" / "multiscale_validation")
    args = parser.parse_args()
    if args.save_data_to_file and not args.gpu:
        parser.error("--save-data-to-file requires --gpu")
    result = validate(args.gpu, args.save_data_to_file, args.results_dir)
    if args.gpu:
        import datetime
        import platform
        import pyflamegpu
        result["environment"] = dict(date=str(datetime.date.today()), python=platform.python_version(),
                                     pyflamegpu=pyflamegpu.VERSION_FULL, precision="float32 device, float64 reference")
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
