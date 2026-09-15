"""Plot exported GPU concentrations against independently evaluated references.

Run after tools/validate_multiscale_diffusion.py and the full-model demo.
The continuum references are closed-form Fourier series/cosine solutions;
the heterogeneous reference uses a separately assembled sparse matrix exponential.
No FLAMEGPU transport code is used to calculate these reference curves.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply


def read_legacy_ecm(path):
    """Read the ASCII point coordinates and concentrations in ECM VTK exports."""
    text = path.read_text()
    header = re.search(r"POINTS\s+(\d+)\s+\w+\s*\n", text)
    count = int(header.group(1))
    points = np.fromstring(text[header.end():], sep=" ", count=3 * count).reshape(count, 3)
    arrays = {}
    point_text = text[text.index("POINT_DATA"):]
    for match in re.finditer(r"SCALARS\s+(\w+)\s+\w+\s+1\s*\nLOOKUP_TABLE\s+default\s*\n", point_text):
        arrays[match.group(1)] = np.fromstring(point_text[match.end():], sep=" ", count=count)
    keep = arrays["is_corner"] == 0
    names = sorted(name for name in arrays if name.startswith("concentration_species_"))
    return points[keep], np.stack([arrays[name][keep] for name in names], axis=-1)


def read_vti(path):
    """Read the harness's ASCII image data, undoing VTK's x-fastest ordering."""
    image = ET.parse(path).getroot().find("ImageData")
    extent = [int(value) for value in image.attrib["WholeExtent"].split()]
    shape = tuple(extent[2 * axis + 1] - extent[2 * axis] + 1 for axis in range(3))
    spacing = np.array([float(value) for value in image.attrib["Spacing"].split()])
    arrays = {item.attrib["Name"]: np.fromstring(item.text, sep=" ").reshape(shape, order="F")
              for item in image.findall("./Piece/PointData/DataArray")}
    field = np.stack([arrays[name] for name in sorted(arrays) if name.startswith("C_sp_")])
    diffusivity = np.stack([arrays[name] for name in sorted(arrays) if name.startswith("D_sp_")])
    return field, diffusivity, spacing


def reference_dirichlet(times, x, length, diffusivity, decay, initial, left, right):
    """Continuum 1-D diffusion/decay with constant Dirichlet end concentrations.

    For D > 0, subtract the steady profile and expand the initial difference
    in sine eigenfunctions. Transverse directions have zero flux and a
    uniform initial state, so this is also the solution in the 3-D demo.
    """
    times = np.asarray(times)
    x = np.asarray(x)
    if diffusivity == 0:
        return initial * np.exp(-decay * times[:, None]) * np.ones((1, len(x)))
    if decay == 0:
        steady = left + (right - left) * x / length
    else:
        k = np.sqrt(decay / diffusivity)
        steady = (left * np.sinh(k * (length - x)) + right * np.sinh(k * x)) / np.sinh(k * length)
    n = np.arange(1, 513, dtype=float)
    beta = n * np.pi / length
    initial_coefficients = 2 * initial * (1 - (-1.0)**n) / (n * np.pi)
    steady_coefficients = 2 * n * np.pi * (left - right * (-1.0)**n) / ((decay / diffusivity) * length**2 + (n * np.pi)**2)
    decay_modes = np.exp(-times[:, None] * (diffusivity * beta**2 + decay)[None, :])
    result = steady[None, :] + (decay_modes * (initial_coefficients - steady_coefficients)[None, :]) @ np.sin(beta[:, None] * x[None, :])
    result[times == 0] = initial
    result[:, np.isclose(x, 0)] = left
    result[:, np.isclose(x, length)] = right
    return result


def diffusion_matrix(coefficients, spacing):
    """Assemble symmetric pair exchanges independently of the GPU stencil loop."""
    shape = coefficients.shape
    rows, columns, values = [], [], []
    for index in np.ndindex(shape):
        a = np.ravel_multi_index(index, shape)
        for axis in range(3):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] >= shape[axis]:
                continue
            neighbor = tuple(neighbor)
            b = np.ravel_multi_index(neighbor, shape)
            da, db = coefficients[index], coefficients[neighbor]
            weight = 0.0 if da + db == 0 else 2 * da * db / (da + db) / spacing[axis]**2
            rows.extend((a, a, b, b))
            columns.extend((a, b, a, b))
            values.extend((-weight, weight, weight, -weight))
    count = int(np.prod(shape))
    return coo_matrix((values, (rows, columns)), shape=(count, count)).tocsr()


def draw_comparison(directory, name, title, point_names, coordinates, times, computed,
                    reference_times, reference, reference_at_samples, reference_label):
    """Save side-by-side plots, an error plot, and the actual sampled values."""
    colors = ["#2477b3", "#d67b1f", "#218564"]
    species_count = computed.shape[-1]
    fig, axes = plt.subplots(len(point_names), 2, figsize=(12, 10), sharex=True, squeeze=False)
    for row, point_name in enumerate(point_names):
        lo = min(computed[:, row].min(), reference[:, row].min(), 0)
        hi = max(computed[:, row].max(), reference[:, row].max())
        for species in range(species_count):
            axes[row, 0].plot(times, computed[:, row, species], "o", ms=4,
                              color=colors[species], label=f"Species {species}")
            axes[row, 1].plot(reference_times, reference[:, row, species], lw=1.8,
                              color=colors[species], label=f"Species {species}")
        coordinate_text = ", ".join(f"{value:g}" for value in coordinates[row])
        for column, label in enumerate(("GPU result", reference_label)):
            ax = axes[row, column]
            ax.set_title(f"{point_name}  ({coordinate_text}) µm\n{label}", fontsize=10, loc="left")
            ax.set_ylim(lo - 0.03 * max(hi - lo, 1e-6), hi + 0.08 * max(hi - lo, 1e-6))
            ax.set_xlim(min(times[0], reference_times[0]), max(times[-1], reference_times[-1]))
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[row, 0].set_ylabel("Concentration [µM]")
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    axes[0, 1].legend(loc="best", frameon=False)
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.text(0.5, 0.006, "Markers: saved GPU values. Lines: independent reference. No interpolation between GPU samples; matched axis ranges.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(directory / f"{name}.png", dpi=180)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)

    error = computed - reference_at_samples
    fig, axes = plt.subplots(1, len(point_names), figsize=(12, 3.8), squeeze=False)
    for point, point_name in enumerate(point_names):
        for species in range(species_count):
            axes[0, point].plot(times, error[:, point, species], "o-", ms=3,
                                color=colors[species], label=f"Species {species}")
        axes[0, point].set_title(point_name)
        axes[0, point].set_xlabel("Time [s]")
        axes[0, point].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[0, point].grid(alpha=0.2)
    axes[0, 0].set_ylabel("GPU − reference [µM]")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(title + " — sampled differences", fontsize=12)
    fig.tight_layout()
    fig.savefig(directory / f"{name}_errors.png", dpi=180)
    plt.close(fig)

    with (directory / f"{name}_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "point", "x_um", "y_um", "z_um", "species", "gpu_concentration", "reference_concentration", "difference"])
        for step, time in enumerate(times):
            for point, point_name in enumerate(point_names):
                for species in range(species_count):
                    writer.writerow([time, point_name, *coordinates[point], species,
                                     computed[step, point, species], reference_at_samples[step, point, species], error[step, point, species]])
    return dict(reference=reference_label, sampled_max_absolute_error=float(np.max(abs(error))),
                sampled_rms_error=float(np.sqrt(np.mean(error**2))),
                points=[dict(name=name, coordinates_um=list(map(float, xyz))) for name, xyz in zip(point_names, coordinates)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("result_files/multiscale_diffusion_review"))
    args = parser.parse_args()
    output = args.results_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    report = {}

    # Full CellFoundry run: its planar boundaries allow a continuum Fourier solution.
    case = args.results_dir / "full_model/diffusion"
    parameters = json.loads((case / "parameters.json").read_text())
    files = sorted(case.glob("ecm_data_t*.vtk"))
    positions, first = read_legacy_ecm(files[0])
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    domain_size = upper - lower
    center = (lower + upper) / 2
    requests = [[upper[0], center[1], center[2]], center,
                lower + domain_size * np.array([0.25, 0.7, 0.5])]
    selected = [int(np.argmin(np.sum((positions - point)**2, axis=1))) for point in requests]
    coordinates = positions[selected]
    computed = np.stack([read_legacy_ecm(path)[1][selected] for path in files])
    times = np.array([int(path.stem.rsplit("t", 1)[1]) * parameters["TIME_STEP"] for path in files])
    dense = np.linspace(times[0], times[-1], 401)
    minimum_x, maximum_x = positions[:, 0].min(), positions[:, 0].max()

    def full_reference(t):
        return np.stack([reference_dirichlet(t, coordinates[:, 0] - minimum_x, maximum_x - minimum_x,
                                            parameters["DIFFUSION_COEFF_MULTI"][s], parameters["ECM_DEGRADATION_RATE_MULTI"][s],
                                            parameters["INIT_ECM_CONCENTRATION_VALS"][s],
                                            parameters["BOUNDARY_CONC_FIXED_MULTI"][s][1], parameters["BOUNDARY_CONC_FIXED_MULTI"][s][0])
                         for s in range(3)], axis=-1)

    domain_label = " × ".join(f"{length:g}" for length in domain_size)
    report["full_model"] = draw_comparison(output, "full_model", f"CellFoundry on the {domain_label} µm domain",
                                            ["Boundary (+X)", "Nearest center point", "Interior point"], coordinates,
                                            times, computed, dense, full_reference(dense), full_reference(times), "Continuum analytical solution")

    # Cosine case: compare against the continuum PDE, and separately quantify
    # agreement with the exact amplification of the discrete Euler algorithm.
    case = args.results_dir / "numerical/three_species_cosine"
    parameters = json.loads((case / "parameters.json").read_text())
    files = sorted(case.glob("ecm_data_t*.vti"))
    initial, _, spacing = read_vti(files[0])
    shape = initial.shape[1:]
    points = [(0, 2, 2), (4, 4, 4), (2, 2, 2)]
    coordinates = np.array(points) * spacing
    computed = np.array([[read_vti(path)[0][(slice(None),) + point] for point in points] for path in files])
    times = np.arange(len(files)) * parameters["TIME_STEP"]
    dense = np.linspace(0, times[-1], 401)
    phi = np.prod(np.cos(np.pi * (np.array(points) + 0.5) / np.array(shape)), axis=1)
    eigenvalue = sum((np.pi / (n * h))**2 for n, h in zip(shape, spacing))
    discrete_eigenvalue = sum(4 * np.sin(np.pi / (2 * n))**2 / h**2 for n, h in zip(shape, spacing))
    amplitudes = [0.2, 0.3, 0.1]

    def cosine_reference(t):
        return np.stack([(1 + amplitude * phi[None, :] * np.exp(-d * eigenvalue * t[:, None])) * np.exp(-loss * t[:, None])
                         for amplitude, d, loss in zip(amplitudes, parameters["DIFFUSION_COEFF_MULTI"], parameters["ECM_DEGRADATION_RATE_MULTI"])], axis=-1)

    report["cosine"] = draw_comparison(output, "cosine", "Three-species cosine decay: independent diffusion clocks",
                                        ["Boundary (−X)", "Center", "Interior point"], coordinates, times, computed,
                                        dense, cosine_reference(dense), cosine_reference(times), "Continuum analytical solution")
    discrete = np.empty_like(computed)
    for group in parameters["groups"]:
        for s in group["species"]:
            factor = (1 - parameters["DIFFUSION_COEFF_MULTI"][s] * group["dt"] * discrete_eigenvalue)**(np.arange(len(times)) * group["substeps"])
            discrete[:, :, s] = (1 + amplitudes[s] * factor[:, None] * phi[None, :]) * np.exp(-parameters["ECM_DEGRADATION_RATE_MULTI"][s] * times[:, None])
    report["cosine"]["sampled_error_vs_discrete_algorithm"] = float(np.max(abs(computed - discrete)))

    # Heterogeneous D: exp(t*A) integrates the same spatial graph exactly in time.
    # A is assembled here by visiting undirected pairs once, independently of
    # the GPU agent loop and the NumPy explicit-Euler validation reference.
    case = args.results_dir / "numerical/heterogeneous_no_flux"
    parameters = json.loads((case / "parameters.json").read_text())
    files = sorted(case.glob("ecm_data_t*.vti"))
    initial, coefficients, spacing = read_vti(files[0])
    shape = initial.shape[1:]
    points = [(0, 2, 2), (2, 2, 1), (4, 1, 1)]
    coordinates = np.array(points) * spacing
    selected = [np.ravel_multi_index(point, shape) for point in points]
    times = np.arange(len(files)) * parameters["TIME_STEP"]
    dense = np.linspace(0, times[-1], 401)
    computed = np.array([[read_vti(path)[0][(slice(None),) + point] for point in points] for path in files])
    reference = []
    for s in range(len(initial)):
        matrix = diffusion_matrix(coefficients[s], spacing)
        assert np.max(abs(np.asarray(matrix.sum(axis=0)))) < 1e-12
        values = expm_multiply(matrix, initial[s].ravel(), start=0, stop=times[-1], num=len(dense))
        reference.append(values[:, selected])
    reference = np.stack(reference, axis=-1)
    sample_indices = np.rint(times / dense[-1] * (len(dense) - 1)).astype(int)
    report["heterogeneous"] = draw_comparison(output, "heterogeneous", "Spatially varying diffusivity and an impermeable sheet",
                                               ["Boundary (−X)", "Nearest center / sheet", "Low-diffusivity region"], coordinates,
                                               times, computed, dense, reference, reference[sample_indices], "Matrix exponential on the same grid")
    (output / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
