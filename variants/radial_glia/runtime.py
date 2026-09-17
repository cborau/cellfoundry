"""Radial-glia initialization, VTK fields and metrics (independent of model.py)."""
import numpy as np
import pandas as pd


def initialize_cell(instance, rng):
    _ap_angle = rng.uniform(0.0, 2.0 * np.pi)  # random in-plane apical direction (no established z-polarity)
    instance.setVariableFloat("apx", float(np.cos(_ap_angle)))
    instance.setVariableFloat("apy", float(np.sin(_ap_angle)))
    instance.setVariableFloat("apz", 0.0)
    instance.setVariableFloat("rg_commit_level",         0.0)  # start at zero; sp2 gradient determines spatial nucleation site
    instance.setVariableFloat("epithelialization_level", 0.0)
    instance.setVariableFloat("rosette_maturity",        0.0)
    instance.setVariableFloat("rg_neighbour_density",    0.0)
    instance.setVariableFloat("morphogen_local",           0.0)
    instance.setVariableInt("rg_committed", 0)
    instance.setVariableFloat("substrate_anchor_x", float(instance.getVariableFloat("x")))
    instance.setVariableFloat("substrate_anchor_y", float(instance.getVariableFloat("y")))

CELL_VTK_EXTRA_SCALARS = [
    ("rg_commit_level",         "rg_commit_level",         "float"),
    ("epithelialization_level", "epithelialization_level", "float"),
    ("rosette_maturity",        "rosette_maturity",        "float"),
    ("rg_neighbour_density",    "rg_neighbour_density",    "float"),
    ("morphogen_local",        "morphogen_local",        "float"),
    ("rg_committed",            "rg_committed",            "int"),
]
CELL_VTK_EXTRA_VECTORS = [
    ("apical_vector", "apx", "apy", "apz"),
]

class Metrics:
    def __init__(self, ctx):
        self.ctx = ctx

    def initialize(self, FLAMEGPU):
        results = self.ctx.runtime_results(FLAMEGPU)
        results["RG_FINAL_METRICS"] = pd.DataFrame()
        results["RG_ROSETTE_METRICS_OVER_TIME"] = pd.DataFrame()

    def run(self, FLAMEGPU):
        config = self.ctx.config
        results = self.ctx.runtime_results(FLAMEGPU)
        STEPS = config["STEPS"]
        SAVE_EVERY_N_STEPS = config["SAVE_EVERY_N_STEPS"]
        CELL_RADIUS = config["CELL_RADIUS"]
        MIN_ROSETTE_SIZE = config["MIN_ROSETTE_SIZE"]
        step = FLAMEGPU.getStepCounter() + 1
        is_final = step == STEPS
        # --- RG rosette metrics over time (radial_glia variant only) ---
        if (step == 1 or step % SAVE_EVERY_N_STEPS == 0 or is_final):
            from sklearn.cluster import DBSCAN as _DBSCAN
            cell_agent = FLAMEGPU.agent("CELL")
            rg_positions = []
            rg_maturities = []
            rg_apz_vals = []
            n_alive_total = 0
            n_alive_rg = 0
            for ai in cell_agent.getPopulationData():
                if int(ai.getVariableInt("dead")) != 0:
                    continue
                n_alive_total += 1
                ct = int(ai.getVariableInt("cell_type"))
                if ct == 2:  # RG
                    n_alive_rg += 1
                    rg_positions.append([
                        float(ai.getVariableFloat("x")),
                        float(ai.getVariableFloat("y")),
                        float(ai.getVariableFloat("z")),
                    ])
                    rg_maturities.append(float(ai.getVariableFloat("rosette_maturity")))
                    rg_apz_vals.append(abs(float(ai.getVariableFloat("apz"))))

            rg_fraction = n_alive_rg / n_alive_total if n_alive_total > 0 else 0.0
            mean_rosette_maturity_val = float(np.mean(rg_maturities)) if rg_maturities else 0.0
            mean_apz_val = float(np.mean(rg_apz_vals)) if rg_apz_vals else 0.0

            pos_arr = np.array(rg_positions) if rg_positions else np.zeros((0, 3))
            labels = np.full(n_alive_rg, -1, dtype=int)
            n_rg_clusters = 0
            mean_cluster_size_val = 0.0
            largest_cluster_size = 0
            if n_alive_rg >= 2:
                eps_cluster = 3.0 * float(CELL_RADIUS[2])  # 3 × RG cell radius ≈ 15 µm
                labels = _DBSCAN(eps=eps_cluster, min_samples=2).fit_predict(pos_arr)
                clustered = labels[labels >= 0]
                if len(clustered) > 0:
                    n_rg_clusters = int(np.unique(clustered).shape[0])
                    cluster_sizes = np.bincount(clustered)
                    mean_cluster_size_val = float(cluster_sizes.mean())
                    largest_cluster_size = int(cluster_sizes.max())
            elif n_alive_rg == 1:
                n_rg_clusters = 1
                mean_cluster_size_val = 1.0
                largest_cluster_size = 1

            # --- Large-cluster metrics (min MIN_ROSETTE_SIZE cells = minimum viable rosette) ---
            n_large_rg_clusters = 0
            large_cluster_fraction = 0.0
            large_cluster_mean_size = 0.0
            if n_alive_rg >= 2:
                _all_labels = labels[labels >= 0]
                if len(_all_labels) > 0:
                    _cs = np.bincount(_all_labels)
                    _large = _cs[_cs >= MIN_ROSETTE_SIZE]
                    n_large_rg_clusters = int(len(_large))
                    if n_large_rg_clusters > 0:
                        large_cluster_fraction = float(_large.sum()) / n_alive_rg
                        large_cluster_mean_size = float(_large.mean())

            # --- Compactness metrics (PCA eigenvalue ratio; 0=linear, 1=circular) ---
            # rg_assembly_compactness: shape of the entire RG assembly in XY
            rg_assembly_compactness_val = 0.0
            if n_alive_rg >= 3:
                cov_all = np.cov(pos_arr[:, :2].T)
                ev_all = np.linalg.eigvalsh(cov_all)  # ascending order
                if ev_all[-1] > 1e-12:
                    rg_assembly_compactness_val = float(ev_all[0] / ev_all[-1])

            # mean_cluster_compactness: per-cluster PCA compactness, weighted by size
            mean_cluster_compactness_val = 0.0
            if n_rg_clusters > 0 and n_alive_rg >= 3:
                weighted_sum = 0.0
                total_weight = 0
                for _lid in range(n_rg_clusters):
                    _mask = (labels == _lid)
                    _sz = int(_mask.sum())
                    if _sz < MIN_ROSETTE_SIZE:
                        continue
                    _cxy = pos_arr[np.where(_mask)[0], :2]
                    _cov = np.cov(_cxy.T)
                    _ev = np.linalg.eigvalsh(_cov)
                    if _ev[-1] > 1e-12:
                        weighted_sum += (_ev[0] / _ev[-1]) * _sz
                        total_weight += _sz
                if total_weight > 0:
                    mean_cluster_compactness_val = weighted_sum / total_weight

            time_val = step * FLAMEGPU.environment.getPropertyFloat("TIME_STEP")
            rg_row = pd.DataFrame([{
                "step": step,
                "time": time_val,
                "n_alive_total": n_alive_total,
                "n_alive_rg": n_alive_rg,
                "rg_fraction": rg_fraction,
                "n_rg_clusters": n_rg_clusters,
                "n_large_rg_clusters": n_large_rg_clusters,
                "large_cluster_fraction": large_cluster_fraction,
                "large_cluster_mean_size": large_cluster_mean_size,
                "mean_cluster_size": mean_cluster_size_val,
                "largest_cluster_size": largest_cluster_size,
                "mean_rosette_maturity": mean_rosette_maturity_val,
                "mean_apz": mean_apz_val,
                "rg_assembly_compactness": rg_assembly_compactness_val,
                "mean_cluster_compactness": mean_cluster_compactness_val,
            }])
            if len(results["RG_ROSETTE_METRICS_OVER_TIME"]) == 0:
                results["RG_ROSETTE_METRICS_OVER_TIME"] = rg_row
            else:
                results["RG_ROSETTE_METRICS_OVER_TIME"] = pd.concat(
                    [results["RG_ROSETTE_METRICS_OVER_TIME"], rg_row], ignore_index=True
                )

        # --- RG final snapshot (radial_glia variant only, final step only) ---
        if is_final:
            rg_rows = []
            cell_agent = FLAMEGPU.agent("CELL")
            for ai in cell_agent.getPopulationData():
                rg_rows.append({
                    "id":                    int(ai.getVariableInt("id")),
                    "cell_type":             int(ai.getVariableInt("cell_type")),
                    "dead":                  int(ai.getVariableInt("dead")),
                    "mother_id":             int(ai.getVariableInt("mother_id")),
                    "rg_commit_level":       float(ai.getVariableFloat("rg_commit_level")),
                    "epithelialization_level": float(ai.getVariableFloat("epithelialization_level")),
                    "rosette_maturity":      float(ai.getVariableFloat("rosette_maturity")),
                    "rg_neighbour_density":  float(ai.getVariableFloat("rg_neighbour_density")),
                    "morphogen_local":       float(ai.getVariableFloat("morphogen_local")),
                    "rg_committed":          int(ai.getVariableInt("rg_committed")),
                    "apx":                   float(ai.getVariableFloat("apx")),
                    "apy":                   float(ai.getVariableFloat("apy")),
                    "apz":                   float(ai.getVariableFloat("apz")),
                })
            results["RG_FINAL_METRICS"] = pd.DataFrame(rg_rows)


class DebugStats:
    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, FLAMEGPU):
        DEBUG_PRINT_INTERVAL = self.ctx.config["DEBUG_PRINT_INTERVAL"]
        N_CELL_TYPES = self.ctx.config["N_CELL_TYPES"]

        if DEBUG_PRINT_INTERVAL <= 0:
            return

        step = FLAMEGPU.getStepCounter() + 1
        if step != 1 and step % DEBUG_PRINT_INTERVAL != 0:
            return

        time_h = step * FLAMEGPU.environment.getPropertyFloat("TIME_STEP") / 3600.0
        type_counts = [0] * N_CELL_TYPES
        commit_sum = 0.0
        commit_max = 0.0
        morph_sum  = 0.0
        morph_max  = 0.0
        n_alive = 0

        for ai in FLAMEGPU.agent("CELL").getPopulationData():
            if int(ai.getVariableInt("dead")) == 1:
                continue
            ct = int(ai.getVariableInt("cell_type"))
            if 0 <= ct < N_CELL_TYPES:
                type_counts[ct] += 1
            commit = float(ai.getVariableFloat("rg_commit_level"))
            morph  = float(ai.getVariableFloat("morphogen_local"))
            commit_sum += commit
            morph_sum  += morph
            if commit > commit_max:
                commit_max = commit
            if morph > morph_max:
                morph_max = morph
            n_alive += 1

        mean_commit = commit_sum / n_alive if n_alive > 0 else 0.0
        mean_morph  = morph_sum  / n_alive if n_alive > 0 else 0.0
        _names = ["iPSC", "NEP", "RG"]
        type_str = "  ".join(
            f"{(_names[i] if i < len(_names) else f'type{i}')}={type_counts[i]}"
            for i in range(N_CELL_TYPES)
        )
        print(
            f"[DBG t={time_h:6.2f}h step={step:5d}]  {type_str}"
            f"  | rg_commit mean={mean_commit:.4f} max={commit_max:.4f}"
            f"  | morphogen mean={mean_morph:.3e} max={morph_max:.3e}"
        )

