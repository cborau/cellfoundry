"""
Generate a random connected vascular network inside a cuboidal domain.

The generated network is represented as point agents connected by edges. It can be
used as a preprocessing step for a FLAME GPU 2 / Cellfoundry model, or inspected
independently with the included 3D plotting function.

Boundary order used by command line inputs:
    [+x, -x, +y, -y, +z, -z]

Example
-------
python generate_vascular_network.py \
    --bounds 50 -50 50 -50 50 -50 \
    --diameter 5 \
    --density 0.02 \
    --resolution 5 \
    --branching-probability 0.08 \
    --nucleation-faces 1 1 0 0 0 0 \
    --nucleation-per-face 8 \
    --seed 1 \
    --output vascular_network.pickle \
    --plot
"""

from __future__ import annotations

import argparse
import sys
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


FACE_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")
FACE_INWARD_DIRECTIONS = {
    0: np.array([-1.0, 0.0, 0.0], dtype=float),  # +x face points inward along -x
    1: np.array([1.0, 0.0, 0.0], dtype=float),   # -x face points inward along +x
    2: np.array([0.0, -1.0, 0.0], dtype=float),  # +y face points inward along -y
    3: np.array([0.0, 1.0, 0.0], dtype=float),   # -y face points inward along +y
    4: np.array([0.0, 0.0, -1.0], dtype=float),  # +z face points inward along -z
    5: np.array([0.0, 0.0, 1.0], dtype=float),   # -z face points inward along +z
}


@dataclass(frozen=True)
class DomainBounds:
    """Axis-aligned cuboidal domain bounds."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def from_plus_minus_order(cls, values: Sequence[float]) -> "DomainBounds":
        """
        Create bounds from the user-facing order [+x, -x, +y, -y, +z, -z].

        For example, [500, -500, 500, -500, 500, -500] creates a cube from
        -500 to +500 in each coordinate.
        """
        if len(values) != 6:
            raise ValueError("Bounds must contain 6 values in order [+x, -x, +y, -y, +z, -z].")
        xmax, xmin, ymax, ymin, zmax, zmin = map(float, values)
        bounds = cls(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)
        bounds.validate()
        return bounds

    def validate(self) -> None:
        if not (self.xmin < self.xmax and self.ymin < self.ymax and self.zmin < self.zmax):
            raise ValueError("Invalid domain bounds. Each min boundary must be smaller than its max boundary.")

    @property
    def lengths(self) -> np.ndarray:
        return np.array([self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin], dtype=float)

    @property
    def volume(self) -> float:
        lx, ly, lz = self.lengths
        return float(lx * ly * lz)

    @property
    def minimum_corner(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.zmin], dtype=float)

    @property
    def maximum_corner(self) -> np.ndarray:
        return np.array([self.xmax, self.ymax, self.zmax], dtype=float)

    def contains(self, p: np.ndarray, eps: float = 1.0e-9) -> bool:
        return bool(
            self.xmin - eps <= p[0] <= self.xmax + eps
            and self.ymin - eps <= p[1] <= self.ymax + eps
            and self.zmin - eps <= p[2] <= self.zmax + eps
        )

    def clamped(self, p: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(p, self.minimum_corner), self.maximum_corner)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class VesselNode:
    """
    Vascular point-agent node.

    parent_ids is a list of parent node ids. Seed nodes placed on nucleation
    faces have parent_ids = [-2] (-2 is the source/boundary sentinel, distinct
    from -1 which means empty slot). Growth nodes start with parent_ids =
    [growth_parent_id]. When an anastomosis connection is formed, the
    anastomosis parent id is appended, giving the child node multiple parents.
    children_ids stores outgoing adjacency.
    """

    id: int
    x: float
    y: float
    z: float
    parent_ids: List[int] = field(default_factory=lambda: [-2])
    children_ids: List[int] = field(default_factory=list)
    tree_id: int = -1
    is_boundary: bool = False
    boundary_face: int = -1

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class VesselEdge:
    """Connection between two vascular nodes."""

    parent_id: int
    child_id: int
    length: float
    edge_type: str  # "growth" or "anastomosis"


@dataclass
class ActiveTip:
    """Terminal node that can keep growing."""

    node_id: int
    direction: np.ndarray
    tree_id: int


def unit_vector(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1.0e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return unit_vector(v)


def random_initial_direction(
    rng: np.random.Generator,
    inward_direction: np.ndarray,
    inward_bias: float,
) -> np.ndarray:
    """
    Direction for the first segment from a boundary seed.

    inward_bias=1 gives a straight inward segment. Lower values increase the
    random component and therefore tortuosity near the boundary.
    """
    inward_bias = float(np.clip(inward_bias, 0.0, 1.0))
    v = inward_bias * inward_direction + (1.0 - inward_bias) * random_unit_vector(rng)
    if np.linalg.norm(v) <= 1.0e-12:
        v = inward_direction.copy()
    return unit_vector(v)


def random_next_direction(
    rng: np.random.Generator,
    previous_direction: np.ndarray,
    direction_persistence: float,
) -> np.ndarray:
    """
    Direction for continuation of a vessel branch.

    direction_persistence close to 1 produces straighter vessels. Smaller values
    produce more tortuous trajectories.
    """
    direction_persistence = float(np.clip(direction_persistence, 0.0, 1.0))
    v = direction_persistence * previous_direction + (1.0 - direction_persistence) * random_unit_vector(rng)
    if np.linalg.norm(v) <= 1.0e-12:
        v = previous_direction.copy()
    return unit_vector(v)


def random_point_on_face(
    rng: np.random.Generator,
    bounds: DomainBounds,
    face_idx: int,
) -> np.ndarray:
    """Sample a point uniformly from one cuboid face."""
    x = rng.uniform(bounds.xmin, bounds.xmax)
    y = rng.uniform(bounds.ymin, bounds.ymax)
    z = rng.uniform(bounds.zmin, bounds.zmax)

    if face_idx == 0:
        x = bounds.xmax
    elif face_idx == 1:
        x = bounds.xmin
    elif face_idx == 2:
        y = bounds.ymax
    elif face_idx == 3:
        y = bounds.ymin
    elif face_idx == 4:
        z = bounds.zmax
    elif face_idx == 5:
        z = bounds.zmin
    else:
        raise ValueError("face_idx must be in [0, 5].")

    return np.array([x, y, z], dtype=float)


def find_boundary_face(p: np.ndarray, bounds: DomainBounds, eps: float = 1.0e-6) -> int:
    """Return the index of the boundary face touched by p, or -1 if none."""
    candidates = [
        (0, abs(p[0] - bounds.xmax)),
        (1, abs(p[0] - bounds.xmin)),
        (2, abs(p[1] - bounds.ymax)),
        (3, abs(p[1] - bounds.ymin)),
        (4, abs(p[2] - bounds.zmax)),
        (5, abs(p[2] - bounds.zmin)),
    ]
    face_idx, distance = min(candidates, key=lambda item: item[1])
    return face_idx if distance <= eps else -1


def segment_exit_point(p0: np.ndarray, p1: np.ndarray, bounds: DomainBounds) -> np.ndarray:
    """
    Intersect segment p0 -> p1 with the domain box and return the exit point.

    p0 is assumed to be inside the domain and p1 outside. The returned point is
    clamped to avoid small floating-point excursions beyond the domain.
    """
    d = p1 - p0
    t_candidates: List[float] = []

    axis_bounds = [
        (bounds.xmin, bounds.xmax),
        (bounds.ymin, bounds.ymax),
        (bounds.zmin, bounds.zmax),
    ]

    for axis, (lower, upper) in enumerate(axis_bounds):
        if d[axis] > 0.0 and p1[axis] > upper:
            t_candidates.append((upper - p0[axis]) / d[axis])
        elif d[axis] < 0.0 and p1[axis] < lower:
            t_candidates.append((lower - p0[axis]) / d[axis])

    valid_t = [t for t in t_candidates if 0.0 <= t <= 1.0]
    if not valid_t:
        return bounds.clamped(p1)

    t_exit = min(valid_t)
    return bounds.clamped(p0 + t_exit * d)


def brute_force_nearest_node(
    query_point: np.ndarray,
    nodes: Sequence[VesselNode],
    min_distance: float,
    excluded_tree_id: Optional[int],
    excluded_node_ids: Iterable[int],
    allow_self_connections: bool,
) -> Tuple[Optional[int], float]:
    """Find the nearest existing node satisfying the connection rules."""
    excluded = set(excluded_node_ids)
    best_id: Optional[int] = None
    best_dist = float("inf")

    for node in nodes:
        if node.id in excluded:
            continue
        if not allow_self_connections and excluded_tree_id is not None and node.tree_id == excluded_tree_id:
            continue
        d = float(np.linalg.norm(query_point - node.position()))
        if d < best_dist:
            best_dist = d
            best_id = node.id

    if best_id is not None and best_dist <= min_distance:
        return best_id, best_dist
    return None, best_dist


def add_node(
    nodes: List[VesselNode],
    position: np.ndarray,
    parent_id: int,
    tree_id: int,
    is_boundary: bool,
    boundary_face: int,
) -> int:
    node_id = len(nodes)
    nodes.append(
        VesselNode(
            id=node_id,
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            parent_ids=[-2 if parent_id < 0 else int(parent_id)],  # -2 = source sentinel; >=0 = growth parent
            tree_id=int(tree_id),
            is_boundary=bool(is_boundary),
            boundary_face=int(boundary_face),
        )
    )
    if parent_id >= 0:
        nodes[parent_id].children_ids.append(node_id)
    return node_id


def add_edge(
    nodes: List[VesselNode],
    edges: List[VesselEdge],
    parent_id: int,
    child_id: int,
    edge_type: str,
) -> float:
    p0 = nodes[parent_id].position()
    p1 = nodes[child_id].position()
    length = float(np.linalg.norm(p1 - p0))
    if length <= 1.0e-12:
        return 0.0

    # Growth edges already add the new node to children_ids in add_node().
    # Anastomosis edges link to an existing node and must add adjacency here.
    if edge_type == "anastomosis":
        if child_id not in nodes[parent_id].children_ids:
            nodes[parent_id].children_ids.append(child_id)
        # Register the anastomosis parent in the child's parent list
        if parent_id not in nodes[child_id].parent_ids:
            nodes[child_id].parent_ids.append(parent_id)

    edges.append(VesselEdge(parent_id=parent_id, child_id=child_id, length=length, edge_type=edge_type))
    return length


def create_seed_batch(
    rng: np.random.Generator,
    bounds: DomainBounds,
    nodes: List[VesselNode],
    active_tips: List[ActiveTip],
    nucleation_faces: Sequence[int],
    nucleation_points_per_face: int,
    tree_id_start: int,
    initial_inward_bias: float,
) -> int:
    """Create one batch of seed nodes on all enabled nucleation faces."""
    tree_id = tree_id_start
    for face_idx, enabled in enumerate(nucleation_faces):
        if int(enabled) == 0:
            continue
        for _ in range(nucleation_points_per_face):
            seed_position = random_point_on_face(rng, bounds, face_idx)
            seed_id = add_node(
                nodes=nodes,
                position=seed_position,
                parent_id=-1,
                tree_id=tree_id,
                is_boundary=True,
                boundary_face=face_idx,
            )
            direction = random_initial_direction(
                rng=rng,
                inward_direction=FACE_INWARD_DIRECTIONS[face_idx],
                inward_bias=initial_inward_bias,
            )
            active_tips.append(ActiveTip(node_id=seed_id, direction=direction, tree_id=tree_id))
            tree_id += 1
    return tree_id


def generate_vascular_network(
    bounds_plus_minus: Sequence[float],
    diameter: float,
    vascularization_density: float,
    resolution: float,
    branching_probability: float,
    nucleation_faces: Sequence[int],
    nucleation_points_per_face: int = 8,
    min_connection_distance: Optional[float] = None,
    direction_persistence: float = 0.85,
    initial_inward_bias: float = 0.80,
    allow_self_connections: bool = False,
    max_nodes: int = 1_000_000,
    max_seed_batches: int = 100,
    seed: Optional[int] = None,
    save_vtk: bool = False,
) -> Dict[str, object]:
    """
    Generate a stochastic vascular network in a cuboidal domain.

    Parameters
    ----------
    bounds_plus_minus:
        Domain coordinates in order [+x, -x, +y, -y, +z, -z].
    diameter:
        Vessel diameter in the same length unit as the domain.
    vascularization_density:
        Target vessel volume fraction: vessel volume / domain volume.
    resolution:
        Approximate distance between consecutive vascular nodes.
    branching_probability:
        Probability that a growing tip creates two children instead of one.
    nucleation_faces:
        Six integers in order [+x, -x, +y, -y, +z, -z]. A value of 1 enables
        nucleation on that face; 0 disables it.
    nucleation_points_per_face:
        Number of seed points created per enabled face and seed batch.
    min_connection_distance:
        Distance below which a tip is connected to another vessel. If None,
        0.75 * resolution is used.
    direction_persistence:
        Controls tortuosity after the first segment. Values closer to 1 produce
        straighter branches.
    initial_inward_bias:
        Controls how strongly boundary seeds point into the domain. Values closer
        to 1 reduce immediate tangential growth along the boundary.
    allow_self_connections:
        If False, tips only connect to nodes from another seeded tree.
    max_nodes:
        Safety limit for the number of generated nodes.
    max_seed_batches:
        If all active tips terminate before the target density is reached, the
        generator can create additional seed batches up to this limit.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        Pickle-ready network with metadata, nodes, and edges.
    """
    bounds = DomainBounds.from_plus_minus_order(bounds_plus_minus)
    if len(nucleation_faces) != 6:
        raise ValueError("nucleation_faces must contain 6 values in order [+x, -x, +y, -y, +z, -z].")
    if not any(int(v) != 0 for v in nucleation_faces):
        raise ValueError("At least one nucleation face must be enabled.")
    if diameter <= 0.0:
        raise ValueError("diameter must be positive.")
    if vascularization_density <= 0.0:
        raise ValueError("vascularization_density must be positive.")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive.")
    if not 0.0 <= branching_probability <= 1.0:
        raise ValueError("branching_probability must be in [0, 1].")
    if nucleation_points_per_face <= 0:
        raise ValueError("nucleation_points_per_face must be positive.")

    rng = np.random.default_rng(seed)
    min_connection_distance = 0.75 * resolution if min_connection_distance is None else float(min_connection_distance)

    vessel_cross_section_area = math.pi * (0.5 * diameter) ** 2
    target_vessel_volume = vascularization_density * bounds.volume
    target_total_length = target_vessel_volume / vessel_cross_section_area

    nodes: List[VesselNode] = []
    edges: List[VesselEdge] = []
    active_tips: List[ActiveTip] = []

    total_length = 0.0
    next_tree_id = 0
    seed_batches = 0

    while total_length < target_total_length and len(nodes) < max_nodes:
        if not active_tips:
            if seed_batches >= max_seed_batches:
                break
            next_tree_id = create_seed_batch(
                rng=rng,
                bounds=bounds,
                nodes=nodes,
                active_tips=active_tips,
                nucleation_faces=nucleation_faces,
                nucleation_points_per_face=nucleation_points_per_face,
                tree_id_start=next_tree_id,
                initial_inward_bias=initial_inward_bias,
            )
            seed_batches += 1
            print(f"Seed batch {seed_batches}: nodes={len(nodes)}, active_tips={len(active_tips)}")

        rng.shuffle(active_tips)
        new_active_tips: List[ActiveTip] = []

        for tip in active_tips:
            if total_length >= target_total_length or len(nodes) >= max_nodes:
                break

            n_children = 2 if rng.random() < branching_probability else 1
            parent = nodes[tip.node_id]
            parent_position = parent.position()

            for _ in range(n_children):
                if total_length >= target_total_length or len(nodes) >= max_nodes:
                    break

                direction = random_next_direction(
                    rng=rng,
                    previous_direction=tip.direction,
                    direction_persistence=direction_persistence,
                )
                proposed_position = parent_position + resolution * direction

                if not bounds.contains(proposed_position):
                    boundary_position = segment_exit_point(parent_position, proposed_position, bounds)
                    boundary_face = find_boundary_face(boundary_position, bounds)
                    child_id = add_node(
                        nodes=nodes,
                        position=boundary_position,
                        parent_id=parent.id,
                        tree_id=tip.tree_id,
                        is_boundary=True,
                        boundary_face=boundary_face,
                    )
                    total_length += add_edge(nodes, edges, parent.id, child_id, edge_type="growth")
                    continue

                excluded_node_ids = {parent.id}
                nearest_id, _ = brute_force_nearest_node(
                    query_point=proposed_position,
                    nodes=nodes,
                    min_distance=min_connection_distance,
                    excluded_tree_id=tip.tree_id,
                    excluded_node_ids=excluded_node_ids,
                    allow_self_connections=allow_self_connections,
                )

                if nearest_id is not None:
                    total_length += add_edge(nodes, edges, parent.id, nearest_id, edge_type="anastomosis")
                    continue

                child_id = add_node(
                    nodes=nodes,
                    position=proposed_position,
                    parent_id=parent.id,
                    tree_id=tip.tree_id,
                    is_boundary=False,
                    boundary_face=-1,
                )
                total_length += add_edge(nodes, edges, parent.id, child_id, edge_type="growth")
                new_active_tips.append(ActiveTip(node_id=child_id, direction=direction, tree_id=tip.tree_id))

        active_tips = new_active_tips
        pct = 100.0 * min(total_length / target_total_length, 1.0)
        print(f"Progress: {pct:.1f}% of target | nodes={len(nodes)} | edges={len(edges)} | active_tips={len(active_tips)}")

    achieved_vessel_volume = total_length * vessel_cross_section_area
    achieved_density = achieved_vessel_volume / bounds.volume

    _result = {
        "metadata": {
            "format": "vascular_network_v1",
            "bounds_order": "+x -x +y -y +z -z",
            "bounds": bounds.as_dict(),
            "diameter": float(diameter),
            "vascularization_density_target": float(vascularization_density),
            "vascularization_density_achieved": float(achieved_density),
            "domain_volume": float(bounds.volume),
            "target_vessel_volume": float(target_vessel_volume),
            "achieved_vessel_volume": float(achieved_vessel_volume),
            "target_total_length": float(target_total_length),
            "achieved_total_length": float(total_length),
            "resolution": float(resolution),
            "branching_probability": float(branching_probability),
            "nucleation_faces": [int(v) for v in nucleation_faces],
            "nucleation_face_names": FACE_NAMES,
            "nucleation_points_per_face": int(nucleation_points_per_face),
            "min_connection_distance": float(min_connection_distance),
            "direction_persistence": float(direction_persistence),
            "initial_inward_bias": float(initial_inward_bias),
            "allow_self_connections": bool(allow_self_connections),
            "seed": seed,
            "seed_batches_used": int(seed_batches),
            "terminated_because": "target_density_reached" if total_length >= target_total_length else "safety_limit_or_no_active_tips",
        },
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in edges],
    }

    if save_vtk:
        save_network_vtk(_result, "vasc_network.vtk")

    return _result


def save_network_vtk(network: Dict[str, object], output_path: str | Path) -> Path:
    """Save a generated network in VTK unstructured grid (ASCII) format."""
    nodes = network["nodes"]
    edges = network["edges"]
    n_points = len(nodes)
    n_cells = len(edges)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Vascular network data\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_points} float\n")
        for node in nodes:
            f.write(f"{node['x']} {node['y']} {node['z']}\n")
        f.write(f"CELLS {n_cells} {n_cells * 3}\n")
        for edge in edges:
            f.write(f"2 {edge['parent_id']} {edge['child_id']}\n")
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("3\n")
        f.write(f"POINT_DATA {n_points}\n")
        f.write("SCALARS node_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for node in nodes:
            f.write(f"{node['id']}\n")

    return output_path


def save_network(network: Dict[str, object], output_path: str | Path) -> Path:
    """Save a generated network to a pickle file."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def load_network(pickle_path: str | Path) -> Dict[str, object]:
    """Load a generated vascular network pickle file."""
    with Path(pickle_path).open("rb") as f:
        return pickle.load(f)


def network_to_arrays(network: Dict[str, object]) -> Dict[str, np.ndarray]:
    """
    Convert the pickle dictionary to NumPy arrays.

    This is useful when converting the network into FLAME GPU agent initial states.
    children are kept as an object array because a node can have a variable number
    of outgoing connections.
    """
    nodes = network["nodes"]
    edges = network["edges"]

    positions = np.array([[n["x"], n["y"], n["z"]] for n in nodes], dtype=float)
    parent_ids = np.array([list(n["parent_ids"]) for n in nodes], dtype=object)
    tree_ids = np.array([n["tree_id"] for n in nodes], dtype=np.int64)
    is_boundary = np.array([n["is_boundary"] for n in nodes], dtype=bool)
    boundary_faces = np.array([n["boundary_face"] for n in nodes], dtype=np.int64)
    children = np.array([list(n["children_ids"]) for n in nodes], dtype=object)
    edge_index = np.array([[e["parent_id"], e["child_id"]] for e in edges], dtype=np.int64)
    edge_lengths = np.array([e["length"] for e in edges], dtype=float)

    return {
        "positions": positions,
        "parent_ids": parent_ids,
        "tree_ids": tree_ids,
        "is_boundary": is_boundary,
        "boundary_faces": boundary_faces,
        "children": children,
        "edge_index": edge_index,
        "edge_lengths": edge_lengths,
    }


def plot_network(
    pickle_path: str | Path,
    show_nodes: bool = True,
    show_domain: bool = True,
    line_width: float = 0.8,
    node_size: float = 4.0,
) -> None:
    """Read a network pickle and show a 3D plot for visual inspection."""
    network = load_network(pickle_path)
    arrays = network_to_arrays(network)
    positions = arrays["positions"]
    edge_index = arrays["edge_index"]
    bounds = network["metadata"]["bounds"]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for parent_id, child_id in edge_index:
        p0 = positions[parent_id]
        p1 = positions[child_id]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=line_width)

    if show_nodes and len(positions) > 0:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=node_size)

    if show_domain:
        _plot_domain_box(ax, bounds)

    md = network["metadata"]
    ax.set_title(
        f"Vascular network: {len(network['nodes'])} nodes, {len(network['edges'])} edges, "
        f"density={md['vascularization_density_achieved']:.4g}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((
        bounds["xmax"] - bounds["xmin"],
        bounds["ymax"] - bounds["ymin"],
        bounds["zmax"] - bounds["zmin"],
    ))
    plt.tight_layout()
    plt.show()


def _plot_domain_box(ax, bounds: Dict[str, float]) -> None:
    """Plot cuboidal domain edges."""
    xmin, xmax = bounds["xmin"], bounds["xmax"]
    ymin, ymax = bounds["ymin"], bounds["ymax"]
    zmin, zmax = bounds["zmin"], bounds["zmax"]

    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=float,
    )
    box_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in box_edges:
        p0 = corners[i]
        p1 = corners[j]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linestyle="--", linewidth=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a random vascular network pickle.")
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        required=True,
        metavar=("+x", "-x", "+y", "-y", "+z", "-z"),
        help="Domain boundaries in order [+x, -x, +y, -y, +z, -z].",
    )
    parser.add_argument("--diameter", type=float, required=True, help="Vessel diameter.")
    parser.add_argument(
        "--density",
        type=float,
        required=True,
        help="Target vascularization density: vessel volume / domain volume.",
    )
    parser.add_argument("--resolution", type=float, required=True, help="Distance between consecutive vessel nodes.")
    parser.add_argument(
        "--branching-probability",
        type=float,
        required=True,
        help="Probability that a growing tip branches into two children.",
    )
    parser.add_argument(
        "--nucleation-faces",
        nargs=6,
        type=int,
        required=True,
        metavar=("+x", "-x", "+y", "-y", "+z", "-z"),
        help="Enable nucleation faces with 1/0 in order [+x, -x, +y, -y, +z, -z].",
    )
    parser.add_argument(
        "--nucleation-per-face",
        type=int,
        default=8,
        help="Number of seed points per enabled face and seed batch.",
    )
    parser.add_argument(
        "--min-connection-distance",
        type=float,
        default=None,
        help="Distance below which a tip connects to another vessel. Default: 0.75 * resolution.",
    )
    parser.add_argument(
        "--direction-persistence",
        type=float,
        default=0.85,
        help="Higher values create straighter vessels. Lower values increase tortuosity.",
    )
    parser.add_argument(
        "--initial-inward-bias",
        type=float,
        default=0.80,
        help="Higher values make seed segments point more strongly into the domain.",
    )
    parser.add_argument(
        "--allow-self-connections",
        action="store_true",
        help="Allow tips to connect to nodes from the same seeded tree.",
    )
    parser.add_argument("--max-nodes", type=int, default=1_000_000, help="Safety limit for node count.")
    parser.add_argument("--max-seed-batches", type=int, default=100, help="Safety limit for reseeding batches.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output", type=str, default="vascular_network.pickle", help="Output pickle path.")
    parser.add_argument("--plot", action="store_true", help="Plot the generated network after saving.")
    parser.add_argument("--save-vtk", action="store_true", help="Save the network as vasc_network.vtk in VTK unstructured grid format.")
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        args = argparse.Namespace(
            bounds=[50, -50, 50, -50, 50, -50],
            diameter=1.0,
            density=0.002,
            resolution=4.0,
            branching_probability=0.08,
            nucleation_faces=[1, 1, 1, 1, 1, 1],
            nucleation_per_face=8,
            min_connection_distance=None,
            direction_persistence=0.65,
            initial_inward_bias=0.80,
            allow_self_connections=False,
            max_nodes=1_000_000,
            max_seed_batches=100,
            seed=1,
            output="vascular_network.pickle",
            plot=True,
            save_vtk=True,
        )
    else:
        args = parse_args()
    network = generate_vascular_network(
        bounds_plus_minus=args.bounds,
        diameter=args.diameter,
        vascularization_density=args.density,
        resolution=args.resolution,
        branching_probability=args.branching_probability,
        nucleation_faces=args.nucleation_faces,
        nucleation_points_per_face=args.nucleation_per_face,
        min_connection_distance=args.min_connection_distance,
        direction_persistence=args.direction_persistence,
        initial_inward_bias=args.initial_inward_bias,
        allow_self_connections=args.allow_self_connections,
        max_nodes=args.max_nodes,
        max_seed_batches=args.max_seed_batches,
        seed=args.seed,
        save_vtk=args.save_vtk,
    )
    output_path = save_network(network, args.output)

    md = network["metadata"]
    print(f"Saved network: {output_path}")
    print(f"Nodes: {len(network['nodes'])}")
    print(f"Edges: {len(network['edges'])}")
    print(f"Target density:   {md['vascularization_density_target']:.6g}")
    print(f"Achieved density: {md['vascularization_density_achieved']:.6g}")
    print(f"Termination: {md['terminated_because']}")

    if args.plot:
        plot_network(output_path)


if __name__ == "__main__":
    main()
