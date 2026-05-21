# Radial Glia Variant - Model Logic

## Goal

This variant simulates the spontaneous self-organisation of a pluripotent stem cell
monolayer into a **radial glia (RG) rosette** -- a hallmark structure of early human
cortical development.  In vivo, iPSCs transition through a neural progenitor
intermediate (NPC) and eventually adopt a radial glia identity.  RG cells form
distinctive wheel-spoke arrangements called rosettes, with apical surfaces facing a
central lumen and basal processes extending toward the substrate.

The model captures:
- Stochastic, asynchronous iPSC -> NPC -> RG fate commitment driven by a secreted
  morphogen (sp2)
- Gradual apicobasal polarization, where cells develop an apical vector pointing
  toward a local lumen proxy with an upward component
- Differential adhesion driving RG cells to cluster and form rosette ring structures
- Substrate attachment and apically-directed migration giving the rosette its 3D
  relief

---

## Cell Types

| Integer code | Name | Biological identity |
|---|---|---|
| 0 | iPSC | Induced pluripotent stem cell; not yet committed |
| 1 | NPC | Neural progenitor cell; N-cadherin expression starts, weak sp2 secretion |
| 2 | RG | Radial glia; strong N-cadherin junctions, full sp2 secretion, active apical bias |

---

## Key State Variables (per cell)

| Variable | Range | Meaning |
|---|---|---|
| `rg_commit_level` | [0, 1] | Fate commitment meter; 0 = pure iPSC, 1 = fully committed RG |
| `cell_type` | {0, 1, 2} | Discretised cell type, set from `rg_commit_level` thresholds |
| `epithelialization_level` | [0, 1] | Maturity of epithelial junctions; grows with rg_commit |
| `apx, apy, apz` | unit vec | Apical vector; direction of the cell's apical surface |
| `rosette_maturity` | [0, 1] | Proxy for how well integrated into a rosette the cell is |
| `morphogen_local` | >= 0 | Local sp2 concentration sampled by the cell from the ECM grid |

---

## The Commitment ODE

Each step, `rg_commit_level` evolves according to:

```
d(commit)/dt = [ k_basal
               + k_autocrine * local_sp2 ]
             * (1 - commit)
             - k_decay  * commit
             - k_inhibit * delta_signal * commit
             + noise
```

where:
- `k_basal` = `RG_COMMIT_RATE` (5e-6 /s) -- slow baseline drift, iPSC only
- `k_autocrine` = `RG_COMMIT_AUTOCRINE_RATE` (3e-5 /s per µM) -- morphogen the
  cell senses from the ECM; the sp2 gradient is the primary driver of the
  RG zone.  RG threshold (x_eq=0.70) requires sp2 > 0.117 µM.
- `k_decay` = `RG_COMMIT_DECAY_RATE` (1.5e-6 /s) -- first-order decay; ensures
  cells outside the morphogen field cannot reach the RG threshold.
  Equilibrium without inhibition: x_eq = drive / (drive + k_decay).
- `k_inhibit` = `RG_COMMIT_INHIBIT_RATE` (8e-6 /s) -- Notch-Delta lateral
  inhibition.  Committed RG cells suppress their neighbours' commitment
  proportional to `delta_signal` (see below).
- `delta_signal` = sum over RG neighbours of (commit_nb × |apz_nb|) / (n+1) --
  Delta ligand expression proxy.  `|apz|` serves as junction-maturity gate:
  a newly-committed RG cell (|apz| ~ 0.1) sends weak signal, giving the
  initial rosette ~8h to nucleate before the inhibitory fence goes up.
- `(1 - commit)` -- logistic saturation on the forward term only
- `noise` = `RG_COMMIT_NOISE * N(0,1) * sqrt(dt)` -- Ito noise

### Equilibrium analysis

Equilibrium commit level with no inhibition:
```
x_eq = (k_basal + k_autocrine * sp2) / (k_basal + k_autocrine * sp2 + k_decay)
```
With k_autocrine=3e-5, k_decay=1.5e-6:
```
RG threshold (x_eq = 0.70)  ->  sp2 > 0.117 µM   (cells in morphogen-rich zone)
NPC threshold (x_eq = 0.35) ->  sp2 > 0.027 µM   (cells within diffusion length)
Periphery (sp2 = 0)         ->  x_eq = 0.0        (stays iPSC)
```
With Notch-Delta inhibition active (1 mature RG neighbour, delta_signal ~ 0.09):
```
effective_decay = 1.5e-6 + 8e-6 * 0.09 = 2.22e-6
x_eq at sp2=0.15 µM = 4.5e-6 / 6.72e-6 = 0.67  -> NPC  (ring edge suppressed)
x_eq at sp2=0.20 µM = 6.0e-6 / 8.22e-6 = 0.73  -> RG   (ring core commits)
```
This sp2-threshold-based pattern formation with lateral inhibition produces
isolated RG islands (radius ~ sp2 > 0.117 µM) surrounded by NPC, with the
rosette ring size limited by how quickly the Delta/apical signal builds up.

### Thresholds

```
commit >= 0.70  ->  cell_type = 2  (RG)
commit >= 0.35  ->  cell_type = 1  (NPC)
commit <  0.35  ->  cell_type = 0  (iPSC)
```

When the type changes, `k_production[2]` (sp2 secretion rate) is updated:
- iPSC  : 0.0 x base
- NPC   : 0.5 x base
- RG    : 1.0 x base

Fate commitment is **irreversible** (ratchet): noise cannot push cell_type back
below a threshold once crossed.

### Why the noise magnitude matters

`RG_COMMIT_NOISE` sets the standard deviation of the Ito increment per sqrt(s).
Over N steps of duration dt each:

```
sigma_total = RG_COMMIT_NOISE * sqrt(N * dt)
```

With `RG_COMMIT_NOISE = 1e-3` and 24h = 1440 steps of 60s:

```
sigma_total ~ 1e-3 * sqrt(1440) ~ 0.038
```

This spreads the NPC crossing time (commit = 0.35) by roughly +/- 3h across the
population, enabling spatially nucleated clusters rather than simultaneous
whole-population transitions.

---

## Morphogen sp2 -- The Signalling Species

- **Species index**: 2 (out of N_SPECIES = 3)
- **Produced by**: NPC (half rate) and RG (full rate) cells
- **Secreted into**: the nearest ECM voxel each step (via `cell_ecm_interaction_metabolism`)
- **Diffuses** through the ECM grid (D = 0.1 um^2/s)
- **Boundary conditions**:
  - z_pos (top)  : Dirichlet = 0 (absorbed)
  - x, y walls   : Dirichlet = 0 (absorbed)
  - z_neg (substrate): zero-flux (Neumann) -- sp2 accumulates at the cell layer

The zero-flux substrate boundary means sp2 builds up where cells actually are.
This concentrates the gradient in the plane of the monolayer, making it strongest
toward the centre of a cluster, which helps gate where rosette polarity can activate.

---

## Apical Polarity

Each cell carries a unit vector `(apx, apy, apz)`. It is updated every step in
`cell_rg_polarity_update` using a gated two-component rule:

### 1. Concentration + type gate

The local sp2 concentration at the cell's voxel is sampled from `C_SP_MACRO[2]`.
The polarity rule only runs for NPC/RG cells (`cell_type >= 1`) when
`C0 > RG_POLARITY_SP2_THRESHOLD`. Outside this gate, only small XY noise is added.

### 2. Local lumen cue in XY (RG neighbours only)

The function reads `cell_spatial_location_message` and looks for nearby alive RG
cells (`cell_type == 2`) within `RG_LUMEN_SEARCH_RADIUS`. If at least
`RG_LUMEN_MIN_NEIGHBOURS` are found, their XY centroid is used as a local lumen
proxy and the apical XY component is blended toward that centroid direction:

```
beta_xy = RG_LUMEN_BIAS_STRENGTH * scale
new_ap_xy = (1 - beta_xy) * ap_xy + beta_xy * dir_to_rg_centroid_xy
```

where `scale = rg_commit` for RG cells and
`scale = epithelialization_level * rg_commit * 0.1` for NPC cells.

If not enough RG neighbours are found, XY falls back to suppression only
(no lumen cue).

### 3. Intrinsic z-bias (NPC and RG only)

A small upward bias is also applied each step:

```
alpha_z = RG_INTRINSIC_APICAL_Z * scale

new_apz = (1 - alpha_z) * apz + alpha_z
ap = ap / |ap|    (re-normalize)
```

RG cells use `rg_commit` directly (not `epi`) so alignment starts immediately at
commitment (~0.7). With `RG_INTRINSIC_APICAL_Z = 2e-3` and commit=0.7:
```
alpha_z = 1.4e-3 per step  ->  half-life ~ 495 steps ~ 8 h
```
NPC cells scale by `epi * commit` so peripheral NPC cells (low commit from the
decay term) develop much weaker z-polarity, matching the gradient of
pseudo-stratified polarity in cortical neuroepithelium.

This gives apical vectors a local inward component in XY (toward the lumen proxy)
and an upward component in Z.

**RG orientation override**: in `cell_move`, RG cells' persistent migration
direction is directly set to their apical vector `(apx, apy, apz)` each step,
overwriting rotational diffusion.  This is biologically correct (polarised
epithelial progenitors do not migrate randomly) and avoids the instability where
rotational diffusion noise (~0.1 rad/step) overwhelms the soft blend (~2×10⁻⁴).

---

## Adhesion and Rosette Formation

Cell-cell adhesion is controlled by a 3x3 matrix `RG_ADHESION_MATRIX[self*3 + nb]`
(flattened row-major, [nN/um]):

```
         nb:  iPSC  NPC   RG
self: iPSC  [ 0.4,  0.4,  0.2 ]
      NPC   [ 0.4,  0.8,  0.6 ]
      RG    [ 0.2,  0.6,  1.2 ]   <- max with epi boost = 1.2 * 2.5 = 3.0
```

RG-RG adhesion is further multiplied by
`1 + (RG_EPITHELIAL_ADHESION_BOOST - 1) * min(epi_self, epi_nb)` as junction
maturity grows.  The maximum adhesion (3.0 nN/um) is below repulsion (4.0 nN/um),
so RG cells maintain a small gap rather than merging into a solid blob.

The combination of:
- Strong RG-RG homotypic adhesion (ring formation)
- Apical vector bias toward cluster centre (polarity alignment)
- Apical-directed migration force (cells rise as polarity matures)

...collectively drives the emergence of radial glia rosettes.

---

## Substrate Attachment

Each cell has a spring force pulling it back toward the substrate:

```
F_z += RG_SUBSTRATE_K * (z_rest - z)
z_rest = COORD_BOUNDARY_Z_NEG + RG_SUBSTRATE_Z0
```

A hard floor prevents cells from falling below the substrate:

```
if (z - r) < Z_NEG:  z = Z_NEG + r
```

Anchor points (N_ANCHOR_POINTS = 50) represent substrate attachment sites.  They
move rigidly with the cell centre -- using the actual cell-centre displacement
`dz_actual = z_new - z_old` after the floor clamp, so that the anchor sphere never
flattens against the substrate independently of the cell.

---

## Simulation Parameters (default)

| Parameter | Value | Meaning |
|---|---|---|
| `TIME_STEP` | 60 s | Integration step size |
| `STEPS` | 10080 | 7 days total |
| `N_CELLS` | 240 | Initial cell count |
| `MONOLAYER_Z` | -150 um | Substrate z position (= COORD_BOUNDARY_Z_NEG) |
| `PERIODIC_BOUNDARIES_FOR_CELLS` | True | Cells wrap in x/y; z is clamped |
| `RG_COMMIT_RATE` | 5e-6 /s | Basal commitment rate (iPSC only) |
| `RG_COMMIT_AUTOCRINE_RATE` | 3e-5 /s/µM | sp2-driven drive; RG threshold at sp2 > 0.117 µM |
| `RG_COMMIT_DECAY_RATE` | 1.5e-6 /s | First-order decay; limits RG zone to morphogen field |
| `RG_COMMIT_INHIBIT_RATE` | 8e-6 /s | Notch-Delta lateral inhibition; limits rosette size |
| `RG_COMMIT_NOISE` | 1e-3 | Ito noise coefficient; spreads NPC timing by ~+/-3h |
| `RG_INTRINSIC_APICAL_Z` | 2e-3 | Per-step z-bias alpha; RG half-life ~8h at commit=0.7 |
| `RG_POLARITY_SP2_THRESHOLD` | 0.1 uM | sp2 concentration gate for z-bias |
| `RG_LUMEN_BIAS_STRENGTH` | 3e-3 | Per-step XY blend strength toward local RG centroid |
| `RG_LUMEN_SEARCH_RADIUS` | 35 um | Radius to collect RG neighbours for lumen proxy |
| `RG_LUMEN_MIN_NEIGHBOURS` | 2 | Minimum RG neighbours required to activate lumen cue |
| `RG_SUBSTRATE_K` | 5e-5 nN/um | Substrate spring stiffness |
| `RG_ADHESION_MATRIX[RG,RG]` | 1.8 nN/um | Homotypic RG adhesion (max 4.5 with boost) |
| `RG_EPITHELIAL_ADHESION_BOOST` | 2.5 | Max adhesion multiplier at full epithelialization |

---

## Layer Execution Order

One time step runs the following layers in order:

| Layer | Function | Key output |
|---|---|---|
| L1 | `cell_spatial_location_data` | Each cell broadcasts position, type, apical vector |
| L2 | `ecm_boundary_concentration_conditions` | Pin boundary ECM voxels to Dirichlet values |
| L3 | `cell_ecm_interaction_metabolism` | Cell <-> ECM: O2/nutrient uptake, sp2 secretion |
| L3b | `cell_rg_differentiation` | Update rg_commit ODE, switch cell_type, update k_production |
| L4 | `ecm_Csp_update` | Apply sp2 from cells into ECM macro-property |
| L5 | `ecm_ecm_interaction` | Diffuse all species through ECM (+ ECM mechanical forces) |
| L6 | `ecm_boundary_concentration_conditions` | Enforce boundaries again after diffusion |
| L6b | `cell_rg_polarity_update` | Update apical vector from local RG-centroid lumen cue + intrinsic z-bias |
| L7 | `cell_cell_interaction` | Compute adhesion/repulsion forces -> cc_dvx/y/z |
| L8 | `cell_move` (RG override) | Apply all forces, substrate spring, apical bias, advance |

---

## Typical Dynamics

| Time | Expected behaviour |
|---|---|
| 0-12h | All cells iPSC; monolayer forms disc on substrate; rg_commit rises slowly |
| 12-30h | Stochastic NPC transitions begin (spread ~+/-3h by noise); first sp2 appears |
| 30-48h | NPC clusters form; sp2 gradient builds; first RG cells commit at sp2 > 0.117 µM |
| 48-72h | Notch-Delta inhibition activates as |apz| rises (~8h after first RG); rosette ring forms |
| 72-96h | Ring stabilises at ~8-12 cells; surrounding NPCs inhibited even if in sp2 field |
| 96-168h | Rosettes mature; RG cells migrate apically; epithelialization_level approaches 1 |

---

## Debugging Checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| All cells become NPC at exactly the same step | `RG_COMMIT_NOISE` too small | Increase to ~1e-3 |
| Apical vectors -> (0,0,1) within a few hours | `RG_INTRINSIC_APICAL_Z` too large OR bias applied to iPSC | Reduce to <= 5e-3; check cell_type gate |
| Cells merge into a solid blob | `RG_ADHESION_MATRIX[RG,RG]` or boost too high vs repulsion | Ensure max adhesion < repulsion |
| sp2 never builds up | `CELL_PRODUCTION_MULTIPLIER[1 or 2]` = 0, or `INIT_CELL_PRODUCTION_RATES[2]` = 0 | Check production config |
| Cells pile up at z = +/-150 on step 0 | z-periodic wrap not replaced by z-clamp | Ensure RG cell_move override is loaded |
| Anchor point sphere flattens against substrate | Anchor z updated with raw_dz instead of dz_actual | Use `dz_actual = z_new - z_old` after floor clamp |
| No radial rosette orientation despite high sp2 | Lumen cue inactive due to too few nearby RG cells or too small search radius | Increase `RG_LUMEN_SEARCH_RADIUS` or reduce `RG_LUMEN_MIN_NEIGHBOURS` |
| All cells become RG regardless of sp2 | Positive paracrine cascade (old k_paracrine) still active | Remove from drive; use RG_COMMIT_INHIBIT_RATE with negative sign |
| Orientation vector oscillates wildly each step | Rotational diffusion >> soft apical blend for RG cells | Use direct override: orientation = apical for cell_type==2 |
| RG apical vector still at < 30 deg after 56h | Alpha gated on epi which starts at 0 at commitment | Use rg_commit (not epi) for alpha in RG cells |
| No rosette spatial pattern; entire sp2 zone is RG | k_inhibit too small or |apz| never rises (apical not aligning) | Increase RG_COMMIT_INHIBIT_RATE; check RG_INTRINSIC_APICAL_Z |

---

## File Map

| File | Role |
|---|---|
| `variants/radial_glia/__init__.py` | PARAMS, FILES list, configure_globals (env properties), configure_layers |
| `variants/radial_glia/cell_rg_differentiation.cpp` | Commitment ODE, cell_type switching, epithelialization update |
| `variants/radial_glia/cell_rg_polarity_update.cpp` | Apical vector update (RG-only local centroid lumen cue + intrinsic z-bias) |
| `variants/radial_glia/cell_move.cpp` | RG override: substrate spring, apical bias force, anchor point update |
| `variants/radial_glia/cell_cell_interaction.cpp` | RG override: type-pair adhesion matrix + epithelialization boost |
| `cell_ecm_interaction_metabolism.cpp` (root) | sp2 secretion into nearest ECM voxel |
| `ecm_ecm_interaction.cpp` (root) | sp2 diffusion; zero-flux ghost cells for Neumann boundaries |
| `ecm_boundary_concentration_conditions.cpp` (root) | Dirichlet pinning (value >= 0) / zero-flux marker (-1) |
