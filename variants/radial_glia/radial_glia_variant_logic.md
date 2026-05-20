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
  away from the substrate
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
               + k_autocrine * local_sp2
               + k_paracrine * f_RG_neighbours ]
             * (1 - commit)
             + noise
```

where:
- `k_basal` = `RG_COMMIT_RATE` (5e-6 /s) -- slow baseline drift
- `k_autocrine` = `RG_COMMIT_AUTOCRINE_RATE` (3e-5 /s per a.u.) -- morphogen the
  cell senses from the ECM; amplifies commitment in cells already sitting in a
  morphogen-rich region
- `k_paracrine` = `RG_COMMIT_PARACRINE_RATE` (2e-5 /s per fraction) -- fraction
  of spatially neighbouring cells that are already RG; local community effect
- `(1 - commit)` -- logistic saturation: a nearly-committed cell cannot go above 1
- `noise` = `RG_COMMIT_NOISE * N(0,1) * sqrt(dt)` -- Ito noise, spreads the
  population timing so cells do not all cross each threshold at exactly the same
  step

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
toward the centre of a cluster -- exactly the direction that drives rosette polarity.

---

## Apical Polarity

Each cell carries a unit vector `(apx, apy, apz)`.  It is updated every step in
`cell_rg_polarity_update` as follows:

### 1. Concentration gate

The local sp2 concentration at the cell's voxel is sampled from `C_SP_MACRO[2]`.
If `C0 < RG_POLARITY_SP2_THRESHOLD`, the cell's apical vector is left unchanged.
This restricts polarity updates to cells inside the morphogen field.

### 2. Intrinsic z-bias (NPC and RG only)

After the gradient blend, a small upward bias is added to `apz`:

```
apz += RG_INTRINSIC_APICAL_Z * epithelialization_level
ap = ap / |ap|    (re-normalize)
```

This represents apicobasal self-organisation: well-epithelialized committed cells
orient their apical surface away from the substrate.  The bias is gated on
`cell_type >= 1` -- iPSC cells have no established polarity axis, and giving them
upward apz would cause the apical-bias force in `cell_move` to drag them off the
substrate before they have differentiated.

**Logistic accumulation warning**: the bias equation is logistic in nature:

```
d(apz)/dn ~ bias * epi * (1 - apz^2)
```

Even small values accumulate: with `RG_INTRINSIC_APICAL_Z = 0.5`, vectors reach
apz ~ 1 within a few hundred steps even at low epi.  The value is set to 3e-4 to
give a half-life of roughly 2 days at full epithelialization.

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
| `RG_COMMIT_RATE` | 5e-6 /s | Basal commitment rate |
| `RG_COMMIT_AUTOCRINE_RATE` | 3e-5 /s | sp2-driven autocrine boost |
| `RG_COMMIT_PARACRINE_RATE` | 2e-5 /s | RG-fraction paracrine boost |
| `RG_COMMIT_NOISE` | 1e-3 | Ito noise coefficient; spreads NPC timing by ~+/-3h |
| `RG_INTRINSIC_APICAL_Z` | 3e-4 | Per-step upward z-bias (NPC/RG only) |
| `RG_POLARITY_SP2_THRESHOLD` | 0.1 uM | sp2 concentration gate for z-bias |
| `RG_SUBSTRATE_K` | 5e-5 nN/um | Substrate spring stiffness |
| `RG_ADHESION_MATRIX[RG,RG]` | 1.2 nN/um | Homotypic RG adhesion (max 3.0 with boost) |
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
| L6b | `cell_rg_polarity_update` | Update apical vector from sp2 gradient + intrinsic z-bias |
| L7 | `cell_cell_interaction` | Compute adhesion/repulsion forces -> cc_dvx/y/z |
| L8 | `cell_move` (RG override) | Apply all forces, substrate spring, apical bias, advance |

---

## Typical Dynamics

| Time | Expected behaviour |
|---|---|
| 0-12h | All cells iPSC; monolayer forms disc on substrate; rg_commit rises slowly |
| 12-30h | Stochastic NPC transitions begin (spread ~+/-3h by noise); first sp2 appears |
| 30-48h | NPC clusters form; sp2 gradient nucleates polarity alignment; RG rosettes begin |
| 48-96h | Rosettes mature; RG cells rise slightly off substrate; ring structures stabilise |
| 96-168h | Full rosette morphology; epithelialization_level approaches 1 in RG cells |

---

## Debugging Checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| All cells become NPC at exactly the same step | `RG_COMMIT_NOISE` too small | Increase to ~1e-3 |
| Apical vectors -> (0,0,1) within a few hours | `RG_INTRINSIC_APICAL_Z` too large (logistic blow-up) OR bias applied to iPSC | Reduce to <= 5e-4; check cell_type gate |
| Cells merge into a solid blob | `RG_ADHESION_MATRIX[RG,RG]` or boost too high vs repulsion | Ensure max adhesion < repulsion |
| sp2 never builds up | `CELL_PRODUCTION_MULTIPLIER[1 or 2]` = 0, or `INIT_CELL_PRODUCTION_RATES[2]` = 0 | Check production config |
| Cells pile up at z = +/-150 on step 0 | z-periodic wrap not replaced by z-clamp | Ensure RG cell_move override is loaded |
| Anchor point sphere flattens against substrate | Anchor z updated with raw_dz instead of dz_actual | Use `dz_actual = z_new - z_old` after floor clamp |
| Polarity gradient points downward (-z) | Gradient sampled from boundary voxel k=0 where sp2 accumulates | Shift to k=1 when gk==0 (already implemented) |
| All cells RG within 1h of first NPC appearing | Synchronous NPC onset -> uniform sp2 -> simultaneous autocrine cascade | Fix noise magnitude first; optionally reduce autocrine rate |

---

## File Map

| File | Role |
|---|---|
| `variants/radial_glia/__init__.py` | PARAMS, FILES list, configure_globals (env properties), configure_layers |
| `variants/radial_glia/cell_rg_differentiation.cpp` | Commitment ODE, cell_type switching, epithelialization update |
| `variants/radial_glia/cell_rg_polarity_update.cpp` | Apical vector update (gradient blend + intrinsic z-bias) |
| `variants/radial_glia/cell_move.cpp` | RG override: substrate spring, apical bias force, anchor point update |
| `variants/radial_glia/cell_cell_interaction.cpp` | RG override: type-pair adhesion matrix + epithelialization boost |
| `cell_ecm_interaction_metabolism.cpp` (root) | sp2 secretion into nearest ECM voxel |
| `ecm_ecm_interaction.cpp` (root) | sp2 diffusion; zero-flux ghost cells for Neumann boundaries |
| `ecm_boundary_concentration_conditions.cpp` (root) | Dirichlet pinning (value >= 0) / zero-flux marker (-1) |
