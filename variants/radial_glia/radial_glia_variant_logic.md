# Radial Glia Variant - Model Logic

## Goal

This variant simulates the spontaneous self-organisation of a pluripotent stem cell
monolayer into a **radial glia (RG) rosette** -- a hallmark structure of early human
cortical development.  In vivo, iPSCs transition through a neural progenitor
intermediate (NEP) and eventually adopt a radial glia identity.  RG cells form
distinctive wheel-spoke arrangements called rosettes, with apical surfaces facing a
central lumen and basal processes extending toward the substrate.

The model captures:
- Stochastic, asynchronous iPSC -> NEP -> RG fate commitment driven by a secreted
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
| 1 | NEP | Neuroepithelial progenitor; N-cadherin expression starts, weak sp2 secretion |
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
NEP threshold (x_eq = 0.35) ->  sp2 > 0.027 µM   (cells within diffusion length)
Periphery (sp2 = 0)         ->  x_eq = 0.0        (stays iPSC)
```
With Notch-Delta inhibition active (1 mature RG neighbour, delta_signal ~ 0.09):
```
effective_decay = 1.5e-6 + 8e-6 * 0.09 = 2.22e-6
x_eq at sp2=0.15 µM = 4.5e-6 / 6.72e-6 = 0.67  -> NEP  (ring edge suppressed)
x_eq at sp2=0.20 µM = 6.0e-6 / 8.22e-6 = 0.73  -> RG   (ring core commits)
```
This sp2-threshold-based pattern formation with lateral inhibition produces
isolated RG islands (radius ~ sp2 > 0.117 µM) surrounded by NEP, with the
rosette ring size limited by how quickly the Delta/apical signal builds up.

### Thresholds

```
commit >= 0.70  ->  cell_type = 2  (RG)
commit >= 0.35  ->  cell_type = 1  (NEP)
commit <  0.35  ->  cell_type = 0  (iPSC)
```

When the type changes, `k_production[2]` (sp2 secretion rate) is updated:
- iPSC  : 0.0 × base  (silent)
- NEP   : 0.1 × base  (low background; see Signal-to-Noise section below)
- RG    : 1.0 × base  (full secretion)

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
- **Diffuses** through the ECM grid (D = 0.03 µm²/s for sp2; D = 0.1 µm²/s for sp0/sp1)
- **Degraded** with first-order rate λ = 4×10⁻⁵ s⁻¹ (half-life ~4.8 h)
- **Diffusion length** L = sqrt(D/λ) = sqrt(0.03/4e-5) ≈ **27 µm** (~2.7 cell radii)
- **Boundary conditions** (sp2):
  - All six faces: zero-flux (Neumann, value = -1 sentinel)
  - With degradation the steady-state profile is set purely by local production vs.
    degradation balance; no boundary sink is needed.

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

## Simulation Parameters (current defaults)

| Parameter | Value | Meaning |
|---|---|---|
| `TIME_STEP` | 60 s | Integration step size |
| `STEPS` | 7200 | 5 days (120 h) |
| `N_CELLS` | 300 | Initial cell count (~10,000 cells/cm² on 0.01 cm² domain) |
| `MONOLAYER_Z` | -15 µm | Starting z of seeded cells above substrate floor |
| `PERIODIC_BOUNDARIES_FOR_CELLS` | True | Cells wrap in x/y; z is hard-clamped at boundaries |
| `RG_COMMIT_RATE` | 5e-6 /s | Basal iPSC→NEP drive (community-gated) |
| `RG_COMMIT_AUTOCRINE_RATE` | 2.5e-5 /s/µM | sp2-driven amplification; x_eq=0.85 at sp2=0.23 µM |
| `RG_COMMIT_DECAY_RATE` | 1.0e-6 /s | First-order decay keeping uncommitted cells below threshold |
| `RG_COMMIT_INHIBIT_RATE` | 2e-5 /s | Notch-Delta lateral inhibition suppressing neighbours |
| `RG_COMMIT_NOISE` | 7e-5 | Ito noise on commitment ODE; spreads first-NEP timing |
| `RG_COMMIT_THRESHOLD_NEP` | 0.35 | commit ≥ 0.35 → cell_type = NEP |
| `RG_COMMIT_THRESHOLD_RG` | 0.67 | commit ≥ 0.67 → cell_type = RG |
| `RG_SYMMETRIC_DIVISION_PROB` | 0.25 | Fraction of RG divisions where both daughters stay RG |
| `DIFFUSION_COEFF_MULTI[2]` | 0.03 µm²/s | sp2 diffusion; L = 27 µm (~2.7 cell radii) |
| `ECM_DEGRADATION_RATE_MULTI[2]` | 4e-5 /s | sp2 half-life 4.8 h |
| `CELL_PRODUCTION_MULTIPLIER[1]` | 0.1 | NEP produces 10% of RG rate (low background; see SNR section) |
| `CELL_PRODUCTION_MULTIPLIER[2]` | 1.0 | RG produces at full base rate |
| `INIT_CELL_PRODUCTION_RATES[2]` | 5e-4 /s | Base sp2 secretion rate for fully committed RG |
| `INIT_ECM_SAT_CONCENTRATION_VALS[2]` | 0.5 µM | Maximum local ECM sp2 concentration a cell drives toward |
| `RG_INTRINSIC_APICAL_Z` | 2e-3 | Per-step z-bias blend; RG half-life ~8 h at commit=0.7 |
| `RG_POLARITY_SP2_THRESHOLD` | 0.1 µM | sp2 gate before polarity update fires |
| `RG_LUMEN_BIAS_STRENGTH` | 4e-3 | Per-step XY blend toward local RG centroid |
| `RG_LUMEN_SEARCH_RADIUS` | 84 µm | Radius for collecting RG neighbours for lumen proxy |
| `RG_LUMEN_MIN_NEIGHBOURS` | 2 | Minimum RG neighbours required to activate lumen cue |
| `RG_SUBSTRATE_K` | 1e-4 /s | Substrate spring stiffness |
| `RG_ADHESION_MATRIX[RG,RG]` | 1.5 nN/µm | Homotypic RG adhesion (max 3.75 with epi boost) |
| `RG_EPITHELIAL_ADHESION_BOOST` | 2.5 | Adhesion multiplier at full epithelialization |
| `MIN_ROSETTE_SIZE` | 5 | Minimum RG cells in a DBSCAN cluster to count as a genuine rosette |

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
| 12-42h | Stochastic NEP transitions begin (spread by noise); first sp2 appears from NEP cells |
| 42-82h | NEP clusters form; sp2 gradient builds; first RG cells commit where local sp2 exceeds threshold |
| 82-96h | Notch-Delta inhibition activates as |apz| rises (~8h after first RG); rosette ring forms |
| 96-120h | Ring stabilises at target 5-15 cells; surrounding NEPs inhibited; second independent cluster may nucleate |

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
| `optimizer/optuna_config_radial_glia.yaml` | NSGA-II search space and objective definitions |
| `optimizer/objectives.py` | Objective function implementations |
| `optimizer/reference_data/` | Target CSV files for each metric |
| `postprocessing/` | DBSCAN clustering analysis and plot scripts |

---

## sp2 Signal-to-Noise Analysis

This section explains why the choice of diffusion coefficient and NEP production
multiplier is non-obvious, and why getting them wrong causes the entire monolayer
to commit to RG rather than forming isolated rosettes.

### The biological goal

For rosette patterning to work, the sp2 morphogen must:
1. Reach **high local concentration** immediately around an RG cell (driving
   autocrine commitment and pulling in nearby cells)
2. Stay **low in the surrounding NEP sea** (so cells 2+ diffusion lengths away
   cannot reach the RG commitment threshold on their own)

Without this spatial contrast, the commitment cascade spreads everywhere and
the entire monolayer becomes RG in one wave — no pattern forms.

### The well-mixed background from NEP

Consider the moment the first RG cells are just appearing (~400 cells total,
~60 RG, ~340 NEP, domain area A = 1000 × 1000 = 10⁶ µm²).

Each NEP cell secretes sp2 at rate:
```
q_NEP = m_NEP * q_RG     [molecules/s, or µM·µm³/s]
```
where `m_NEP = CELL_PRODUCTION_MULTIPLIER[1]` and `q_RG = INIT_CELL_PRODUCTION_RATES[2]`.

In the quasi-2D cell layer (thickness ~ cell diameter), the steady-state
background concentration can be derived from the diffusion–degradation balance.
For a uniform source density (well-mixed limit, valid when the diffusion length
L is much larger than inter-cell spacing), the background is **independent of D**:

```
C_bg = (N_NEP * m_NEP * q_RG) / (A * λ)
```

where λ = `ECM_DEGRADATION_RATE_MULTI[2]` (the first-order degradation rate, 1/s)
and A is the domain area in consistent units.

**Why is C_bg independent of D?**
In steady state, every molecule produced anywhere must eventually degrade (there
is no escape for a Neumann-boundary domain). The steady-state total number of
molecules is fixed by production/degradation balance, and the mean concentration
is therefore set by how much is produced vs. degraded — not by how quickly it
diffuses.

More precisely: ∂C/∂t = D∇²C − λC + S. At steady state, integrating over the
domain, the ∇²C term vanishes (zero-flux boundaries) and you get
⟨C⟩ = ⟨S⟩/λ regardless of D.

### The local signal from one RG cell

A single RG cell secreting at rate q_RG produces a local steady-state profile
described by the 2D modified Bessel function:

```
C_local(r) = q_RG / (2π D) * K₀(r / L)
```

where L = √(D/λ) is the diffusion length and K₀ is the zeroth-order modified
Bessel function of the second kind.

At the cell surface (r = R_cell ≈ 10 µm):
```
C_local(R) = q_RG / (2π D) * K₀(R/L)
```

**Unlike C_bg, C_local does depend on D.** Lower D gives a taller, narrower
peak at the source: the same number of molecules pile up closer to the cell.

### Signal-to-noise ratio

Let SNR = C_local(R) / C_bg. We want SNR > 1 for spatial patterning.

```
SNR = [q_RG / (2π D) * K₀(R/L)] / [(N_NEP * m_NEP * q_RG) / (A * λ)]
    = [A * λ * K₀(R/L)] / [2π D * N_NEP * m_NEP]
```

The q_RG cancels — the SNR does not depend on how strongly RG cells secrete,
only on the ratio of local concentration (set by D and L) to background
(set by m_NEP and total NEP number).

### Old vs new parameter comparison

| Quantity | Old (D=0.1, m_NEP=0.5) | New (D=0.03, m_NEP=0.1) |
|---|---|---|
| L (diffusion length) | √(0.1/4e-5) = 50 µm | √(0.03/4e-5) = 27 µm |
| R/L | 10/50 = 0.20 | 10/27 = 0.37 |
| K₀(R/L) | K₀(0.20) ≈ 1.75 | K₀(0.37) ≈ 1.14 |
| C_local(R) / q_RG | 1.75 / (2π×0.1) ≈ 2.8 | 1.14 / (2π×0.03) ≈ 6.0 |
| C_bg / q_RG (N_NEP=400) | 400×0.5×q/(10⁶×4e-5) ≈ 5.0 | 400×0.1×q/(10⁶×4e-5) ≈ 1.0 |
| **SNR** | **0.56** (background WINS) | **6.0** (local WINS) |

With the old parameters, every NEP cell was exposed to more sp2 from the
distributed background than from any individual RG cell — so the threshold
was crossed everywhere simultaneously. No spatial pattern could form.

With the new parameters, only cells within ~15 µm of an RG cell (≈ 1 cell
diameter) see a local concentration above the commitment threshold, while the
broader population sits comfortably below it.

### What the two parameters independently control

| Parameter | Controls | Effect of increasing |
|---|---|---|
| D | C_local ONLY (C_bg invariant) | Lower peak → worse SNR |
| m_NEP | C_bg ONLY (C_local invariant) | Higher background → worse SNR |
| λ | Both (L shrinks, C_bg drops) | Higher degradation → smaller range but better SNR |
| q_RG | Both equally (SNR invariant) | More signal but proportionally more background too |

This means you cannot fix a bad SNR by tweaking q_RG alone; you must change
D (lower it), m_NEP (lower it), or λ (raise it). The chosen fix uses both
D=0.03 (local signal ×2.1) and m_NEP=0.1 (background ÷5), giving a combined
improvement of ~11×.

### K₀ table for reference

| R/L | K₀(R/L) | C_local/q per µm²/s of D |
|---|---|---|
| 0.10 | 2.43 | 2.43/(2πD) |
| 0.20 | 1.75 | 1.75/(2πD) |
| 0.37 | 1.14 | 1.14/(2πD) |
| 0.50 | 0.92 | 0.92/(2πD) |
| 1.00 | 0.42 | 0.42/(2πD) |

K₀(x) → ∞ as x → 0 (point source divergence), so using a cell radius as the
minimum r avoids the singularity. For x > 2, K₀ decays exponentially as
K₀(x) ≈ √(π/2x) e^(-x).

---

## Numerical Stability of the ECM Diffusion Solver

The ECM species concentrations are integrated with a standard forward-Euler
(FTCS) explicit finite-difference scheme. The grid has spacing `dx = dy = dz`
in µm. Stability requires the **Fourier number** (dimensionless CFL):

```
F = D * dt / dx²  ≤  0.5 / (2*N_dim)
```

In 3D (N_dim = 3) the limit is 1/6 ≈ 0.167.

| Species | D (µm²/s) | dt (s) | dx (µm) | F | Stable? |
|---|---|---|---|---|---|
| sp0 (O2 proxy) | 0.1 | 60 | 10 | 0.06 | ✓ |
| sp1 (nutrient) | 0.1 | 60 | 10 | 0.06 | ✓ |
| sp2 (RG signal) | 0.03 | 60 | 10 | 0.018 | ✓ |

The old sp2 D=0.1 gave F=0.06, also stable, but with weaker local signal.
Reducing D to 0.03 improves SNR while still leaving a comfortable safety
margin (F < limit by 9×).

**If you want to increase D for any reason**, the limit is:
```
D_max = 0.5/6 * dx² / dt = 0.0833 * 100 / 60 ≈ 0.139 µm²/s
```
Do not exceed D ≈ 0.13 µm²/s without either reducing dt or coarsening dx.

---

## Rosette Cluster Metrics

The step callback in `model.py` analyses RG cell positions every `ROSETTE_SAMPLE_EVERY`
steps using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

### DBSCAN settings

```python
from sklearn.cluster import DBSCAN
eps = 3.0 * CELL_RADIUS[2]     # 30 µm  — contact distance * 3
min_samples = 2                 # 2 cells touching counts as a cluster seed
```

DBSCAN assigns each point to either a cluster (label ≥ 0) or noise (label = -1).
With `min_samples=2`, **any two RG cells within 30 µm of each other will form a
cluster** — including isolated doublets that are just two cells touching after
division. This produces a large number of tiny "clusters" even when no organised
rosette has formed.

### Why the raw cluster count is misleading

A typical run at T=48h might show:
- 80 RG cells
- 35 DBSCAN clusters (mostly doublets and triplets)

But this is not 35 rosettes — it is 35 noise events. A real rosette needs at
least 5–8 cells arranged in a ring so that the apical vectors have a coherent
inward target. The raw `n_rg_clusters` metric therefore has no biological
meaning at early timepoints.

### Metrics tracked in RG_ROSETTE_METRICS_OVER_TIME

| Column | Definition | Biological meaning |
|---|---|---|
| `n_alive_rg` | Count of living cells with cell_type == 2 | Total RG population |
| `n_rg_clusters` | DBSCAN cluster count (min_samples=2) | Raw cluster count including doublets |
| `mean_rg_cluster_size` | Mean cells per cluster | Low → many tiny clusters |
| `n_large_rg_clusters` | Clusters with ≥ MIN_ROSETTE_SIZE cells | Genuine rosette candidates |
| `large_cluster_fraction` | Fraction of RG cells in large clusters | How well organised the population is |
| `large_cluster_mean_size` | Mean size of large clusters only | Typical rosette ring size |
| `mean_rosette_maturity` | Mean `rosette_maturity` of all RG cells | Per-cell apical alignment quality |
| `mean_epithelialization` | Mean `epithelialization_level` of RG cells | Junction maturity |
| `mean_rg_commit` | Mean `rg_commit_level` of RG cells | Commitment depth |

`MIN_ROSETTE_SIZE = 5` (configurable in `variants/radial_glia/__init__.py`)
means a genuine rosette must have at least 5 RG cells in contact — this is
consistent with the smallest ring structures visible in cortical organoid imaging.

### Rosette maturity (per cell)

Inside `cell_rg_differentiation.cpp`, each RG cell's `rosette_maturity` is
updated as a running mean of how many nearby cells share a similar apical
direction:

```cpp
float cos_sim = apx*nbr_apx + apy*nbr_apy + apz*nbr_apz;  // dot product
if (cos_sim > COS_THRESHOLD) ++aligned_count;
rosette_maturity = aligned_count / (float)(total_rg_nbrs + 1);
```

A fully aligned rosette ring has cells whose apical vectors point inward toward
the same centre, so they all have a large cos_sim with each other's vectors
(all vectors converge). A value near 1.0 means the cell is surrounded by
well-aligned neighbours; near 0 means the surrounding cells are randomly
oriented.

The `mean_rosette_maturity` at T=final is the primary indicator of whether the
simulation produced well-formed rosettes, and is used as one of the two
optimizer objectives.

---

## Optimizer Setup and Objective Functions

The optimizer (`optimizer/optimize.py`) uses **Optuna with the NSGA-II sampler**
(Non-dominated Sorting Genetic Algorithm II) to find parameter combinations that
minimise two objectives simultaneously.

### Why two objectives?

A single objective like "get N rosettes" can be satisfied by degenerate solutions:
- Many small clusters → high n_large_rg_clusters but low maturity (no real rosettes)
- One huge blob → high n_alive_rg but all cells touching → passes size filter

Two objectives make it harder to game:
- **Objective 1**: `|n_large_rg_clusters − target_count|`  (target = 2)
- **Objective 2**: `|mean_rosette_maturity − target_maturity|`  (target = 0.7)

NSGA-II returns a Pareto front of solutions where improving one objective would
worsen the other. The user then picks a point on the front that balances the two.

### Configuration file

`optimizer/optuna_config_radial_glia.yaml` defines:
- `n_trials`: number of Optuna trials per run (300+ recommended; see below)
- `objectives`: list of objective specifications with `metric_name`, `reference_csv`
- `search_params`: list of parameters with type, bounds, log-scale flag

### Reference CSVs

Each objective reads a target value from a CSV file in
`optimizer/reference_data/`. The CSV must have exactly the column named in
`metric_name` and exactly one data row (the target). No `time` column is needed:

```
n_large_rg_clusters
2
```

A common mistake is to accidentally include a `time` column header:
```
time,n_large_rg_clusters    ← WRONG: pandas will parse "2" as the time value,
2                               leaving n_large_rg_clusters = NaN
```

The objectives code now guards against NaN target values with an early error.

### Search parameters (13 total)

| Parameter | Log scale | Range | Rationale |
|---|---|---|---|
| `RG_COMMIT_RATE` | yes | [1e-6, 2e-5] | Controls how long cells wait before first NEP transition |
| `RG_COMMIT_DECAY_RATE` | yes | [1e-6, 1e-5] | Controls how sharp the morphogen threshold is |
| `RG_COMMIT_AUTOCRINE_RATE` | yes | [5e-6, 5e-4] | sp2 sensitivity; widened because lower NEP background means less autocrine drive needed |
| `RG_COMMIT_INHIBIT_RATE` | yes | [2e-6, 1e-4] | Lateral inhibition strength controlling rosette ring size |
| `RG_SYMMETRIC_DIVISION_PROB` | no | [0.1, 0.5] | Fraction of RG daughters staying RG; affects population growth rate |
| `RG_EPITHELIAL_RATE` | yes | [5e-7, 1e-5] | How fast junctions mature after RG commitment; gates inhibition speed |
| `RG_APICAL_BIAS_RG` | yes | [1e-4, 5e-3] | Z-bias strength; too high → all cells vertical; too low → no rosette relief |
| `RG_ADHESION_MATRIX[8]` | yes | [0.5, 10.0] | RG-RG homotypic adhesion; index 8 in flattened 3×3 matrix |
| `DIFFUSION_COEFF_MULTI[2]` | yes | [2e-3, 0.1] | sp2 D; controls diffusion length and local signal amplitude |
| `CELL_PRODUCTION_MULTIPLIER[1]` | yes | [0.01, 0.5] | NEP production fraction; controls background level and SNR |
| `INIT_CELL_PRODUCTION_RATES[2]` | yes | [1e-4, 2e-3] | Base sp2 secretion by RG cells; sets absolute concentration scale |
| `ECM_DEGRADATION_RATE_MULTI[2]` | yes | [1e-5, 5e-4] | sp2 half-life; shorter → sharper gradients, lower background |

(Two parameters are fixed: `RG_COMMIT_THRESHOLD_NEP=0.35`, `RG_COMMIT_THRESHOLD_RG=0.67`)

### How many trials are needed?

NSGA-II works by evolving a population of solutions. With 13 search parameters:
- The search space has ~10^13 combinations on the log scale
- Each trial is a full simulation (~10 min on H100-NVL for 120h)
- Optuna's NSGA-II needs roughly 10–20 generations × population_size to converge
- A typical `population_size = 20` means 200–400 evaluations for a reasonable
  Pareto front

**100 trials is not enough for 13 parameters.** The optimizer will sample
mostly random points and not have time to hill-climb. Use **300–500 trials**
for meaningful results, or reduce the parameter count by fixing parameters you
have good biological intuition for (e.g., fix D and m_NEP based on the SNR
analysis above, leaving only the kinetic parameters to optimise).

### Objective output format

Each trial prints a line like:

```
Trial 042 | rg_rosette_2d_error[n_large_rg_clusters]=1.000000  ...
                                 ^^^^^^^^^^^^^^^^^^^^
                                 metric name shown in brackets
```

The `[metric_name]` label is added by `_format_objective_label` in `optimize.py`
so you can distinguish multiple objectives of the same function type.

---

## NPC → NEP Rename Note

The cell type previously called "NPC" (neural progenitor cell) was renamed to
"NEP" (neuroepithelial progenitor) in all source files, configs, and output
CSVs. This was done to better match the developmental biology literature, where
neuroepithelial cells are the direct columnar precursor of radial glia and are
distinct from multipotent NPCs at later stages.

If you encounter "NPC" anywhere in the codebase or output files it refers to the
same cell type (type index 1). The rename is purely terminological.

---

## Frequently Asked Questions

**Q: Why does sp2 concentration appear uniform across the entire dish?**

A: Check `CELL_PRODUCTION_MULTIPLIER[1]`. If NEP cells are producing at more
than ~15% of the RG rate, the background from 300–600 NEP cells can swamp the
local signal from a handful of RG cells. Use the SNR formula above to calculate
whether your parameters give SNR > 2. Also check that the boundary conditions
for sp2 are zero-flux (not Dirichlet zero) — Dirichlet boundaries absorb
molecules and create an artificial gradient from walls inward.

**Q: The optimizer returns trial values that are all very similar (clustered
near boundaries of the search space). What went wrong?**

A: Most likely the simulation crashes or returns NaN for many trials, leaving
those trials with undefined objectives. Optuna's NSGA-II then converges on the
few non-NaN trials. Check for:
- Division by zero in the objective function (added NaN guard in objectives.py)
- Reference CSV with wrong column names (no `time` header needed for single-value)
- Simulation completing with 0 RG cells (model failed to differentiate)

**Q: DBSCAN shows 40 clusters but there are clearly only 2 rosettes. Why?**

A: DBSCAN with `min_samples=2` counts every touching pair as a cluster. Use
`n_large_rg_clusters` (requires ≥ MIN_ROSETTE_SIZE = 5 cells) rather than
`n_rg_clusters`. The raw count is only useful for confirming that RG cells are
not completely isolated from each other.

**Q: Rosettes form but they are all on one side of the dish. Is this expected?**

A: Yes, to some degree. The first stochastic NEP→RG transition is a single-cell
random event; it nucleates a rosette at a random location. The second rosette
forms at a distance set by the inhibitory field radius (roughly 2–3 diffusion
lengths ≈ 54–81 µm from the first). With a 1000 µm domain and only 2 target
rosettes, both structures could plausibly sit anywhere in the domain including
the same half.

**Q: Cells' apical vectors all point up (0,0,1) almost immediately. What is wrong?**

A: `RG_INTRINSIC_APICAL_Z` is too large, or the polarity update is not gated on
cell_type ≥ 1. Check that the z-bias `alpha_z = RG_INTRINSIC_APICAL_Z * scale`
is only applied to NEP/RG cells and that `scale` correctly uses `rg_commit` (not
a constant 1.0). With `RG_INTRINSIC_APICAL_Z = 2e-3` and commit=0.7, the
e-folding time is ~710 steps = 12 h — well short of the 96 h simulation.
