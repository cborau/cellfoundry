# Radial Glia Variant — Development Strategy

**Last updated:** 2026-05-19 (rev 2)  
**Target file:** `variants/radial_glia/__init__.py` + supporting `.cpp` files

---

## 1. Goal

Model iPSC differentiation into radial glia (RG) and the spontaneous formation of rosette structures on a 2D vitronectin-coated culture plate.  The simulation uses CellFoundry's `MONOLAYER_ASSAY` mode for initialisation, ECM diffusion (species 2) for an instructive morphogen signal, and two new CELL agent functions that drive fate commitment and apicobasal polarity.

---

## 2. Biological Assumptions

### Cell types  (N_CELL_TYPES = 3, hard-coded in C++)

| `cell_type` index | Identity | Key behaviour |
|---|---|---|
| 0 | iPSC | random motility, low adhesion, undifferentiated |
| 1 | Neural Progenitor Cell (NPC) | slower motility, intermediate N-cad adhesion, weak apical bias |
| 2 | Radial Glia (RG) | low motility, high N-cad adhesion with other RG, substrate-anchored, established apicobasal polarity |

Cell types switch discretely at commitment thresholds (`rg_commitment` crosses 0→1 boundary at 0.35; 1→2 boundary at 0.70).  All cells start as iPSC (cell_type = 0); differentiation is driven entirely by the `cell_rg_differentiation` function.

### Apical pole vector `(apx, apy, apz)` [dimensionless, unit vector]

Each cell carries an apical unit vector.  It is **not** an initial polarity assumption — undifferentiated iPSC on a flat plate have no established apicobasal axis.  Initialisation:

- `apx`, `apy` = random unit-vector components in the xy-plane (random angle θ: `apx = cos θ`, `apy = sin θ`)
- `apz = 0.0`

This reflects zero z-polarity at the start.  Polarity develops during simulation as species 2 builds up and creates a z-biased gradient (cells on the substrate sense a gradient pointing away from the plate).

The apical vector is used **only** to:
1. Read it in `cell_rg_differentiation` for computing `rosette_maturity` (mean alignment with neighbours — a readout, not a force).
2. Apply a weak active velocity bias along `(apx, apy, apz)` in the overridden `cell_move.cpp` for NPC and RG cells.  This is a *migration bias*, not a deformation force; cell shape deformation comes exclusively from cell-cell contact and substrate spring forces.

### Substrate

A flat vitronectin-coated glass surface at z = `COORD_BOUNDARY_Z_NEG`.  A linear spring force pulls each cell toward `z_rest = COORD_BOUNDARY_Z_NEG + RG_SUBSTRATE_Z0` [µm].  The z-floor clamping already in `cell_move.cpp` is kept.

---

## 3. Diffusion Species

Three species are hard-coded (N_SPECIES = 3).  For the RG variant they represent:

| Index | Species | Role | Boundary condition |
|---|---|---|---|
| 0 | Oxygen (O₂) | permissive — keeps cells alive | Fixed `BOUNDARY_CONC_FIXED = 2.5` on all 6 walls [a.u.] |
| 1 | Nutrient (e.g. glucose) | permissive — keeps cells alive | Fixed `BOUNDARY_CONC_FIXED = 2.5` on all 6 walls [a.u.] |
| 2 | RG morphogen (e.g. Wnt/Notch surrogate) | instructive — drives commitment | Fixed `0.0` on all walls; secreted only by RG cells |

Species 0 and 1 are maintained at saturation on all boundaries.  Their diffusion and consumption are active but are permissive — cells are never nutrient-limited in the RG variant.  This avoids confounding signals.

Species 2 is produced by differentiated RG cells (cell_type == 2) and diffuses through the ECM.  Since boundaries are fixed at 0, concentration builds up in regions dense with RG cells and decays away from them, creating a gradient that guides polarity alignment.

### Production rate for species 2

The existing metabolism infrastructure already handles this:

- `k_production[sp]` is a per-cell, per-species agent variable [1/s], initialised as `INIT_CELL_PRODUCTION_RATES[sp] * CELL_PRODUCTION_MULTIPLIER[cell_type]`.
- `CELL_PRODUCTION_MULTIPLIER` is already registered as an environment property (array, length N_CELL_TYPES).

For the RG variant set in `PARAMS`:
```python
INIT_CELL_PRODUCTION_RATES = [0.0, 0.0, 5e-4]   # [1/s]  species 2 rate for a type-2 cell
CELL_PRODUCTION_MULTIPLIER = [0.0, 0.5, 1.0]    # [-]   type 0: silent, type 1: half, type 2: full
```

When `cell_rg_differentiation` switches `cell_type`, it also updates `k_production[2]` directly:
```cpp
// k_production[2] tracks cell_type: read base rate from env, scale by multiplier
const float base_prod = FLAMEGPU->environment.getProperty<float, N_SPECIES>("INIT_CELL_PRODUCTION_RATES_ENV", 2);
const float mult      = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_PRODUCTION_MULTIPLIER", new_cell_type);
k_production[2]       = base_prod * mult;
```

This requires `INIT_CELL_PRODUCTION_RATES` to be registered as an environment property (currently it is only a Python list used during init).  Add it to the env block in `model.py`:
```python
env.newPropertyArrayFloat("INIT_CELL_PRODUCTION_RATES_ENV", INIT_CELL_PRODUCTION_RATES)
```

### Metabolism override

The base `cell_ecm_interaction_metabolism.cpp` uses a PhysiCell-style backward-Euler ODE that already reads `k_production[sp]` and `k_consumption[sp]` from agent variables.  Since `cell_rg_differentiation` keeps `k_production[2]` in sync with cell_type, **the base metabolism function requires no override for species 2**.

#### Why the base linear ODE is insufficient for O2 and nutrient

The base ODE for ECM ↔ cell exchange is:

$$\frac{dC_\text{ecm}}{dt} = \alpha \left( k_\text{prod} \cdot C_\text{sat} - (k_\text{prod} + k_\text{cons}) \cdot C_\text{ecm} \right)$$

The consumption term $k_\text{cons} \cdot C_\text{ecm}$ is **first-order linear** — uptake grows without bound as ECM concentration rises.  For the RG variant the boundaries are fixed at $C = 2.5$ a.u. on all six faces, so ECM concentration is kept close to saturation everywhere.  At saturation, the linear term gives consumption $\propto 2.5 \cdot k_\text{cons}$, which is an over-estimate of actual biological rates that saturate at the cell's enzymatic capacity.  This risks artificially depleting species 0/1 near dense cell clusters even though the experiment is designed to be nutrient-unlimited.

Michaelis-Menten kinetics saturate the consumption rate at a maximum $V_\text{max}$ [a.u./s] independently of how high ECM concentration rises:

$$\text{uptake rate} = \frac{V_\text{max} \cdot C_\text{ecm}}{K_m + C_\text{ecm}}$$

At $C_\text{ecm} \gg K_m$ (boundary condition), uptake ≈ $V_\text{max}$ (constant maximum).  At $C_\text{ecm} \ll K_m$ (severely depleted), uptake ≈ $\frac{V_\text{max}}{K_m} \cdot C_\text{ecm}$ (linear, first-order).

#### Backward-Euler integration of M-M kinetics

The existing solver cannot handle M-M directly without solving a quadratic.  The standard approach used by PhysiCell is to **linearize M-M around the previous time-step concentration** $C^n$:

$$k_\text{eff}(C^n) = \frac{V_\text{max}}{K_m + C^n} \quad \text{[1/s, locally linear rate]}$$

Then substitute $k_\text{cons} \leftarrow k_\text{eff}(C^n)$ into the existing backward-Euler formula.  The original coefficients:

```
c1 = dt * alpha * k_production[i] * C_sp_sat[i]
c2 = 1  + dt * alpha * (k_production[i] + k_consumption[i])
C_ecm_new = (C_ecm_old + c1) / c2
```

become:

```cpp
// Michaelis-Menten effective consumption rate  [1/s]
const float V_max_i = FLAMEGPU->environment.getProperty<float, N_SPECIES>("RG_MM_VMAX", i);
const float K_m_i   = FLAMEGPU->environment.getProperty<float, N_SPECIES>("RG_MM_KM",   i);
const float k_eff   = V_max_i / (K_m_i + C_ecm_old + 1e-12f);    // [1/s]  linearised at C^n

const float c1 = TIME_STEP * alpha * k_production[i] * C_sp_sat[i];
const float c2 = 1.0f + TIME_STEP * alpha * (k_production[i] + k_eff);
const float C_ecm_prop = (C_ecm_old + c1) / c2;
```

This is a **single-line change per species** relative to the base file.  The rest of the mass-conservation clamping, atomic write, and cell-amount update are unchanged.

#### Per-species configuration

| Species | $V_\text{max}$ [a.u./s] | $K_m$ [a.u.] | Note |
|---|---|---|---|
| 0 (O₂) | `RG_MM_VMAX[0]` e.g. 0.05 | `RG_MM_KM[0]` e.g. 0.05 | half-saturation at 2 % of boundary value |
| 1 (Nutrient) | `RG_MM_VMAX[1]` e.g. 0.02 | `RG_MM_KM[1]` e.g. 0.1  | slower turnover |
| 2 (Morphogen) | `RG_MM_VMAX[2]` = 0 | any (irrelevant) | no M-M consumption; production managed by differentiation |

Setting `RG_MM_VMAX[i] = 0` recovers the base behaviour for that species (k_eff = 0 → no M-M term).  This means the override file can replace the base for **all** species uniformly — species 2 simply ignores the M-M block.

The existing `k_consumption` agent variable is **not used** in the override for species 0 and 1 (it can be left at 0 in PARAMS); $V_\text{max}$ and $K_m$ are env properties only.  For species 2, `k_production[2]` is still the operative variable (managed by differentiation).

#### Intracellular reaction block

The base file has a hard-coded intracellular reaction after the ECM exchange loop:

```cpp
C_sp[0] -= TIME_STEP * k_reaction[0] * C_sp[0];  // species 0 consumed inside cell
C_sp[1] += TIME_STEP * k_reaction[1] * C_sp[0];  // species 1 produced accordingly
```

This is also linear (first-order).  For the RG variant, apply M-M inside the cell:

```cpp
// Intracellular M-M consumption of O2 (species 0)
const float V_max_intra = FLAMEGPU->environment.getProperty<float, N_SPECIES>("RG_MM_VMAX_INTRA", 0);
const float K_m_intra   = FLAMEGPU->environment.getProperty<float, N_SPECIES>("RG_MM_KM_INTRA",   0);
const float consume_0   = TIME_STEP * V_max_intra * C_sp[0] / (K_m_intra + C_sp[0] + 1e-12f);
C_sp[0] -= consume_0;
C_sp[1] += TIME_STEP * k_reaction[1] * consume_0;  // nutrient produced proportional to O2 consumed
```

The intracellular term uses separate env properties (`RG_MM_VMAX_INTRA`, `RG_MM_KM_INTRA`) so it can be tuned independently of the ECM-level exchange.

#### New env properties (registered in `configure_layers`)

```python
_env.newPropertyArrayFloat("RG_MM_VMAX",       [0.05, 0.02, 0.0])  # [a.u./s]  ECM-level M-M Vmax per species
_env.newPropertyArrayFloat("RG_MM_KM",          [0.05, 0.10, 1.0])  # [a.u.]    ECM-level M-M Km   per species
_env.newPropertyArrayFloat("RG_MM_VMAX_INTRA", [0.02, 0.0,  0.0])  # [a.u./s]  intracellular Vmax per species
_env.newPropertyArrayFloat("RG_MM_KM_INTRA",   [0.05, 1.0,  1.0])  # [a.u.]    intracellular Km   per species
```

And add the file to `FILES` in `__init__.py`:
```python
"cell_ecm_interaction_metabolism_file": str(_HERE / "cell_ecm_interaction_metabolism.cpp"),
```

#### Implementation decision

This override is **optional for the first test run**.  Skip it initially and set all `k_consumption = 0` in PARAMS to avoid any unwanted depletion of species 0/1.  Introduce the override once differentiation and polarity are working correctly.

---

## 4. Domain and Simulation Parameters (variant defaults)

Keep the default domain (no BOUNDARY_COORDS override needed) and default N = 21 grid:

```
BOUNDARY_COORDS   = ±500 µm (default, cubical)
N                 = 21   → ECM_POPULATION_SIZE = 9261 (21³)
ECM_ECM_EQUILIBRIUM_DISTANCE ≈ 50 µm  (500/10 inter-node spacing)
TIME_STEP         = 10.0  [s]
STEPS             = 3600  (10 simulated hours)
SAVE_EVERY_N      = 60    (save every 10 min simulated)
N_CELLS           = 60
CELL_RADIUS       = [5.0, 5.0, 5.0]  [µm]
MONOLAYER_ASSAY   = True
ORGANOID_ASSAY    = False
INCLUDE_DIFFUSION = True
INCLUDE_CELL_CYCLE = False
INCLUDE_FIBRE_NETWORK = False
INCLUDE_VASCULARIZATION = False
INCLUDE_LUMEN     = False
```

With `MONOLAYER_ASSAY = True`, cells are placed at `z = COORD_BOUNDARY_Z_NEG = -500 µm` with random in-plane orientations, well separated with `min_dist = CELL_RADIUS[0]`.

Species boundary conditions in PARAMS:
```python
BOUNDARY_CONC_FIXED_MULTI  = [[2.5]*6, [2.5]*6, [0.0]*6]   # sp0=O2, sp1=nutrient saturated; sp2=0
BOUNDARY_CONC_INIT_MULTI   = [[2.5]*6, [2.5]*6, [0.0]*6]
INIT_CELL_CONCENTRATION_VALS = [2.5, 2.5, 0.0]   # cells start with sp0/sp1 saturated; sp2=0 [a.u.]
```

---

## 5. Variant-Specific Variable Gating in `model.py`

To avoid polluting the base model with variables irrelevant to non-RG variants, new CELL agent variables and message variables are gated behind a variant-name check.

### Mechanism

At the top of `model.py`, define a module-level list of variants that require the RG state variables:
```python
# List of variant names that require the RG-specific CELL variables and message fields.
# Add your variant name here if it reads/writes any of the rg_* agent variables.
VARIANTS_WITH_RG_VARIABLES = ["radial_glia"]
```

Derived flag (placed after the `_VARIANT_NAME` resolution block):
```python
INCLUDE_RG_VARIABLES = _VARIANT_NAME in VARIANTS_WITH_RG_VARIABLES
```

Use `INCLUDE_RG_VARIABLES` to gate:
- CELL agent variable declarations in model.py
- `cell_spatial_location_message` extra variables
- CELL host-function initialisation of the new variables
- `helper_module.py` VTK writing of the new variables

### New CELL agent variables  (added only when `INCLUDE_RG_VARIABLES`)

Add immediately after the existing `chemokinesis_inhibitory_adapt_state` variable (~line 1588):

```python
if INCLUDE_RG_VARIABLES:
    CELL_agent.newVariableFloat("rg_commitment",           0.0)  # [-]      fate commitment in [0,1]; 0=iPSC, 1=fully RG
    CELL_agent.newVariableFloat("epithelialization_level", 0.0)  # [-]      apicobasal polarity / junction degree in [0,1]
    CELL_agent.newVariableFloat("rosette_maturity",        0.0)  # [-]      readout: mean apical-vector alignment with RG neighbours
    CELL_agent.newVariableFloat("apx",                     0.0)  # [–]      apical unit vector x-component
    CELL_agent.newVariableFloat("apy",                     0.0)  # [–]      apical unit vector y-component
    CELL_agent.newVariableFloat("apz",                     0.0)  # [–]      apical unit vector z-component (0 at init: no polarity)
    CELL_agent.newVariableFloat("local_rg_fraction",       0.0)  # [-]      fraction of spatial neighbours that are RG (cell_type==2)
    CELL_agent.newVariableFloat("local_lumen_signal",      0.0)  # [a.u.]   ECM species-2 concentration at cell's nearest voxel
    CELL_agent.newVariableInt(  "lumen_id",                -1)   # [-]      nearest LUMEN agent id; -1 if none or INCLUDE_LUMEN=False
```

### New `cell_spatial_location_message` variables  (added only when `INCLUDE_RG_VARIABLES`)

Add after the existing `cell_type` variable (~line 1257):
```python
if INCLUDE_RG_VARIABLES:
    CELL_spatial_location_message.newVariableFloat("rg_commitment")       # [-]
    CELL_spatial_location_message.newVariableFloat("epithelialization_level")  # [-]
    CELL_spatial_location_message.newVariableFloat("apx")                 # [–]
    CELL_spatial_location_message.newVariableFloat("apy")                 # [–]
    CELL_spatial_location_message.newVariableFloat("apz")                 # [–]
```

### `MONOLAYER_CELL_TYPE_RATIOS` parameter

Add to the cell parameter block:
```python
MONOLAYER_CELL_TYPE_RATIOS = [1, 0, 0]  # [-]  relative counts per cell type at init; all iPSC for RG variant
```

Update the MONOLAYER_ASSAY init branch in the host function to use it (replacing the hard-coded `[70, 20, 10]`).

### Host-function initialisation additions  (inside the CELL init loop, gated by `INCLUDE_RG_VARIABLES`)

```python
if INCLUDE_RG_VARIABLES:
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    instance.setVariableFloat("rg_commitment",           0.0)
    instance.setVariableFloat("epithelialization_level", 0.0)
    instance.setVariableFloat("rosette_maturity",        0.0)
    instance.setVariableFloat("apx",  float(np.cos(angle)))   # random in-plane apical vector
    instance.setVariableFloat("apy",  float(np.sin(angle)))
    instance.setVariableFloat("apz",  0.0)                    # no z-polarity at init
    instance.setVariableFloat("local_rg_fraction",       0.0)
    instance.setVariableFloat("local_lumen_signal",      0.0)
    instance.setVariableInt(  "lumen_id",                -1)
```

### `INIT_CELL_PRODUCTION_RATES_ENV` environment property

Add to the env registration block:
```python
env.newPropertyArrayFloat("INIT_CELL_PRODUCTION_RATES_ENV", INIT_CELL_PRODUCTION_RATES)
```
(Needed so `cell_rg_differentiation.cpp` can read the base production rate for species 2.)

---

## 6. Variant File Structure

```
variants/radial_glia/
├── __init__.py                        ← PARAMS, FILES, configure_globals, configure_layers
├── cell_spatial_location_data.cpp     ← override: broadcasts 5 new message variables
├── cell_cell_interaction.cpp          ← override: type-pair adhesion matrix + epithelialization boost
├── cell_move.cpp                      ← override: substrate spring + apical migration bias + z-floor
├── cell_rg_differentiation.cpp        ← NEW: rg_commitment ODE + cell_type switching
└── cell_rg_polarity_update.cpp        ← NEW: apical vector update from species-2 gradient (26-neighbour)
```

Optionally (decide during implementation):
```
└── cell_ecm_interaction_metabolism.cpp  ← override: Michaelis-Menten O2/nutrient consumption
```

---

## 7. Environment Properties (registered in `configure_layers`)

| Property | C++ type | Default | Unit | Description |
|---|---|---|---|---|
| `RG_COMMIT_RATE` | float | 2e-5 | 1/s | baseline commitment accumulation rate (autocatalytic floor) |
| `RG_COMMIT_AUTOCRINE_RATE` | float | 5e-4 | 1/(s·a.u.) | rate scaling with local species-2 concentration |
| `RG_COMMIT_PARACRINE_RATE` | float | 1e-4 | 1/s | rate scaling with local RG neighbour fraction |
| `RG_COMMIT_THRESHOLD_NPC` | float | 0.35 | – | rg_commitment value at which iPSC → NPC switch fires |
| `RG_COMMIT_THRESHOLD_RG` | float | 0.70 | – | rg_commitment value at which NPC → RG switch fires |
| `RG_EPITHELIAL_RATE` | float | 1e-4 | 1/s | epithelialization growth rate (logistic, gated by rg_commitment) |
| `RG_POLARITY_TAU` | float | 600.0 | s | time constant for exponential apical-vector blending |
| `RG_POLARITY_GRADIENT_THRESHOLD` | float | 1e-6 | a.u./µm | minimum species-2 gradient magnitude to update polarity |
| `RG_SUBSTRATE_K` | float | 0.5 | nN/µm | spring stiffness pulling cell toward substrate plane |
| `RG_SUBSTRATE_Z0` | float | 0.0 | µm | resting offset above z_neg boundary (0 = on substrate) |
| `RG_APICAL_BIAS_RG` | float | 0.002 | µm/s | apical-direction velocity bias for RG cells (cell_type == 2) |
| `RG_APICAL_BIAS_NPC` | float | 0.0005 | µm/s | apical-direction velocity bias for NPC cells (cell_type == 1) |
| `RG_ADHESION_MATRIX` | float[9] | see below | nN/µm | per-pair adhesion stiffness (3×3, row=self type, col=neighbour type) |
| `RG_REPULSION_MATRIX` | float[9] | see below | nN/µm | per-pair repulsion stiffness (uniform volume-exclusion) |
| `RG_EPITHELIAL_ADHESION_BOOST` | float | 3.0 | – | RG-RG adhesion multiplier at full epithelialization_level |

Default adhesion matrix (nN/µm), indexed `[self_type * N_CELL_TYPES + neighbour_type]`:
```
         iPSC   NPC    RG
iPSC  [  0.4,   0.4,   0.2 ]
NPC   [  0.4,   0.8,   0.6 ]
RG    [  0.2,   0.6,   2.0 ]   ← strong N-cadherin between RG cells
```

Default repulsion matrix (uniform volume-exclusion, all pairs equal):
```
all  [  4.0,   4.0,   4.0 ]
```

---

## 8. New Agent Function: `cell_rg_differentiation.cpp`

**Layer:** `L3b_RG_Differentiation` — after L3 (metabolism) so fresh `C_sp` values are available; before L4 (ECM update) so updated `k_production[2]` is used in the same step's diffusion solve.

**Message input:** `cell_spatial_location_message` (same spatial message broadcast in L1; read-only).

**Reads from env:** `RG_COMMIT_RATE`, `RG_COMMIT_AUTOCRINE_RATE`, `RG_COMMIT_PARACRINE_RATE`, `RG_COMMIT_THRESHOLD_NPC`, `RG_COMMIT_THRESHOLD_RG`, `RG_EPITHELIAL_RATE`, `CELL_PRODUCTION_MULTIPLIER`, `INIT_CELL_PRODUCTION_RATES_ENV`, `ECM_AGENTS_PER_DIR`, `COORDS_BOUNDARIES`, `TIME_STEP`.

**Algorithm:**

```
1. Read own position, cell_type, rg_commitment, epithelialization_level, apx/y/z.

2. Find own ECM voxel index (same formula as in cell_move.cpp).
   Sample local_lumen_signal = C_SP_MACRO[2][vox_idx]  [a.u.]

3. Iterate spatial message neighbours → count neighbours with cell_type == 2.
   local_rg_fraction = (count RG neighbours) / (total neighbours + ε)   [-]

4. Commitment ODE (logistic, Euler):
     d_commit = ( RG_COMMIT_RATE
                + RG_COMMIT_AUTOCRINE_RATE * local_lumen_signal
                + RG_COMMIT_PARACRINE_RATE * local_rg_fraction )
               * (1.0 - rg_commitment)          // logistic saturation
     rg_commitment = clamp(rg_commitment + d_commit * TIME_STEP, 0, 1)

5. Cell-type switch:
     new_type = (rg_commitment >= RG_COMMIT_THRESHOLD_RG)  ? 2
              : (rg_commitment >= RG_COMMIT_THRESHOLD_NPC) ? 1
              : 0
   If new_type != old cell_type:
     cell_type = new_type
     k_production[2] = INIT_CELL_PRODUCTION_RATES_ENV[2]
                       * CELL_PRODUCTION_MULTIPLIER[new_type]   // [1/s] update secretion

6. Epithelialization ODE (logistic, Euler):
     d_epith = RG_EPITHELIAL_RATE * rg_commitment * (1.0 - epithelialization_level)
     epithelialization_level = clamp(epithelialization_level + d_epith * TIME_STEP, 0, 1)

7. Rosette maturity (readout — iterates neighbours a second time):
     Accumulate dot(own_ap, neighbour_ap) for each neighbour with cell_type == 2.
     rosette_maturity = mean over RG neighbours (0 if none)   [-]

8. Write: rg_commitment, epithelialization_level, cell_type, k_production,
          local_rg_fraction, local_lumen_signal, rosette_maturity.
```

---

## 9. New Agent Function: `cell_rg_polarity_update.cpp`

**Layer:** `L6b_RG_Polarity_Update` — after L6 (ECM boundary update) so the species-2 field reflects the latest diffusion; before L7 (cell-cell interaction, which reads `apx/y/z` from the spatial message at L1 — note: the L1 message is from the *previous* step, so the polarity used for `rosette_maturity` in L3b is one step behind; this is acceptable).

**Message:** none.

**Reads from env:** `ECM_AGENTS_PER_DIR`, `COORDS_BOUNDARIES`, `RG_POLARITY_TAU`, `RG_POLARITY_GRADIENT_THRESHOLD`, `TIME_STEP`.

**Algorithm — 26-neighbour gradient (same pattern as chemotaxis in `cell_move.cpp`):**

```
1. Find own ECM voxel (i, j, k).

2. Loop over di, dj, dk in {-1, 0, 1}³, skip (0,0,0):
     Clamp neighbour index to grid bounds.
     Compute displacement vector (ddx, ddy, ddz) in µm.
     dist = ||(ddx, ddy, ddz)||
     unit = (ddx, ddy, ddz) / dist
     dC   = C_SP_MACRO[2][neighbour_idx] - C_SP_MACRO[2][own_idx]   [a.u.]
     weight = dC / dist²                                             [a.u./µm²]
     Accumulate: grad += weight * unit

3. |grad| is the signal gradient magnitude [a.u./µm].

4. If |grad| > RG_POLARITY_GRADIENT_THRESHOLD:
     target = normalize(grad)            // apical vector toward rising species-2
   Else:
     target = current (apx, apy, apz)   // no update if signal is flat

5. Exponential blend (tau = RG_POLARITY_TAU [s]):
     alpha   = 1 - expf(-TIME_STEP / RG_POLARITY_TAU)
     ap_new  = (1-alpha)*(apx, apy, apz) + alpha*target
     normalize(ap_new)

6. Write: apx, apy, apz.
```

---

## 10. Override: `cell_spatial_location_data.cpp`

Copy of base file.  Add five extra `message.setVariable` calls:

```cpp
message.setVariable<float>("rg_commitment",
    FLAMEGPU->getVariable<float>("rg_commitment"));            // [-]
message.setVariable<float>("epithelialization_level",
    FLAMEGPU->getVariable<float>("epithelialization_level"));  // [-]
message.setVariable<float>("apx",
    FLAMEGPU->getVariable<float>("apx"));                      // [–]
message.setVariable<float>("apy",
    FLAMEGPU->getVariable<float>("apy"));                      // [–]
message.setVariable<float>("apz",
    FLAMEGPU->getVariable<float>("apz"));                      // [–]
```

---

## 11. Override: `cell_cell_interaction.cpp`

Copy of base file.  Changes:

1. Read `epithelialization_level` from own agent and from neighbour message.
2. Replace the scalar `CELL_CELL_ADHESION_K[type]` lookup with the `RG_ADHESION_MATRIX` array:
   ```cpp
   const float K_adh = FLAMEGPU->environment.getProperty<float, 9>(
       "RG_ADHESION_MATRIX", agent_cell_type * 3 + nb_cell_type);  // [nN/µm]
   ```
3. Similarly for `RG_REPULSION_MATRIX`.
4. RG-RG adhesion boost:
   ```cpp
   float adh_eff = K_adh;
   if (agent_cell_type == 2 && nb_cell_type == 2) {
       const float boost = FLAMEGPU->environment.getProperty<float>("RG_EPITHELIAL_ADHESION_BOOST");
       const float epith_min = fminf(epithelialization_level, nb_epithelialization);
       adh_eff *= (1.0f + (boost - 1.0f) * epith_min);   // scales from K_adh to boost*K_adh
   }
   ```

---

## 12. Override: `cell_move.cpp`

Copy of base file.  Two additions inserted **before** the final velocity accumulation step.

### Substrate spring (all cell types)

```cpp
// Restore cell toward substrate plane  [µm/s]
// RG_SUBSTRATE_K [nN/µm] / d_dumping [nN·s/µm] gives velocity contribution
const float z_rest = COORD_BOUNDARY_Z_NEG + RG_SUBSTRATE_K_Z0;  // [µm]
const float dz_sub = z_rest - agent_z;      // positive when cell is above rest height
const float v_sub  = RG_SUBSTRATE_K * dz_sub / agent_d_dumping;  // [µm/s]
agent_vz += v_sub;   // positive pushes cell down; z-floor clamping below prevents undercut
```

`d_dumping` is already read by the base function (`agent_d_dumping = FLAMEGPU->getVariable<float>("d_dumping")`).

### Apical migration bias (NPC and RG only)

```cpp
if (agent_cell_type >= 1) {
    const float bias_strength = (agent_cell_type == 2)
        ? FLAMEGPU->environment.getProperty<float>("RG_APICAL_BIAS_RG")    // [µm/s]
        : FLAMEGPU->environment.getProperty<float>("RG_APICAL_BIAS_NPC");  // [µm/s]
    const float epith = FLAMEGPU->getVariable<float>("epithelialization_level");  // [-]
    const float effective_bias = bias_strength * epith;  // ramps up with polarity
    const float ap_x = FLAMEGPU->getVariable<float>("apx");  // [–]
    const float ap_y = FLAMEGPU->getVariable<float>("apy");  // [–]
    const float ap_z = FLAMEGPU->getVariable<float>("apz");  // [–]
    agent_vx += effective_bias * ap_x;   // [µm/s]
    agent_vy += effective_bias * ap_y;   // [µm/s]
    agent_vz += effective_bias * ap_z;   // [µm/s]
}
```

The z-floor clamping (`agent_z = max(agent_z, COORD_BOUNDARY_Z_NEG)`) already present in the base file is kept unchanged.

---

## 13. Layer Order in `configure_layers`

The RG variant inserts two layers mid-sequence.  It **cannot** call `_build_default_layers()` and must replicate the full registration:

```
L1   cell_spatial_location_data     ← OVERRIDE
     (+ cell_bucket_location_data if INCLUDE_FOCAL_ADHESIONS)
L2   ECM boundary conditions         ← default
L3   cell_ecm_interaction_metabolism ← default (or OVERRIDE for Michaelis-Menten)
     (no cell_cycle in RG variant)
L3b  cell_rg_differentiation         ← NEW (reads L1 spatial message)
L4   ECM Csp update                  ← default
L5   ECM-ECM diffusion               ← default
L6   ECM boundary (second call)      ← default
L6b  cell_rg_polarity_update         ← NEW (reads updated ECM macro)
L7   cell_cell_interaction           ← OVERRIDE
     cell_stress_state_update        ← default
L8   cell_move                       ← OVERRIDE
```

L0 (VASC) is omitted because `INCLUDE_VASCULARIZATION = False`.

---

## 14. Variant `__init__.py` — Key Contents

```python
# variants/radial_glia/__init__.py

from pathlib import Path
_HERE = Path(__file__).parent

PARAMS = {
    # --- Assay type ---
    "MONOLAYER_ASSAY":            True,
    "ORGANOID_ASSAY":             False,
    "MONOLAYER_CELL_TYPE_RATIOS": [1, 0, 0],   # [-]  all iPSC at start
    # --- Agents ---
    "INCLUDE_CELLS":              True,
    "INCLUDE_CELL_CELL_INTERACTION": True,
    "INCLUDE_CELL_CYCLE":         False,
    "INCLUDE_DIFFUSION":          True,
    "INCLUDE_VASCULARIZATION":    False,
    "INCLUDE_FIBRE_NETWORK":      False,
    "INCLUDE_LUMEN":              False,
    # --- Cell geometry ---
    "N_CELLS":                    60,
    "CELL_RADIUS":                [5.0, 5.0, 5.0],        # [µm]
    # --- Cell motility (type 0=iPSC fastest, 2=RG slowest) ---
    "CELL_SPEED_REF":             [5e-4, 3e-4, 1e-4],     # [µm/s]
    "ROTATIONAL_DIFFUSION_RATE":  [2e-3, 1e-3, 2e-4],     # [rad²/s]
    # --- Cell-cell mechanics replaced by RG_ADHESION_MATRIX ---
    "CELL_CELL_ADHESION_K":       [0.0, 0.0, 0.0],        # [nN/µm]  disabled; matrix used instead
    "CELL_CELL_REPULSION_K":      [4.0, 4.0, 4.0],        # [nN/µm]
    # --- Diffusion ---
    "DIFFUSION_COEFF_MULTI":      [5.0, 5.0, 2.0],        # [µm²/s]  sp2 slower (morphogen)
    "BOUNDARY_CONC_INIT_MULTI":   [[2.5]*6, [2.5]*6, [0.0]*6],
    "BOUNDARY_CONC_FIXED_MULTI":  [[2.5]*6, [2.5]*6, [0.0]*6],
    "INIT_CELL_CONCENTRATION_VALS": [2.5, 2.5, 0.0],      # [a.u.]  sp2 starts at 0
    # --- Metabolism / secretion ---
    "INIT_CELL_PRODUCTION_RATES": [0.0, 0.0, 5e-4],       # [1/s]   sp2 base rate for fully committed RG
    "CELL_PRODUCTION_MULTIPLIER": [0.0, 0.5, 1.0],        # [-]     type 0: silent, 1: half, 2: full
    # --- Simulation timing ---
    "TIME_STEP":                  10.0,                    # [s]
    "STEPS":                      3600,                    # [-]     → 10 simulated hours
    "SAVE_EVERY_N_STEPS":         60,                      # [-]     → save every 10 min
}

FILES = {
    "cell_spatial_location_data_file": str(_HERE / "cell_spatial_location_data.cpp"),
    "cell_cell_interaction_file":      str(_HERE / "cell_cell_interaction.cpp"),
    "cell_move_file":                  str(_HERE / "cell_move.cpp"),
    # cell_rg_differentiation and cell_rg_polarity_update are registered
    # programmatically in configure_layers (no base *_file variable exists).
}

# VTK extra outputs (used by helper_module when INCLUDE_RG_VARIABLES is True)
CELL_VTK_EXTRA_SCALARS = [
    ("float", "rg_commitment"),           # [-]
    ("float", "epithelialization_level"), # [-]
    ("float", "rosette_maturity"),        # [-]
    ("float", "local_rg_fraction"),       # [-]
    ("float", "local_lumen_signal"),      # [a.u.]
]
CELL_VTK_EXTRA_VECTORS = [
    ("apical_vector", "apx", "apy", "apz"),   # [–]  unit vector
]


def configure_globals(g: dict) -> None:
    """Inject RG-specific scalars that aren't in PARAMS (not pre-existing model.py globals)."""
    # RG differentiation
    g["RG_COMMIT_RATE"]               = 2e-5    # [1/s]
    g["RG_COMMIT_AUTOCRINE_RATE"]     = 5e-4    # [1/(s·a.u.)]
    g["RG_COMMIT_PARACRINE_RATE"]     = 1e-4    # [1/s]
    g["RG_COMMIT_THRESHOLD_NPC"]      = 0.35    # [-]
    g["RG_COMMIT_THRESHOLD_RG"]       = 0.70    # [-]
    g["RG_EPITHELIAL_RATE"]           = 1e-4    # [1/s]
    # Polarity
    g["RG_POLARITY_TAU"]              = 600.0   # [s]
    g["RG_POLARITY_GRADIENT_THRESHOLD"] = 1e-6  # [a.u./µm]
    # Substrate
    g["RG_SUBSTRATE_K"]               = 0.5     # [nN/µm]
    g["RG_SUBSTRATE_Z0"]              = 0.0     # [µm]
    # Migration bias
    g["RG_APICAL_BIAS_RG"]            = 2e-3    # [µm/s]
    g["RG_APICAL_BIAS_NPC"]           = 5e-4    # [µm/s]
    # Cell-cell adhesion matrix (3×3 flattened, row=self, col=neighbour)
    g["RG_ADHESION_MATRIX"]  = [0.4, 0.4, 0.2,
                                 0.4, 0.8, 0.6,
                                 0.2, 0.6, 2.0]   # [nN/µm]
    g["RG_REPULSION_MATRIX"] = [4.0]*9            # [nN/µm]
    g["RG_EPITHELIAL_ADHESION_BOOST"] = 3.0       # [-]


def configure_layers(model, g: dict) -> None:
    """Full layer sequence for the RG variant with two inserted layers."""
    _env = g.get("env")
    if _env is not None:
        _register_rg_env_properties(_env, g)

    # Register new RTC agent functions on the CELL agent
    from pathlib import Path
    CELL_agent = model.Agent("CELL")
    rg_diff_code = (_HERE / "cell_rg_differentiation.cpp").read_text()
    rg_diff_fn = CELL_agent.newRTCFunction("cell_rg_differentiation", rg_diff_code)
    rg_diff_fn.setMessageInput("cell_spatial_location_message")

    rg_pol_code = (_HERE / "cell_rg_polarity_update.cpp").read_text()
    CELL_agent.newRTCFunction("cell_rg_polarity_update", rg_pol_code)

    # Build the full layer sequence (cannot call _build_default_layers() because
    # insertions are mid-sequence; replicate relevant portions here).
    _build_rg_layers(model, g)
```

The helper `_register_rg_env_properties` and `_build_rg_layers` are module-level functions in `__init__.py`.

---

## 15. VTK Extra Outputs

Handled in `helper_module.py` once `CELL_VTK_EXTRA_SCALARS` / `CELL_VTK_EXTRA_VECTORS` support is added.  The writing block is gated by `config.get("INCLUDE_RG_VARIABLES", False)` (or equivalently by checking whether the lists are non-empty).

---

## 16. Implementation Order

| Step | Target | What |
|---|---|---|
| 1 | `model.py` | Add `VARIANTS_WITH_RG_VARIABLES`, `INCLUDE_RG_VARIABLES` flag |
| 2 | `model.py` | Add 9 CELL agent variables (gated) |
| 3 | `model.py` | Add 5 message variables (gated) |
| 4 | `model.py` | Add `MONOLAYER_CELL_TYPE_RATIOS`, update MONOLAYER_ASSAY init branch |
| 5 | `model.py` | Initialize RG variables in CELL host loop (gated) |
| 6 | `model.py` | Register `INIT_CELL_PRODUCTION_RATES_ENV` env property |
| 7 | `helper_module.py` | Add `CELL_VTK_EXTRA_SCALARS` / `CELL_VTK_EXTRA_VECTORS` loop |
| 8 | `variants/radial_glia/` | Copy + modify `cell_spatial_location_data.cpp` |
| 9 | `variants/radial_glia/` | Copy + modify `cell_cell_interaction.cpp` |
| 10 | `variants/radial_glia/` | Copy + modify `cell_move.cpp` |
| 11 | `variants/radial_glia/` | Write `cell_rg_differentiation.cpp` from scratch |
| 12 | `variants/radial_glia/` | Write `cell_rg_polarity_update.cpp` from scratch |
| 13 | `variants/radial_glia/` | Write `__init__.py` |
| 14 | smoke test | `python model.py --variant radial_glia` — 1-step run, no errors |
| 15 | validation | Full 3600-step run; check rg_commitment rises, apical vectors align |

---

## 17. Known Constraints and Hard-Coded C++ Constants

| Constant | Value | Location | Note |
|---|---|---|---|
| `N_CELL_TYPES` | 3 | all agent .cpp | No change; 3 types fit exactly |
| `N_SPECIES` | 3 | all ECM/cell .cpp | No change; species 2 already allocated |
| `N_ANCHOR_POINTS` | 50 | cell_move, cell_stress | No change |
| `ECM_POPULATION_SIZE` | 9261 | all .cpp that read macro | No change; N=21, cubical domain kept |
| `MAX_CONNECTIVITY` | 8 | fnode .cpp | Irrelevant (FN disabled) |

---

## 18. Open Questions / Future Work

- **Interkinetic nuclear migration (INM)**: RG cells move their nucleus apico-basally during the cell cycle (G2/M near apical surface, G1 near basal).  Could be modelled by time-varying `apz` amplitude tied to `clock`.  Not in scope.
- **Lumen agents**: Once rosette geometry is established, LUMEN agents can fill the central apical space (`INCLUDE_LUMEN = True`, `MONOLAYER_ASSAY` branch of lumen init).  The `lumen_id` variable is already reserved.
- **Substrate stiffness sensing**: RG fate is stiffness-sensitive.  Could weight `RG_COMMIT_RATE` by `sig_zz` (already computed by `cell_stress_state_update`).
- **Basal anchoring via FOCAD**: Would replace the simple substrate spring with a proper molecular-clutch adhesion model.
- **Metabolism override**: Design fully documented in Section 3. Implement after core differentiation and polarity are validated.
