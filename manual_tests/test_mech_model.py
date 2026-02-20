import numpy as np

# ── Material ──────────────────────────────────────────────────────────────────
E   = 600.0
nu  = 0.45
G   = E / (2 * (1 + nu))
dt  = 0.2
tau = np.inf   # relaxation time (ignored in step 1 since ε₀ = 0)

# ── Springs / adhesion points ─────────────────────────────────────────────────
# Each entry: (k, y - x) → gives force f = k*(y-x), position r relative to cell centre
springs = [
    {"k": 20, "displacement": np.array([0.2, 0.1, 0.0]), "r": np.array([1.0, 0.0, 0.0])},
    {"k": 10, "displacement": np.array([0.1, 0.3, 0.0]), "r": np.array([0.0, 1.0, 0.0])},
]

V = 1.0   # normalisation volume (set to 1 for this example)

# ── Step 1: Forces ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: FORCES")
print("=" * 60)
for i, s in enumerate(springs, 1):
    s["f"] = s["k"] * s["displacement"]
    print(f"  f{i} = {s['k']} × {s['displacement']} = {s['f']}")

# ── Step 2: Stress tensor (C++ logic reproduced exactly) ─────────────────────
print("\n" + "=" * 60)
print("STEP 2: TRACTION DIPOLE STRESS (Voigt accumulation)")
print("=" * 60)

# These mirror the C++ agent accumulators
S_xx = S_yy = S_zz = 0.0
S_xy = S_xz = S_yz = 0.0

# Cell centre (the "agent" position)
agent_pos = np.array([0.0, 0.0, 0.0])

for i, s in enumerate(springs, 1):
    # Position of adhesion point (message position)
    adhesion_pos = s["r"]   # here r IS the adhesion position relative to centre
    fx, fy, fz   = s["f"]

    # r vector: from agent to adhesion point (mirrors C++ message - agent)
    rx = adhesion_pos[0] - agent_pos[0]
    ry = adhesion_pos[1] - agent_pos[1]
    rz = adhesion_pos[2] - agent_pos[2]

    print(f"\n  Spring {i}:")
    print(f"    r = ({rx}, {ry}, {rz}),  f = ({fx}, {fy}, {fz})")

    # Diagonal terms (no factor of ½ needed — symmetric by construction)
    S_xx += rx * fx
    S_yy += ry * fy
    S_zz += rz * fz

    # Off-diagonal terms (the sym() operation)
    S_xy += 0.5 * (rx * fy + ry * fx)
    S_xz += 0.5 * (rx * fz + rz * fx)
    S_yz += 0.5 * (ry * fz + rz * fy)

    # Show contribution of this spring
    contrib = np.array([
        [rx*fx,                       0.5*(rx*fy+ry*fx),  0.5*(rx*fz+rz*fx)],
        [0.5*(rx*fy+ry*fx),           ry*fy,              0.5*(ry*fz+rz*fy)],
        [0.5*(rx*fz+rz*fx),           0.5*(ry*fz+rz*fy), rz*fz            ]
    ])
    print(f"    sym(r ⊗ f) =\n{contrib}")

# Assemble full stress tensor (symmetric)
sigma_raw = np.array([
    [S_xx, S_xy, S_xz],
    [S_xy, S_yy, S_yz],
    [S_xz, S_yz, S_zz]
])
sigma = sigma_raw / V

print("\n  σ = (1/V) Σ sym(rᵢ ⊗ fᵢ) =")
print(sigma)

# ── Step 3: Elastic strain via compliance (Voigt) ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: ELASTIC STRAIN FROM COMPLIANCE (Voigt mapping)")
print("=" * 60)

# Build 6×6 compliance matrix (Voigt order: xx, yy, zz, yz, xz, xy)
S_compliance = np.array([
    [ 1/E,   -nu/E,  -nu/E,  0,    0,    0   ],
    [-nu/E,   1/E,   -nu/E,  0,    0,    0   ],
    [-nu/E,  -nu/E,   1/E,   0,    0,    0   ],
    [ 0,      0,      0,     1/G,  0,    0   ],
    [ 0,      0,      0,     0,    1/G,  0   ],
    [ 0,      0,      0,     0,    0,    1/G ]
])

# Map stress tensor → Voigt vector (order: xx, yy, zz, yz, xz, xy)
sigma_voigt = np.array([
    sigma[0, 0],   # xx
    sigma[1, 1],   # yy
    sigma[2, 2],   # zz
    sigma[1, 2],   # yz
    sigma[0, 2],   # xz
    sigma[0, 1],   # xy
])
print(f"\n  σ (Voigt) = {sigma_voigt}")

# Apply compliance
eps_voigt = S_compliance @ sigma_voigt
print(f"  ε (Voigt) = {eps_voigt}")

# Map back → 3×3 tensor
eps_el = np.array([
    [eps_voigt[0], eps_voigt[5], eps_voigt[4]],
    [eps_voigt[5], eps_voigt[1], eps_voigt[3]],
    [eps_voigt[4], eps_voigt[3], eps_voigt[2]]
])
print(f"\n  εᵉˡ =\n{eps_el}")

# ── Step 4: Volume-preserving projection ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: VOLUME-PRESERVING PROJECTION")
print("=" * 60)

trace_eps = np.trace(eps_el)
print(f"\n  tr(εᵉˡ) = {trace_eps:.6f}")
print(f"  tr(εᵉˡ)/3 = {trace_eps/3:.6f}")

eps_dev = eps_el - (trace_eps / 3) * np.eye(3)
print(f"\n  ε̃ = εᵉˡ - (tr/3)·I =\n{eps_dev}")
print(f"\n  Verification tr(ε̃) = {np.trace(eps_dev):.2e}  (should be ≈ 0)")

# ── Step 5: Kelvin–Voigt update ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: KELVIN–VOIGT UPDATE")
print("=" * 60)

eps_t  = np.zeros((3, 3))   # ε₀ = 0
# Full update: εₜ₊Δₜ = εₜ + Δt·(ε̃ - εₜ/τ)
# Since ε₀ = 0, the -ε/τ term vanishes
eps_new = eps_t + dt * (eps_dev - eps_t / tau if tau != np.inf else eps_dev)

print(f"\n  Δt = {dt}")
print(f"  ε₀ = 0  →  -ε/τ term vanishes")
print(f"\n  ε₁ = ε₀ + Δt·ε̃ =\n{eps_new}")
