"""
Analytical solver for the CHEMOKINESIS (TGFb) case.

In this scenario the control parameters (CELL_SPEED_REF, ROTATIONAL_DIFFUSION_RATE)
are FIXED from the control calibration, and chemokinesis modulates the persistent
speed.  The free parameters are:
  - CHEMOKINESIS_ALPHA  (per cell type)
  - BROWNIAN_MOTION_STRENGTH_FACTOR  (per cell type)

Model (per timestep, with chemokinesis):
  persistent_speed = speed_ref * (1 + ALPHA * h_promo)
  brownian_strength = speed_ref * BF
  vmean = sqrt(persistent_speed^2 + brownian_strength^2)
  MSD   = 2*tau*g(T)*persistent_speed^2 + dt*T*brownian_strength^2
  veff  = sqrt(MSD) / T

where:
  h_promo   = hill01(signal_eff, K, n)
  signal_eff = saturate(SENSITIVITY * C_local, SAT_LEVEL) - adapt_state
  tau = 1/(2*D_rot_eff),  g(T) = T - tau*(1-exp(-T/tau))
  D_rot_eff = D_rot_nominal / 3  (see ROT_DIFF_NOISE_CORRECTION below)

With constant uniform concentration C, SENSITIVITY=[1,0], large ADAPT_TAU
(adaptation disabled), SAT, SAT_MULT, K, n all fixed, h_promo is a constant
that is the same for all cell types (but see note on per-type SAT_MULT).

NOTE: The chemokinesis signal chain can be computed analytically for a
      constant concentration field (no gradients, no metabolism).
"""

import math
import sys

# ============================================================================
# CONFIGURABLE PARAMETERS — edit these to match your scenario
# ============================================================================

# ----- Simulation timing -----
T_target = 86400.0      # Total simulation time [s]
dt_target = 60.0        # Timestep [s]

# ----- Rotational diffusion noise correction -----
# The GPU code (cell_move.cpp) uses uniform(-1,1) noise for rotational
# diffusion instead of Gaussian N(0,1). Since Var(U(-1,1)) = 1/3 while
# Var(N(0,1)) = 1, the effective rotational diffusion rate in simulation
# is D_rot_eff = D_rot_nominal / 3.  This makes cells ~3x more persistent
# than a Gaussian scheme would give.  Set to 1.0 to disable if the GPU
# code is changed to use Gaussian noise.
ROT_DIFF_NOISE_CORRECTION = 3.0

# ----- Fixed control parameters (from control calibration) -----
# Replace these with your best control-case values for the target (T, dt)
speed_ref = [0.00050676141519674043, 0.0008477627954233326, 0.0005371730179291174]
D_rot     = [0.000035, 0.000037, 0.000055]

# ----- TGFb targets (from optimizer/reference_data/target_cell_speed_tgfb.csv) -----
tgfb_vmean = [0.0013, 0.0013, 0.0013]
tgfb_veff  = [0.0004, 0.00037, 0.00039]

# ----- Chemokinesis signal chain (fixed overrides from YAML) -----
C_LOCAL       = 2.5       # Uniform concentration of species 0
SENSITIVITY_0 = 1.0       # CHEMOKINESIS_SENSITIVITY[0]
SAT_BASE      = 50.0      # CHEMOKINESIS_SIGNAL_SAT[0]
SAT_MULT      = [1.0, 1.0, 1.0]  # CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER per type
ADAPT_TAU     = 1e6       # CHEMOKINESIS_ADAPT_TAU (very large = no adaptation)
K_HILL        = 2.0       # CHEMOKINESIS_K
N_HILL        = 2.0       # CHEMOKINESIS_HILL_N

# ============================================================================
# Derived chemokinesis signal computation (mirrors cell_move.cpp exactly)
# ============================================================================

def saturate_signal(signal, sat_level):
    s = max(signal, 0.0)
    smax = max(sat_level, 1e-12)
    return smax * s / (smax + s)

def hill01(c, K, n):
    c_pos = max(c, 0.0)
    K_pos = max(K, 1e-12)
    nn = max(n, 1.0)
    cn = c_pos ** nn
    Kn = K_pos ** nn
    return cn / (Kn + cn + 1e-12)

def compute_h_promo(cell_type, n_steps=None):
    """
    Compute the effective h_promo for a given cell type.
    With constant C and very large ADAPT_TAU, adaptation state stays ~0,
    so signal_eff ≈ signal_sat.
    If n_steps is given, simulate the adaptation state evolution.
    """
    sat_level = SAT_BASE * SAT_MULT[cell_type]
    raw_signal = SENSITIVITY_0 * C_LOCAL
    signal_sat = saturate_signal(raw_signal, sat_level)

    if n_steps is not None and ADAPT_TAU < 1e12:
        # Simulate adaptation
        dt_sim = dt_target
        adapt_state = 0.0
        for _ in range(n_steps):
            adapt_state += (dt_sim / ADAPT_TAU) * (signal_sat - adapt_state)
            adapt_state = max(adapt_state, 0.0)
        signal_eff = max(signal_sat - adapt_state, 0.0)
    else:
        # Large ADAPT_TAU: adapt_state ≈ 0 for short/medium sims
        # For very long sims: adapt_state → signal_sat eventually
        if ADAPT_TAU > 1e10:
            signal_eff = signal_sat
        else:
            n_steps_default = int(T_target / dt_target)
            adapt_state = signal_sat * (1.0 - math.exp(-n_steps_default * dt_target / ADAPT_TAU))
            signal_eff = max(signal_sat - adapt_state, 0.0)

    h = hill01(signal_eff, K_HILL, N_HILL)
    return h, signal_sat, signal_eff


# ============================================================================
# Core analytical functions (same as control script)
# ============================================================================

def g_func(T, tau):
    ratio = T / tau
    if ratio > 500:
        return T - tau
    return T - tau * (1.0 - math.exp(-ratio))


def predict(p_speed, b_speed, D_rot_val, T, dt):
    vmean = math.sqrt(p_speed**2 + b_speed**2)
    tau = ROT_DIFF_NOISE_CORRECTION / (2.0 * D_rot_val)
    g = g_func(T, tau)
    msd_p = 2.0 * tau * g * p_speed**2
    msd_b = dt * T * b_speed**2
    veff = math.sqrt(msd_p + msd_b) / T
    return {
        'vmean': vmean, 'veff': veff,
        'msd_persistent': msd_p, 'msd_brownian': msd_b,
        'msd_total': msd_p + msd_b,
    }


def solve_chemokinesis(type_i, target_vm, target_ve, T, dt):
    """
    Solve for (ALPHA, BF) given fixed speed_ref[i] and D_rot[i].

    persistent_speed = speed_ref * (1 + ALPHA * h_promo)
    brownian_speed   = speed_ref * BF

    From vmean and veff targets:
      vmean^2 = p^2 + b^2
      (veff*T)^2 = coeff*p^2 + dt*T*b^2

    where coeff = 2*tau*g(T), b = speed_ref*BF, p = speed_ref*(1+ALPHA*h)

    Substituting b^2 = vmean^2 - p^2:
      (veff*T)^2 = p^2*(coeff - dt*T) + dt*T*vmean^2
      p^2 = [(veff*T)^2 - dt*T*vmean^2] / [coeff - dt*T]

    Then:
      ALPHA = (p / speed_ref - 1) / h_promo
      BF    = b / speed_ref
    """
    sr = speed_ref[type_i]
    dr = D_rot[type_i]
    tau = ROT_DIFF_NOISE_CORRECTION / (2.0 * dr)
    g = g_func(T, tau)
    coeff = 2.0 * tau * g

    h_promo, sig_sat, sig_eff = compute_h_promo(type_i, int(T / dt))

    numerator = (target_ve * T)**2 - dt * T * target_vm**2
    denominator = coeff - dt * T

    if abs(denominator) < 1e-30:
        return None, "denominator ~= 0 (degenerate)"

    p2 = numerator / denominator
    if p2 < 0:
        return None, f"p^2 = {p2:.2e} < 0 (no real solution)"

    b2 = target_vm**2 - p2
    if b2 < 0:
        return None, f"b^2 = {b2:.2e} < 0 (persistent speed exceeds vmean)"

    p = math.sqrt(p2)
    b = math.sqrt(b2)

    if h_promo < 1e-15:
        return None, "h_promo ~= 0 (chemokinesis signal ineffective)"

    alpha = (p / sr - 1.0) / h_promo
    bf = b / sr

    # Verify
    p_check = sr * (1.0 + alpha * h_promo)
    b_check = sr * bf
    pred = predict(p_check, b_check, dr, T, dt)

    return {
        'alpha': alpha,
        'bf': bf,
        'h_promo': h_promo,
        'signal_sat': sig_sat,
        'signal_eff': sig_eff,
        'p': p,
        'b': b,
        'p_check': p_check,
        'b_check': b_check,
        'vmean_pred': pred['vmean'],
        'veff_pred': pred['veff'],
        'vmean_err': abs(pred['vmean'] - target_vm) / target_vm * 100,
        'veff_err': abs(pred['veff'] - target_ve) / target_ve * 100,
        'msd_p_frac': pred['msd_persistent'] / pred['msd_total'] * 100 if pred['msd_total'] > 0 else 0,
        'chemokinesis_factor': 1.0 + alpha * h_promo,
    }, None


# ============================================================================
# PART 1: Chemokinesis signal chain analysis
# ============================================================================
print('=' * 85)
print('PART 1: CHEMOKINESIS SIGNAL CHAIN')
print('=' * 85)

print(f'\n  C_local = {C_LOCAL}')
print(f'  SENSITIVITY[0] = {SENSITIVITY_0}')
print(f'  SAT_BASE[0] = {SAT_BASE}')
print(f'  K = {K_HILL}, n = {N_HILL}')
print(f'  ADAPT_TAU = {ADAPT_TAU:.0e}')
print()

n_steps = int(T_target / dt_target)
for i in range(3):
    h, sig_sat, sig_eff = compute_h_promo(i, n_steps)
    sat_level = SAT_BASE * SAT_MULT[i]
    print(f'  Type {i}: SAT_MULT={SAT_MULT[i]:.2f} -> sat_level={sat_level:.1f}')
    print(f'           signal_raw={SENSITIVITY_0*C_LOCAL:.3f} -> signal_sat={sig_sat:.4f} -> signal_eff={sig_eff:.4f}')
    print(f'           h_promo = {h:.6f}')
    print(f'           chemokinesis_factor range = [{1-h:.4f}, {1+h:.4f}]  (for ALPHA in [-1, +1])')
    print()


# ============================================================================
# PART 2: Fixed control parameters review
# ============================================================================
print('=' * 85)
print('PART 2: FIXED CONTROL PARAMETERS')
print('=' * 85)

print(f'\n  Simulation: T = {T_target:.0f}s ({T_target/3600:.0f}h), dt = {dt_target:.0f}s, steps = {n_steps}')
print()
for i in range(3):
    tau_eff = ROT_DIFF_NOISE_CORRECTION / (2.0 * D_rot[i])
    print(f'  Type {i}: speed_ref = {speed_ref[i]:.6f}, D_rot = {D_rot[i]:.2e} (tau_eff = {tau_eff:.0f}s)')


# ============================================================================
# PART 3: What do the OLD chemokinesis params predict at T_target?
# ============================================================================
print()
print('=' * 85)
print('PART 3: OLD PARAMS (from speed_tgfb_paper.json) AT DIFFERENT T')
print('=' * 85)

# Old chemokinesis params (from speed_tgfb_paper.json)
old_speed_ref = [0.00039467864789234224, 0.0006043654028727149, 0.00032985670642352277]
old_bf        = [3.1339, 2.0602, 3.7610]
old_drot      = [0.0003757389100327902, 0.0006028399869074992, 0.0003342841399542819]
old_alpha     = [0.0234, -0.6506, 0.3031]

configs = [
    ('T=200s, dt=10s (original optimization)',                                         200.0,    10.0),
    (f'T={T_target:.0f}s, dt={dt_target:.0f}s ({T_target/3600:.1f}h target)',          T_target, dt_target),
]

for label, T, dt in configs:
    print(f'\n  --- {label} ---')
    for i in range(3):
        h, _, _ = compute_h_promo(i, int(T / dt))
        p = old_speed_ref[i] * (1.0 + old_alpha[i] * h)
        b = old_speed_ref[i] * old_bf[i]
        pred = predict(p, b, old_drot[i], T, dt)
        vm_err = (pred['vmean'] - tgfb_vmean[i]) / tgfb_vmean[i] * 100
        ve_err = (pred['veff'] - tgfb_veff[i]) / tgfb_veff[i] * 100
        print(f'    Type {i}: vmean={pred["vmean"]:.6f} (err {vm_err:+.1f}%)'
              f'  |  veff={pred["veff"]:.6f} (err {ve_err:+.1f}%)')


# ============================================================================
# PART 4: Analytical solutions for the TGFb case
# ============================================================================
print()
print('=' * 85)
print(f'PART 4: ANALYTICAL SOLUTIONS FOR T={T_target:.0f}s, dt={dt_target:.0f}s')
print('=' * 85)

best_solutions = []
all_feasible = True

for i in range(3):
    sol, err = solve_chemokinesis(i, tgfb_vmean[i], tgfb_veff[i], T_target, dt_target)

    if sol is None:
        print(f'\n  Type {i}: NO SOLUTION — {err}')
        best_solutions.append(None)
        all_feasible = False
        continue

    # Check feasibility
    flags = []
    if abs(sol['alpha']) > 1.0:
        flags.append(f'ALPHA={sol["alpha"]:.3f} OUT OF [-1,1]')
    if sol['bf'] < 0 or sol['bf'] > 10:
        flags.append(f'BF={sol["bf"]:.3f} OUT OF [0,10]')

    best_solutions.append(sol)
    s = sol
    print(f'\n  Cell type {i}:')
    print(f'    speed_ref (fixed)  = {speed_ref[i]:.6f}')
    print(f'    D_rot (fixed)      = {D_rot[i]:.2e} (tau_eff = {ROT_DIFF_NOISE_CORRECTION/(2*D_rot[i]):.0f}s)')
    print(f'    h_promo            = {s["h_promo"]:.6f}')
    print(f'    ------------------------------------')
    print(f'    CHEMOKINESIS_ALPHA[{i}] = {s["alpha"]:.6f}')
    print(f'    BROWNIAN_MOTION_STRENGTH_FACTOR[{i}] = {s["bf"]:.6f}')
    print(f'    ------------------------------------')
    print(f'    chemokinesis_factor = {s["chemokinesis_factor"]:.6f}')
    print(f'    persistent_speed   = {s["p"]:.6f} um/s')
    print(f'    brownian_strength  = {s["b"]:.6f} um/s')
    print(f'    vmean = {s["vmean_pred"]:.6f}  (target {tgfb_vmean[i]:.6f},  err {s["vmean_err"]:.6f}%)')
    print(f'    veff  = {s["veff_pred"]:.6f}  (target {tgfb_veff[i]:.6f},  err {s["veff_err"]:.6f}%)')
    print(f'    MSD: {s["msd_p_frac"]:.1f}% persistent, {100-s["msd_p_frac"]:.1f}% brownian')
    if flags:
        for f in flags:
            print(f'    *** WARNING: {f} ***')
        all_feasible = False


# ============================================================================
# PART 5: JSON snippet and optimizer suggestions
# ============================================================================
print()
print('=' * 85)
print('PART 5: RECOMMENDED CONFIGURATION')
print('=' * 85)

if all_feasible:
    print(f'\n  JSON snippet for TGFb chemokinesis at T={T_target:.0f}s, dt={dt_target:.0f}s:')
    print('  {')
    print(f'    "STEPS": {int(T_target/dt_target)},')
    print(f'    "TIME_STEP": {dt_target},')
    for i in range(3):
        print(f'    "CELL_SPEED_REF[{i}]": {speed_ref[i]},')
    for i in range(3):
        print(f'    "ROTATIONAL_DIFFUSION_RATE[{i}]": {D_rot[i]},')
    for i in range(3):
        s = best_solutions[i]
        print(f'    "CHEMOKINESIS_ALPHA[{i}]": {s["alpha"]},')
    for i in range(3):
        s = best_solutions[i]
        print(f'    "BROWNIAN_MOTION_STRENGTH_FACTOR[{i}]": {s["bf"]},')
    print('    ...')
    print('  }')

    print()
    print('  Suggested optimizer YAML ranges:')
    print()
    print('  CHEMOKINESIS_ALPHA:')
    print('    type: array_float')
    print('    elements:')
    for i in range(3):
        a = best_solutions[i]['alpha']
        lo = max(-1.0, a - 0.5)
        hi = min(1.0, a + 0.5)
        print(f'      {i}: {{low: {lo:.2f}, high: {hi:.2f}}}')
    print()
    print('  BROWNIAN_MOTION_STRENGTH_FACTOR:')
    print('    type: array_float')
    print('    elements:')
    for i in range(3):
        bf = best_solutions[i]['bf']
        lo = max(0.0, bf / 2.0)
        hi = min(10.0, bf * 2.0)
        print(f'      {i}: {{low: {lo:.2f}, high: {hi:.2f}}}')
else:
    print('\n  WARNING: Not all types have feasible solutions.')
    print('  This may indicate that the control parameters (speed_ref, D_rot)')
    print('  cannot reach the TGFb targets even with chemokinesis modulation.')
    print('  Consider:')
    print('    1. Re-calibrating D_rot at the target T (lower D_rot = higher veff)')
    print('    2. Relaxing the TGFb targets')
    print('    3. Widening the ALPHA range beyond [-1, 1]')

    # Show partial results anyway
    print()
    for i in range(3):
        s = best_solutions[i]
        if s is not None:
            print(f'  Type {i}: ALPHA={s["alpha"]:.4f}, BF={s["bf"]:.4f}'
                  f'  (vmean err {s["vmean_err"]:.2f}%, veff err {s["veff_err"]:.2f}%)')
        else:
            print(f'  Type {i}: NO SOLUTION')


# ============================================================================
# PART 6: Cross-check at different T
# ============================================================================
print()
print('=' * 85)
print('PART 6: CROSS-CHECK - CHEMOKINESIS PARAMS AT DIFFERENT T')
print('=' * 85)

if not all_feasible:
    print('  Show results for types that have solutions.')

check_T_values = [200.0, 3600.0, T_target, T_target * 2]
check_configs = []
for T_chk in check_T_values:
    label = f'T={T_chk:.0f}s, dt={dt_target:.0f}s ({T_chk/3600:.1f}h)'
    check_configs.append((label, T_chk, dt_target))

for label, T, dt in check_configs:
    print(f'\n  --- {label} ---')
    for i in range(3):
        s = best_solutions[i]
        if s is None:
            print(f'    Type {i}: (no solution)')
            continue
        h, _, _ = compute_h_promo(i, int(T / dt))
        p = speed_ref[i] * (1.0 + s['alpha'] * h)
        b = speed_ref[i] * s['bf']
        pred = predict(p, b, D_rot[i], T, dt)
        ve_err = (pred['veff'] - tgfb_veff[i]) / tgfb_veff[i] * 100
        print(f'    Type {i}: vmean={pred["vmean"]:.6f}  veff={pred["veff"]:.6f} '
              f'(target {tgfb_veff[i]:.6f}, err {ve_err:+.1f}%)')


# ============================================================================
# PART 7: Sensitivity — how does ALPHA affect vmean/veff?
# ============================================================================
print()
print('=' * 85)
print('PART 7: SENSITIVITY - ALPHA SWEEP (with BF fixed at analytical value)')
print('=' * 85)

if any(s is not None for s in best_solutions):
    for i in range(3):
        s = best_solutions[i]
        if s is None:
            continue
        print(f'\n  Cell type {i} (BF={s["bf"]:.4f}, speed_ref={speed_ref[i]:.6f}):')
        print(f'    {"ALPHA":>7s}  {"factor":>7s}  {"p_speed":>8s}  {"vmean":>8s}  {"veff":>8s}  {"vm_err":>7s}  {"ve_err":>7s}')
        for alpha in [-1.0, -0.5, -0.3, 0.0, s['alpha'], 0.3, 0.5, 1.0]:
            h, _, _ = compute_h_promo(i, n_steps)
            factor = 1.0 + alpha * h
            if factor < 0:
                continue
            p = speed_ref[i] * factor
            b = speed_ref[i] * s['bf']
            pred = predict(p, b, D_rot[i], T_target, dt_target)
            vm_err = (pred['vmean'] - tgfb_vmean[i]) / tgfb_vmean[i] * 100
            ve_err = (pred['veff'] - tgfb_veff[i]) / tgfb_veff[i] * 100
            marker = ' <<<' if abs(alpha - s['alpha']) < 0.001 else ''
            print(f'    {alpha:+7.3f}  {factor:7.4f}  {p:8.6f}  {pred["vmean"]:8.6f}  {pred["veff"]:8.6f}  {vm_err:+6.1f}%  {ve_err:+6.1f}%{marker}')
