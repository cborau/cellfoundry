"""
Analytical solver for the CONTROL case of a migration scenario.

Shows why parameters optimized at some simulation times (e.g. T=200s) fail at other times (e.g. T=21600s), and computes
the best analytical (speed_ref, BF, D_rot) for a given simulation duration.

Model (per timestep):
  displacement = v * dt
  v = speed_ref * orientation + BMS * rand(-1,1)^3
  BMS = speed_ref * BF

Derived statistics:
  vmean ≈ sqrt(p^2 + b^2)                          (time-independent)
  MSD   = 2*tau*g(T)*p^2 + dt*T*b^2                (time-dependent!)
  veff  = sqrt(MSD) / T

where  p = speed_ref,  b = speed_ref * BF,
       tau = 1/(2*D_rot_eff),  g(T) = T - tau*(1-exp(-T/tau))
       D_rot_eff = D_rot_nominal / 3  (see ROT_DIFF_NOISE_CORRECTION below)

KEY INSIGHT: veff depends on BOTH T (total sim time) and dt (timestep).
Parameters that match targets at one (T, dt) will NOT match at another.
"""

import math
import sys

# ============================================================================
# CONFIGURABLE PARAMETERS — edit these to match your scenario
# ============================================================================
T_target  = 86400.0     # Total simulation time [s]
dt_target = 60.0        # Timestep [s]

# Correction for uniform noise in GPU rotational diffusion (cell_move.cpp).
# The GPU uses uniform(-1,1) noise (variance=1/3) instead of Gaussian N(0,1)
# (variance=1), so the effective D_rot is D_rot_nominal / 3.  This makes
# cells ~3x more persistent.  Set to 1.0 to disable if GPU code changes.
ROT_DIFF_NOISE_CORRECTION = 3.0

# ---------------------------------------------------------------------------
# Control targets (from optimizer/reference_data/target_cell_speed_control.csv)
# ---------------------------------------------------------------------------
target_vmean = [0.0011, 0.0015, 0.0012]
target_veff  = [0.00038, 0.00063, 0.000335]

# Current best_params_control.json (optimized at T=200, dt=1)
old_speed_ref = [0.00041817020062396415, 0.0006199050301202626, 0.0004034913399763545]
old_bf        = [2.5189218869227887, 2.3736333480245912, 3.116570743208639]
old_drot      = [0.001, 0.001, 0.001]  # default value (not in json)


def g_func(T, tau):
    """g(T) = T - tau*(1-exp(-T/tau))"""
    ratio = T / tau
    if ratio > 500:
        return T - tau  # exp term negligible
    return T - tau * (1.0 - math.exp(-ratio))


def compute_msd(p, b, D_rot, T, dt):
    tau = ROT_DIFF_NOISE_CORRECTION / (2.0 * D_rot)
    g = g_func(T, tau)
    msd_persistent = 2.0 * tau * g * p**2
    msd_brownian = dt * T * b**2
    return msd_persistent, msd_brownian


def predict(speed_ref, bf, D_rot, T, dt):
    """Predict vmean and veff from model parameters."""
    p = speed_ref
    b = speed_ref * bf
    vmean = math.sqrt(p**2 + b**2)
    msd_p, msd_b = compute_msd(p, b, D_rot, T, dt)
    veff = math.sqrt(msd_p + msd_b) / T
    tau = ROT_DIFF_NOISE_CORRECTION / (2.0 * D_rot)
    return {
        'vmean': vmean, 'veff': veff,
        'p': p, 'b': b, 'tau': tau,
        'msd_persistent': msd_p, 'msd_brownian': msd_b,
        'msd_total': msd_p + msd_b,
    }


def solve_for_drot(target_vm, target_ve, D_rot, T, dt):
    """
    Given targets (vmean, veff), D_rot, T, and dt, solve for (speed_ref, BF).

    From:
      vmean^2 = p^2 + b^2
      (veff*T)^2 = 2*tau*g*p^2 + dt*T*b^2

    Substituting b^2 = vmean^2 - p^2:
      (veff*T)^2 = p^2*(2*tau*g - dt*T) + dt*T*vmean^2
      p^2 = [(veff*T)^2 - dt*T*vmean^2] / [2*tau*g - dt*T]
    """
    tau = ROT_DIFF_NOISE_CORRECTION / (2.0 * D_rot)
    g = g_func(T, tau)
    coeff = 2.0 * tau * g

    numerator = (target_ve * T)**2 - dt * T * target_vm**2
    denominator = coeff - dt * T

    if denominator == 0:
        return None, "denominator = 0 (degenerate case)"

    p2 = numerator / denominator
    if p2 < 0:
        return None, f"p^2 = {p2:.2e} < 0 (no real solution)"

    b2 = target_vm**2 - p2
    if b2 < 0:
        return None, f"b^2 = {b2:.2e} < 0 (persistent speed exceeds vmean)"

    p = math.sqrt(p2)
    b = math.sqrt(b2)
    speed_ref = p
    bf = b / p if p > 1e-15 else float('inf')

    # Verify
    pred = predict(speed_ref, bf, D_rot, T, dt)
    vmean_err = abs(pred['vmean'] - target_vm) / target_vm * 100
    veff_err = abs(pred['veff'] - target_ve) / target_ve * 100

    return {
        'speed_ref': speed_ref,
        'bf': bf,
        'D_rot': D_rot,
        'tau': tau,
        'p': p,
        'b': b,
        'vmean_pred': pred['vmean'],
        'veff_pred': pred['veff'],
        'vmean_err': vmean_err,
        'veff_err': veff_err,
        'msd_p_frac': pred['msd_persistent'] / pred['msd_total'] * 100,
    }, None


# ============================================================================
# PART 1: Why do old parameters fail at the target T?
# ============================================================================
print('=' * 85)
print(f'PART 1: WHY OLD PARAMETERS FAIL AT T={T_target:.0f}s ({T_target/3600:.1f}h)')
print('=' * 85)

configs = [
    ('T=200s, dt=1s (original optimization)',                          200.0,       1.0),
    (f'T={T_target:.0f}s, dt=1s ({T_target/3600:.1f}h, same dt)',     T_target,    1.0),
    (f'T={T_target:.0f}s, dt={dt_target:.0f}s ({T_target/3600:.1f}h, target dt)', T_target, dt_target),
]

for label, T, dt in configs:
    print(f'\n--- {label} ---')
    for i in range(3):
        pred = predict(old_speed_ref[i], old_bf[i], old_drot[i], T, dt)
        vm_err = (pred['vmean'] - target_vmean[i]) / target_vmean[i] * 100
        ve_err = (pred['veff'] - target_veff[i]) / target_veff[i] * 100
        print(f'  Type {i}: vmean={pred["vmean"]:.6f} (target {target_vmean[i]:.6f}, '
              f'err {vm_err:+.1f}%)  |  veff={pred["veff"]:.6f} (target {target_veff[i]:.6f}, '
              f'err {ve_err:+.1f}%)')

print()
print('  >> vmean is time-independent, so it stays correct.')
print('  >> veff drops dramatically at longer T because the persistent random walk')
print('     transitions from ballistic (T < tau) to diffusive (T >> tau) regime.')
print('  >> Changing dt also affects the Brownian MSD contribution (scales as dt*T*b^2).')


# ============================================================================
# PART 2: Physical constraints on D_rot from the veff/vmean ratio
# ============================================================================
print()
print('=' * 85)
print('PART 2: UNDERSTANDING THE CONSTRAINTS')
print('=' * 85)

print(f'\nTarget simulation: T={T_target:.0f}s ({T_target/3600:.1f}h), dt={dt_target:.0f}s')
print(f'Number of steps: {T_target/dt_target:.0f}')
print()

for i in range(3):
    ratio = target_veff[i] / target_vmean[i]
    print(f'  Type {i}: veff/vmean = {ratio:.4f}  '
          f'(vmean={target_vmean[i]:.6f}, veff={target_veff[i]:.6f})')

print()
print('  For T >> tau (diffusive regime): veff ~= sqrt(p^2/D_rot + dt*b^2) / sqrt(T)')
print('  This sets a LOWER bound on D_rot: higher D_rot -> lower veff.')
print('  But D_rot too low -> veff too high (can\'t reduce it).')


# ============================================================================
# PART 3: Per-type D_rot feasibility analysis
# ============================================================================
print()
print('=' * 85)
print(f'PART 3: PER-TYPE D_rot FEASIBILITY FOR T={T_target:.0f}s, dt={dt_target:.0f}s')
print('=' * 85)

print('\n  Each cell type may need a DIFFERENT D_rot. The optimizer already allows')
print('  per-type D_rot, so we solve each type independently.')
print()

drot_sweep = [d * 1e-5 for d in range(1, 51)] + \
             [d * 1e-4 for d in range(6, 51)] + \
             [d * 1e-3 for d in range(6, 21)]

per_type_feasible = {i: [] for i in range(3)}

for i in range(3):
    print(f'  --- Cell type {i} (vmean={target_vmean[i]:.6f}, veff={target_veff[i]:.6f}) ---')
    print(f'    {"D_rot":>10s}  {"tau":>7s}  {"speed_ref":>10s}  {"BF":>8s}  {"status"}')
    for D_rot in drot_sweep:
        sol, err = solve_for_drot(target_vmean[i], target_veff[i], D_rot, T_target, dt_target)
        if sol is None:
            continue
        sr_ok = 0.00005 <= sol['speed_ref'] <= 0.005
        bf_ok = 0 <= sol['bf'] <= 10
        if sr_ok and bf_ok:
            per_type_feasible[i].append((D_rot, sol))
            tau_eff = ROT_DIFF_NOISE_CORRECTION / (2.0 * D_rot)
            if len(per_type_feasible[i]) <= 6 or len(per_type_feasible[i]) % 5 == 0:
                print(f'    {D_rot:10.5f}  {tau_eff:7.0f}  {sol["speed_ref"]:10.6f}  {sol["bf"]:8.4f}  OK'
                      f'  ({sol["msd_p_frac"]:.0f}% pers)')
    if per_type_feasible[i]:
        lo = per_type_feasible[i][0][0]
        hi = per_type_feasible[i][-1][0]
        print(f'    Feasible D_rot range: [{lo:.6f}, {hi:.6f}]')
    else:
        print(f'    *** NO FEASIBLE D_rot FOUND - targets may be unreachable ***')
    print()


# ============================================================================
# PART 4: Find best per-type solutions — all types must be feasible at SOME D_rot
# ============================================================================
print()
print('=' * 85)
print('PART 4: BEST PER-TYPE ANALYTICAL SOLUTIONS')
print('=' * 85)

best_per_type = []
for i in range(3):
    if not per_type_feasible[i]:
        print(f'\n  Type {i}: NO FEASIBLE SOLUTION')
        best_per_type.append(None)
        continue

    # Pick D_rot that yields the most moderate BF and speed_ref
    best_score_i = float('inf')
    best_sol_i = None
    best_drot_i = None
    for D_rot, sol in per_type_feasible[i]:
        score = (sol['bf'] - 2.0)**2 + (math.log10(sol['speed_ref']) + 3.3)**2
        if score < best_score_i:
            best_score_i = score
            best_sol_i = sol
            best_drot_i = D_rot

    best_per_type.append(best_sol_i)
    s = best_sol_i
    tau_eff = ROT_DIFF_NOISE_CORRECTION / (2.0 * best_drot_i)
    print(f'\n  Cell type {i} (best D_rot = {best_drot_i:.6f}, tau_eff = {tau_eff:.0f}s):')
    print(f'    CELL_SPEED_REF[{i}]                  = {s["speed_ref"]}')
    print(f'    BROWNIAN_MOTION_STRENGTH_FACTOR[{i}]  = {s["bf"]}')
    print(f'    ROTATIONAL_DIFFUSION_RATE[{i}]        = {best_drot_i}')
    print(f'    persistent_speed  = {s["p"]:.6f} um/s')
    print(f'    brownian_strength = {s["b"]:.6f} um/s')
    print(f'    vmean = {s["vmean_pred"]:.6f}  (target {target_vmean[i]:.6f},  err {s["vmean_err"]:.6f}%)')
    print(f'    veff  = {s["veff_pred"]:.6f}  (target {target_veff[i]:.6f},  err {s["veff_err"]:.6f}%)')
    print(f'    MSD: {s["msd_p_frac"]:.1f}% persistent, {100-s["msd_p_frac"]:.1f}% brownian')


# ============================================================================
# PART 5: Recommended JSON and optimizer YAML ranges
# ============================================================================
print()
print('=' * 85)
print('PART 5: RECOMMENDED OPTIMIZER CONFIGURATION')
print('=' * 85)

all_feasible = all(s is not None for s in best_per_type)

if all_feasible:
    print(f'\n  JSON snippet for control simulation (T={T_target:.0f}s, {T_target/3600:.1f}h):')
    print('  {')
    print(f'    "STEPS": {int(T_target/dt_target)},')
    print(f'    "TIME_STEP": {dt_target},')
    for i in range(3):
        print(f'    "CELL_SPEED_REF[{i}]": {best_per_type[i]["speed_ref"]},')
    for i in range(3):
        print(f'    "BROWNIAN_MOTION_STRENGTH_FACTOR[{i}]": {best_per_type[i]["bf"]},')
    for i in range(3):
        print(f'    "ROTATIONAL_DIFFUSION_RATE[{i}]": {best_per_type[i]["D_rot"]},')
    print('    ...')
    print('  }')

    print()
    print('  Suggested optimizer YAML parameter ranges (widened):')
    print()
    for pname, key in [('CELL_SPEED_REF', 'speed_ref'),
                       ('BROWNIAN_MOTION_STRENGTH_FACTOR', 'bf'),
                       ('ROTATIONAL_DIFFUSION_RATE', 'D_rot')]:
        print(f'  {pname}:')
        print(f'    type: array_float')
        print(f'    elements:')
        for i in range(3):
            val = best_per_type[i][key]
            if key == 'bf':
                lo = max(0.0, val / 3.0)
                hi = min(10.0, val * 3.0)
                print(f'      {i}: {{low: {lo:.2f}, high: {hi:.2f}}}')
            else:
                lo = val / 5.0
                hi = val * 5.0
                is_log = key in ('speed_ref', 'D_rot')
                print(f'      {i}: {{low: {lo:.6f}, high: {hi:.6f}}}')
        if key in ('speed_ref', 'D_rot'):
            print(f'    log: true')
        print()
else:
    print('\n  WARNING: Not all types have feasible solutions.')
    print(f'  The target veff/vmean ratios may be unreachable at T={T_target:.0f}s, dt={dt_target:.0f}s.')
    print('  Consider:')
    print('    1. Relaxing veff targets (reduce the ratio veff/vmean)')
    print('    2. Using smaller dt (reduces Brownian MSD contribution)')
    print('    3. Increasing the simulation dt to reduce Brownian noise impact')


# ============================================================================
# PART 6: Cross-check — predictions at different T
# ============================================================================
print()
print('=' * 85)
print('PART 6: CROSS-CHECK - NEW PARAMS EVALUATED AT DIFFERENT T')
print('=' * 85)

if not all_feasible:
    print('  Skipped (some types infeasible).')
else:
    check_T_values = [200.0, 3600.0, T_target, T_target * 2]
    check_configs = []
    for T_chk in check_T_values:
        label = f'T={T_chk:.0f}s, dt={dt_target:.0f}s ({T_chk/3600:.1f}h)'
        check_configs.append((label, T_chk, dt_target))

    for label, T, dt in check_configs:
        print(f'\n  --- {label} ---')
        for i in range(3):
            s = best_per_type[i]
            pred = predict(s['speed_ref'], s['bf'], s['D_rot'], T, dt)
            ve_err = (pred['veff'] - target_veff[i]) / target_veff[i] * 100
            print(f'    Type {i}: vmean={pred["vmean"]:.6f}  veff={pred["veff"]:.6f} '
                  f'(target {target_veff[i]:.6f}, err {ve_err:+.1f}%)')


# ============================================================================
# PART 7: Sensitivity — how much does veff change with T?
# ============================================================================
print()
print('=' * 85)
print('PART 7: SENSITIVITY - veff vs. OBSERVATION TIME T')
print('=' * 85)

if all_feasible:
    T_values = sorted(set([60, 200, 600, 1800, 3600, 7200, 14400,
                           int(T_target/2), int(T_target), int(T_target*2)]))
    print(f'\n  {"T(s)":>7s} {"T(h)":>5s}', end='')
    for i in range(3):
        print(f'  veff_{i}   err_{i}', end='')
    print()
    for T in T_values:
        print(f'  {T:7d} {T/3600:5.1f}', end='')
        for i in range(3):
            s = best_per_type[i]
            pred = predict(s['speed_ref'], s['bf'], s['D_rot'], T, dt_target)
            err = (pred['veff'] - target_veff[i]) / target_veff[i] * 100
            print(f'  {pred["veff"]:.5f} {err:+5.0f}%', end='')
        print()
    print()
    print('  >> veff depends strongly on T. Parameters are only exact at the target T.')
    print('  >> This is inherent to persistent random walks: veff ~ 1/sqrt(T) at long T.')
