import math

# Fixed control parameters
speed_ref = [0.00039467864789234224, 0.0006043654028727149, 0.00032985670642352277]

# TGFb targets
tgfb_vmean = [0.0013, 0.0013, 0.0013]
tgfb_veff  = [0.0004, 0.00037, 0.00039]

# With constant C=2.5, SENSITIVITY=[1,0], SAT=50, SAT_MULT=1, ADAPT_TAU=1e6:
# signal_sat = saturate(1.0*2.5, 50.0) = 50*2.5/(50+2.5) = 2.381
# adapt_state after 200 steps ~ 0 (tau=1e6)
# h_promo = hill01(2.381, K=2.0, n=2.0) = 2.381^2/(2^2+2.381^2) = 0.586
# chemokinesis_factor = 1 + ALPHA * 0.586
h_const = 2.381**2 / (2.0**2 + 2.381**2)
print(f'h_promo (constant, all types) = {h_const:.4f}')
print(f'chemokinesis_factor range = [{1-h_const:.3f}, {1+h_const:.3f}]')
print()

T = 200.0  # total sim time
dt = 1.0   # timestep

def compute_solution(type_i, D_rot):
    tau = 1.0 / (2.0 * D_rot)
    # g(tau) = T - tau*(1-exp(-T/tau))
    g = T - tau * (1.0 - math.exp(-T / tau))
    coeff = 2.0 * tau * g  # MSD_persistent / p^2

    # From veff target:
    # sqrt(coeff*p^2 + dt*T*b^2) = veff*T
    # coeff*p^2 + dt*T*b^2 = (veff*T)^2
    # But b^2 = vmean^2 - p^2 (from vmean equation)
    # coeff*p^2 + dt*T*(vmean^2 - p^2) = (veff*T)^2
    # p^2*(coeff - dt*T) = (veff*T)^2 - dt*T*vmean^2

    vm = tgfb_vmean[type_i]
    ve = tgfb_veff[type_i]
    
    rhs = (ve * T)**2 - dt * T * vm**2
    lhs_coeff = coeff - dt * T
    
    if lhs_coeff <= 0:
        return None
    
    p2 = rhs / lhs_coeff
    if p2 < 0:
        return None
    p = math.sqrt(p2)
    
    b2 = vm**2 - p2
    if b2 < 0:
        return None
    b = math.sqrt(b2)
    
    alpha = (p / speed_ref[type_i] - 1.0) / h_const
    bf = b / speed_ref[type_i]
    
    # Verify
    factor = 1.0 + alpha * h_const
    persistent = speed_ref[type_i] * factor
    brownian = speed_ref[type_i] * bf
    vmean_check = math.sqrt(persistent**2 + brownian**2)
    msd_p = coeff * persistent**2
    msd_b = dt * T * brownian**2
    veff_check = math.sqrt(msd_p + msd_b) / T
    
    return {
        'alpha': alpha, 'bf': bf, 'D_rot': D_rot,
        'persistent': persistent, 'brownian': brownian,
        'vmean_check': vmean_check, 'veff_check': veff_check,
        'vmean_err': abs(vmean_check - vm)/vm*100,
        'veff_err': abs(veff_check - ve)/ve*100,
    }

print('=' * 80)
print('ANALYTICAL SOLUTIONS (approximate) for different D_rot values')
print('=' * 80)

for D_rot in [0.003, 0.005, 0.007, 0.01]:
    print(f'\\n--- D_rot = {D_rot} (tau = {1/(2*D_rot):.0f}s) ---')
    all_valid = True
    for i in range(3):
        sol = compute_solution(i, D_rot)
        if sol is None:
            print(f'  Type {i}: NO SOLUTION at this D_rot')
            all_valid = False
            continue
        if abs(sol['alpha']) > 1.0:
            flag = ' *** ALPHA OUT OF RANGE [-1,1] ***'
            all_valid = False
        elif sol['bf'] > 10.0 or sol['bf'] < 0:
            flag = ' *** BF OUT OF RANGE [0,10] ***'
            all_valid = False
        else:
            flag = ' OK'
        print(f'  Type {i}: ALPHA={sol["alpha"]:+.4f}, BF={sol["bf"]:.3f}, '
              f'vmean={sol["vmean_check"]:.6f}, veff={sol["veff_check"]:.6f}{flag}')
    if all_valid:
        print(f'  >>> ALL TYPES FEASIBLE at D_rot={D_rot} <<<')

print()
print('=' * 80)
print('BEST SOLUTION (D_rot=0.005 for all types)')
print('=' * 80)
for i in range(3):
    sol = compute_solution(i, 0.005)
    print(f'  Type {i}:')
    print(f'    ALPHA = {sol["alpha"]:+.4f}')
    print(f'    BROWNIAN_MOTION_STRENGTH_FACTOR = {sol["bf"]:.4f}')
    print(f'    ROTATIONAL_DIFFUSION_RATE = 0.005')
    print(f'    persistent_speed = {sol["persistent"]:.6f} (factor={1+sol["alpha"]*h_const:.3f})')
    print(f'    brownian_strength = {sol["brownian"]:.6f}')
    print(f'    vmean = {sol["vmean_check"]:.6f} (target {tgfb_vmean[i]:.6f})')
    print(f'    veff  = {sol["veff_check"]:.6f} (target {tgfb_veff[i]:.6f})')
    print()

# Now check: what does the optimizer see if it tries with control BF values?
print('=' * 80)
print('Check control BF values (not free)')
print('=' * 80)
control_bf = [2.7345, 2.3718, 3.5305]
for i in range(3):
    b = speed_ref[i] * control_bf[i]
    vm = tgfb_vmean[i]
    p_needed_sq = vm**2 - b**2
    if p_needed_sq < 0:
        print(f'  Type {i}: brownian={b:.6f} > target_vmean={vm:.6f} -> IMPOSSIBLE')
    else:
        p_needed = math.sqrt(p_needed_sq)
        alpha_needed = (p_needed / speed_ref[i] - 1.0) / h_const
        print(f'  Type {i}: need p={p_needed:.6f}, ALPHA={alpha_needed:+.4f}', end='')
        if abs(alpha_needed) > 1:
            print(' *** OUT OF RANGE ***')
        else:
            print(' OK')