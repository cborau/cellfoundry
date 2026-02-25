/**
 * clampf
 *
 * Clamps a scalar to the closed interval [lo, hi].
 */
FLAMEGPU_DEVICE_FUNCTION float clampf(const float x, const float lo, const float hi) {
  return fminf(hi, fmaxf(lo, x));
}

/**
 * safeInv
 *
 * Returns 1/x when |x| > eps, otherwise returns 0.
 * Used to avoid division by near-zero values in device code.
 */
FLAMEGPU_DEVICE_FUNCTION float safeInv(const float x, const float eps) {
  return (fabsf(x) > eps) ? (1.0f / x) : 0.0f;
}

/**
 * normalize3
 *
 * Normalizes a 3D vector in-place; if near-zero, sets a default unit vector.
 */
FLAMEGPU_DEVICE_FUNCTION void normalize3(float &x, float &y, float &z) {
  const float n2 = x*x + y*y + z*z;
  if (n2 > 1e-20f) {
    const float inv = rsqrtf(n2);
    x *= inv;
    y *= inv;
    z *= inv;
  } else {
    x = 1.0f;
    y = 0.0f;
    z = 0.0f;
  }
}

/**
 * swapf
 *
 * Swaps two floats by reference.
 */
FLAMEGPU_DEVICE_FUNCTION void swapf(float &a, float &b) {
  const float t = a;
  a = b;
  b = t;
}

/**
 * swap_col3
 *
 * Swaps two columns of a 3x3 matrix (used for eigenvector column reordering).
 */
FLAMEGPU_DEVICE_FUNCTION void swap_col3(float V[3][3], const int c1, const int c2) {
  swapf(V[0][c1], V[0][c2]);
  swapf(V[1][c1], V[1][c2]);
  swapf(V[2][c1], V[2][c2]);
}

/**
 * eig_sym_3x3
 *
 * Jacobi eigendecomposition for a real symmetric 3x3 matrix:
 *   [a00 a01 a02]
 *   [a01 a11 a12]
 *   [a02 a12 a22]
 *
 * Outputs:
 *   - Eigenvalues l1, l2, l3
 *   - Corresponding eigenvectors (v1, v2, v3)
 *
 * Ordering contract:
 *   Eigenpairs are sorted in descending eigenvalue order:
 *     l1 >= l2 >= l3
 *   and vector i corresponds to li.
 */
FLAMEGPU_DEVICE_FUNCTION void eig_sym_3x3(
  const float a00, const float a01, const float a02,
  const float a11, const float a12, const float a22,
  float &l1, float &l2, float &l3,
  float &v1x, float &v1y, float &v1z,
  float &v2x, float &v2y, float &v2z,
  float &v3x, float &v3y, float &v3z) {

  float A[3][3] = {
    {a00, a01, a02},
    {a01, a11, a12},
    {a02, a12, a22}
  };

  float V[3][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f}
  };

  const int MAX_ITERS = 10;
  for (int it = 0; it < MAX_ITERS; ++it) {
    int p = 0, q = 1;
    float max_off = fabsf(A[0][1]);

    const float a02_abs = fabsf(A[0][2]);
    if (a02_abs > max_off) {
      max_off = a02_abs;
      p = 0; q = 2;
    }
    const float a12_abs = fabsf(A[1][2]);
    if (a12_abs > max_off) {
      max_off = a12_abs;
      p = 1; q = 2;
    }

    if (max_off < 1e-10f) {
      break;
    }

    const float app = A[p][p];
    const float aqq = A[q][q];
    const float apq = A[p][q];

    const float tau = (aqq - app) / (2.0f * apq);
    const float t = (tau >= 0.0f)
      ? (1.0f / (tau + sqrtf(1.0f + tau * tau)))
      : (-1.0f / (-tau + sqrtf(1.0f + tau * tau)));
    const float c = 1.0f / sqrtf(1.0f + t * t);
    const float s = t * c;

    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0.0f;
    A[q][p] = 0.0f;

    for (int r = 0; r < 3; ++r) {
      if (r == p || r == q) continue;
      const float arp = A[r][p];
      const float arq = A[r][q];
      A[r][p] = c * arp - s * arq;
      A[p][r] = A[r][p];
      A[r][q] = s * arp + c * arq;
      A[q][r] = A[r][q];
    }

    for (int r = 0; r < 3; ++r) {
      const float vrp = V[r][p];
      const float vrq = V[r][q];
      V[r][p] = c * vrp - s * vrq;
      V[r][q] = s * vrp + c * vrq;
    }
  }

  float eval[3] = {A[0][0], A[1][1], A[2][2]};

  if (eval[0] < eval[1]) {
    swapf(eval[0], eval[1]);
    swap_col3(V, 0, 1);
  }
  if (eval[0] < eval[2]) {
    swapf(eval[0], eval[2]);
    swap_col3(V, 0, 2);
  }
  if (eval[1] < eval[2]) {
    swapf(eval[1], eval[2]);
    swap_col3(V, 1, 2);
  }

  l1 = eval[0];
  l2 = eval[1];
  l3 = eval[2];

  v1x = V[0][0]; v1y = V[1][0]; v1z = V[2][0];
  v2x = V[0][1]; v2y = V[1][1]; v2z = V[2][1];
  v3x = V[0][2]; v3y = V[1][2]; v3z = V[2][2];
}

/**
 * cell_focad_update
 *
 * Reads all focal adhesion (FOCAD) messages in a bucket keyed by this cell id.
 * Each message provides:
 *   - anchor position on the nucleus surface: (x_i, y_i, z_i)   [um]
 *   - traction / pulling force at that anchor: (fx, fy, fz)     [nN]
 *
 * The cell then:
 *   1) Accumulates the symmetric stresslet tensor sum_i sym(r_i ⊗ f_i)
 *   2) Converts it into an average nucleus stress sigma = (1/V) * stresslet
 *   3) Computes elastic strain eps_el from sigma using isotropic linear compliance
 *   4) Projects eps_el to deviatoric (nearly incompressible) strain eps_tilde
 *   5) Updates the stored strain eps via first-order relaxation with time constant tau
 *   6) Clamps eps components to stay in the intended small-strain regime
 *   7) Updates all stored nucleus anchor positions using x_i = x + R * (I + eps) * u_ref_i
 *
 * Units:
 *   length: um
 *   force:  nN
 *   stress: nN/um^2 [kPa]
 */
FLAMEGPU_AGENT_FUNCTION(cell_focad_update, flamegpu::MessageBucket, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    return flamegpu::ALIVE; // Note: if DEAD_CELLS_DISAPPEAR = True, a dead CELL agent remains ALIVE for flamegpu purposes and may still interact with other agents.
  }
  // -------------------------
  // Read CELL agent state
  // -------------------------
  const int agent_id = FLAMEGPU->getVariable<int>("id");

  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_focad_birth_cooldown = FLAMEGPU->getVariable<float>("focad_birth_cooldown");
  float agent_orx = FLAMEGPU->getVariable<float>("orx");
  float agent_ory = FLAMEGPU->getVariable<float>("ory");
  float agent_orz = FLAMEGPU->getVariable<float>("orz");

  const float agent_radius = FLAMEGPU->getVariable<float>("radius");
  const float agent_nucleus_radius = FLAMEGPU->getVariable<float>("nucleus_radius");

  // Stored viscoelastic strain state (symmetric small strain tensor)
  float agent_eps_xx = FLAMEGPU->getVariable<float>("eps_xx");
  float agent_eps_yy = FLAMEGPU->getVariable<float>("eps_yy");
  float agent_eps_zz = FLAMEGPU->getVariable<float>("eps_zz");
  float agent_eps_xy = FLAMEGPU->getVariable<float>("eps_xy");
  float agent_eps_xz = FLAMEGPU->getVariable<float>("eps_xz");
  float agent_eps_yz = FLAMEGPU->getVariable<float>("eps_yz");

  // -------------------------
  // Material and numerical parameters (environment)
  // -------------------------
  // Note: With nN and um, modulus and stress are in nN/um^2, numerically equal to kPa.
  const float NUCLEUS_E         = FLAMEGPU->environment.getProperty<float>("NUCLEUS_E");         // [kPa] = [nN/um^2]
  const float NUCLEUS_NU        = FLAMEGPU->environment.getProperty<float>("NUCLEUS_NU");        // [-]
  const float NUCLEUS_TAU       = FLAMEGPU->environment.getProperty<float>("NUCLEUS_TAU");       // [s]
  const float NUCLEUS_EPS_CLAMP = FLAMEGPU->environment.getProperty<float>("NUCLEUS_EPS_CLAMP"); // [-]
  const float TIME_STEP         = FLAMEGPU->environment.getProperty<float>("TIME_STEP");         // [s]
  const int INCLUDE_ORIENTATION_ALIGN = FLAMEGPU->environment.getProperty<int>("INCLUDE_ORIENTATION_ALIGN");
  const float ORIENTATION_ALIGN_RATE = FLAMEGPU->environment.getProperty<float>("ORIENTATION_ALIGN_RATE");
  const int ORIENTATION_ALIGN_USE_STRESS = FLAMEGPU->environment.getProperty<int>("ORIENTATION_ALIGN_USE_STRESS");
  const uint32_t ENABLE_FOCAD_BIRTH = FLAMEGPU->environment.getProperty<uint32_t>("ENABLE_FOCAD_BIRTH");
  const uint32_t FOCAD_BIRTH_SPECIES_INDEX = FLAMEGPU->environment.getProperty<uint32_t>("FOCAD_BIRTH_SPECIES_INDEX");
  const uint32_t FOCAD_BIRTH_N_MIN = FLAMEGPU->environment.getProperty<uint32_t>("FOCAD_BIRTH_N_MIN");
  const uint32_t FOCAD_BIRTH_N_MAX = FLAMEGPU->environment.getProperty<uint32_t>("FOCAD_BIRTH_N_MAX");
  const float FOCAD_BIRTH_K_0 = FLAMEGPU->environment.getProperty<float>("FOCAD_BIRTH_K_0");
  const float FOCAD_BIRTH_K_MAX = FLAMEGPU->environment.getProperty<float>("FOCAD_BIRTH_K_MAX");
  const float FOCAD_BIRTH_K_SIGMA = FLAMEGPU->environment.getProperty<float>("FOCAD_BIRTH_K_SIGMA");
  const float FOCAD_BIRTH_HILL_SIGMA = FLAMEGPU->environment.getProperty<float>("FOCAD_BIRTH_HILL_SIGMA");
  const float FOCAD_BIRTH_K_C = FLAMEGPU->environment.getProperty<float>("FOCAD_BIRTH_K_C");
  const float FOCAD_BIRTH_HILL_CONC = FLAMEGPU->environment.getProperty<float>("FOCAD_BIRTH_HILL_CONC");
  const float FOCAD_BIRTH_REFRACTORY = FLAMEGPU->environment.getProperty<float>("FOCAD_BIRTH_REFRACTORY");
  const float FOCAD_REST_LENGTH_0 = FLAMEGPU->environment.getProperty<float>("FOCAD_REST_LENGTH_0");
  const float FOCAD_K_FA = FLAMEGPU->environment.getProperty<float>("FOCAD_K_FA");
  const float FOCAD_F_MAX = FLAMEGPU->environment.getProperty<float>("FOCAD_F_MAX");
  const float FOCAD_V_C = FLAMEGPU->environment.getProperty<float>("FOCAD_V_C");
  const float FOCAD_K_ON = FLAMEGPU->environment.getProperty<float>("FOCAD_K_ON");
  const float FOCAD_K_OFF_0 = FLAMEGPU->environment.getProperty<float>("FOCAD_K_OFF_0");
  const float FOCAD_F_C = FLAMEGPU->environment.getProperty<float>("FOCAD_F_C");
  const float FOCAD_K_REINF = FLAMEGPU->environment.getProperty<float>("FOCAD_K_REINF");

  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  // -------------------------
  // Accumulate stresslet S = sum_i sym(r_i ⊗ f_i)
  // where r_i = x_i - x_c
  // -------------------------
  float agent_S_xx = 0.0f, agent_S_yy = 0.0f, agent_S_zz = 0.0f;
  float agent_S_xy = 0.0f, agent_S_xz = 0.0f, agent_S_yz = 0.0f;
  uint32_t current_focad_count = 0;

  // Iterate over all FOCAD messages addressed to this cell id
  for (const auto &message : FLAMEGPU->message_in(agent_id)) {
    current_focad_count += 1;
    // Anchor position on nucleus and force at the anchor (message variables)
    const float message_x_i = message.getVariable<float>("x_i"); // [um]
    const float message_y_i = message.getVariable<float>("y_i"); // [um]
    const float message_z_i = message.getVariable<float>("z_i"); // [um]

    const float message_fx  = message.getVariable<float>("fx");  // [nN]
    const float message_fy  = message.getVariable<float>("fy");  // [nN]
    const float message_fz  = message.getVariable<float>("fz");  // [nN]

    //printf("cell_update_stress -- FOCAD message for CELL %d: anchor=(%.4f, %.4f, %.4f) um, force=(%.4f, %.4f, %.4f) nN\n", agent_id, message_x_i, message_y_i, message_z_i, message_fx, message_fy, message_fz);

    // Lever arm from nucleus center to adhesion location
    const float agent_rx = message_x_i - agent_x; // [um]
    const float agent_ry = message_y_i - agent_y; // [um]
    const float agent_rz = message_z_i - agent_z; // [um]

    // Symmetric stresslet contributions
    agent_S_xx += agent_rx * message_fx;                                  
    agent_S_yy += agent_ry * message_fy;                                  
    agent_S_zz += agent_rz * message_fz;                                  
    agent_S_xy += 0.5f * (agent_rx * message_fy + agent_ry * message_fx); 
    agent_S_xz += 0.5f * (agent_rx * message_fz + agent_rz * message_fx);
    agent_S_yz += 0.5f * (agent_ry * message_fz + agent_rz * message_fy);
    // printf("cell_update_stress -- message: r=(%.3f, %.3f, %.3f) f=(%.3f, %.3f, %.3f) S_contrib=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)\n", 
    //        agent_rx, agent_ry, agent_rz, message_fx, message_fy, message_fz,
    //        agent_S_xx,
    //        agent_S_yy,
    //        agent_S_zz,
    //        agent_S_xy,
    //        agent_S_xz,
    //        agent_S_yz);

  }

  // -------------------------
  // Average stress: sigma = (1/V) * S
  // V is nucleus volume (sphere)
  // Stress units: nN/um^2 (kPa)
  // -------------------------
  const float PI = 3.14159265358979323846f;
  const float agent_V = (4.0f / 3.0f) * PI * agent_nucleus_radius * agent_nucleus_radius * agent_nucleus_radius; // [um^3]
  const float invV = safeInv(agent_V, 1e-20f);

  const float agent_sig_xx = invV * agent_S_xx;
  const float agent_sig_yy = invV * agent_S_yy;
  const float agent_sig_zz = invV * agent_S_zz;
  const float agent_sig_xy = invV * agent_S_xy;
  const float agent_sig_xz = invV * agent_S_xz;
  const float agent_sig_yz = invV * agent_S_yz;

  //printf("cell_update_stress -- agent_sig = (%.3e, %.3e, %.3e, %.3e, %.3e, %.3e) nN/um^2\n", agent_sig_xx, agent_sig_yy, agent_sig_zz, agent_sig_xy, agent_sig_xz, agent_sig_yz);

  // -------------------------
  // Isotropic compliance inversion:
  //
  // Using isotropic Hooke’s law:
  //   sigma = 2G eps + lambda tr(eps) I
  // Inversion:
  //   eps = (1/(2G)) sigma - alpha tr(sigma) I
  // where:
  //   G = E / (2(1+nu))
  //   lambda = E nu / ((1+nu)(1-2nu))
  //   alpha = lambda / (2G(3lambda + 2G))
  //
  // Note:
  //   eps_xy here is the tensor shear strain component, not engineering shear gamma_xy (= 2 * eps_xy).
  // -------------------------
  const float nu = clampf(NUCLEUS_NU, 0.0f, 0.499f);
  const float E  = fmaxf(NUCLEUS_E, 1e-12f);

  const float G = E / (2.0f * (1.0f + nu));
  const float lambda = (E * nu) / ((1.0f + nu) * (1.0f - 2.0f * nu));

  const float tr_sig = agent_sig_xx + agent_sig_yy + agent_sig_zz;

  const float denom = (3.0f * lambda + 2.0f * G);
  const float alpha = (denom > 0.0f) ? (lambda / (2.0f * G * denom)) : 0.0f;
  const float inv2G = 1.0f / (2.0f * G);

  float eps_el_xx = inv2G * agent_sig_xx - alpha * tr_sig;
  float eps_el_yy = inv2G * agent_sig_yy - alpha * tr_sig;
  float eps_el_zz = inv2G * agent_sig_zz - alpha * tr_sig;

  float eps_el_xy = inv2G * agent_sig_xy;
  float eps_el_xz = inv2G * agent_sig_xz;
  float eps_el_yz = inv2G * agent_sig_yz;

  // -------------------------
  // Nearly incompressible projection:
  // Remove volumetric strain from eps_el:
  //   eps_tilde = eps_el - (tr(eps_el)/3) I
  // -------------------------
  const float tr_eps_el = eps_el_xx + eps_el_yy + eps_el_zz;
  const float m = tr_eps_el / 3.0f;

  const float agent_eps_tilde_xx = eps_el_xx - m;
  const float agent_eps_tilde_yy = eps_el_yy - m;
  const float agent_eps_tilde_zz = eps_el_zz - m;
  const float agent_eps_tilde_xy = eps_el_xy;
  const float agent_eps_tilde_xz = eps_el_xz;
  const float agent_eps_tilde_yz = eps_el_yz;

  // -------------------------
  // Viscoelastic relaxation of stored strain:
  //   d eps / dt = (eps_tilde - eps) / tau
  // Explicit Euler:
  //   eps <- eps + (dt/tau) (eps_tilde - eps)
  //
  // rel = dt/tau is clamped to <= 1 for robustness.
  // -------------------------
  const float tau_s = fmaxf(NUCLEUS_TAU, 1e-6f);
  const float rel_raw = TIME_STEP / tau_s;
  const float rel = fminf(rel_raw, 1.0f); // rel = 1 
  // printf("cell_update_stress -- agent_eps = (%.3e, %.3e, %.3e, %.3e, %.3e, %.3e) eps_tilde=(%.3e, %.3e, %.3e, %.3e, %.3e, %.3e) rel=%.3f\n", agent_eps_xx, agent_eps_yy, agent_eps_zz, agent_eps_xy, agent_eps_xz, agent_eps_yz,
  //                                                                                                                       agent_eps_tilde_xx, agent_eps_tilde_yy, agent_eps_tilde_zz, agent_eps_tilde_xy, agent_eps_tilde_xz, agent_eps_tilde_yz, rel);

  agent_eps_xx += rel * (agent_eps_tilde_xx - agent_eps_xx);
  agent_eps_yy += rel * (agent_eps_tilde_yy - agent_eps_yy);
  agent_eps_zz += rel * (agent_eps_tilde_zz - agent_eps_zz);
  agent_eps_xy += rel * (agent_eps_tilde_xy - agent_eps_xy);
  agent_eps_xz += rel * (agent_eps_tilde_xz - agent_eps_xz);
  agent_eps_yz += rel * (agent_eps_tilde_yz - agent_eps_yz);

  // -------------------------
  // Small-strain regime enforcement
  // Clamp each component of eps to keep the kinematic update U = I + eps valid
  // and prevent runaway due to force-geometry feedback.
  // -------------------------
  agent_eps_xx = clampf(agent_eps_xx, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_yy = clampf(agent_eps_yy, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_zz = clampf(agent_eps_zz, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_xy = clampf(agent_eps_xy, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_xz = clampf(agent_eps_xz, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_yz = clampf(agent_eps_yz, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);

  // Store updated strain
  FLAMEGPU->setVariable<float>("eps_xx", agent_eps_xx);
  FLAMEGPU->setVariable<float>("eps_yy", agent_eps_yy);
  FLAMEGPU->setVariable<float>("eps_zz", agent_eps_zz);
  FLAMEGPU->setVariable<float>("eps_xy", agent_eps_xy);
  FLAMEGPU->setVariable<float>("eps_xz", agent_eps_xz);
  FLAMEGPU->setVariable<float>("eps_yz", agent_eps_yz);

  // Store stress (for visualization/debugging; not used in mechanics update)
  FLAMEGPU->setVariable<float>("sig_xx", agent_sig_xx);
  FLAMEGPU->setVariable<float>("sig_yy", agent_sig_yy);
  FLAMEGPU->setVariable<float>("sig_zz", agent_sig_zz);
  FLAMEGPU->setVariable<float>("sig_xy", agent_sig_xy);
  FLAMEGPU->setVariable<float>("sig_xz", agent_sig_xz);
  FLAMEGPU->setVariable<float>("sig_yz", agent_sig_yz);

  // Principal values/vectors (stress tensor)
  float sig_l1, sig_l2, sig_l3;
  float sig_v1x, sig_v1y, sig_v1z;
  float sig_v2x, sig_v2y, sig_v2z;
  float sig_v3x, sig_v3y, sig_v3z;
  eig_sym_3x3(
    agent_sig_xx, agent_sig_xy, agent_sig_xz,
    agent_sig_yy, agent_sig_yz, agent_sig_zz,
    sig_l1, sig_l2, sig_l3,
    sig_v1x, sig_v1y, sig_v1z,
    sig_v2x, sig_v2y, sig_v2z,
    sig_v3x, sig_v3y, sig_v3z);

  FLAMEGPU->setVariable<float>("sig_eig_1", sig_l1);
  FLAMEGPU->setVariable<float>("sig_eig_2", sig_l2);
  FLAMEGPU->setVariable<float>("sig_eig_3", sig_l3);
  FLAMEGPU->setVariable<float>("sig_eigvec1_x", sig_v1x);
  FLAMEGPU->setVariable<float>("sig_eigvec1_y", sig_v1y);
  FLAMEGPU->setVariable<float>("sig_eigvec1_z", sig_v1z);
  FLAMEGPU->setVariable<float>("sig_eigvec2_x", sig_v2x);
  FLAMEGPU->setVariable<float>("sig_eigvec2_y", sig_v2y);
  FLAMEGPU->setVariable<float>("sig_eigvec2_z", sig_v2z);
  FLAMEGPU->setVariable<float>("sig_eigvec3_x", sig_v3x);
  FLAMEGPU->setVariable<float>("sig_eigvec3_y", sig_v3y);
  FLAMEGPU->setVariable<float>("sig_eigvec3_z", sig_v3z);

  // Principal values/vectors (strain tensor)
  float eps_l1, eps_l2, eps_l3;
  float eps_v1x, eps_v1y, eps_v1z;
  float eps_v2x, eps_v2y, eps_v2z;
  float eps_v3x, eps_v3y, eps_v3z;
  eig_sym_3x3(
    agent_eps_xx, agent_eps_xy, agent_eps_xz,
    agent_eps_yy, agent_eps_yz, agent_eps_zz,
    eps_l1, eps_l2, eps_l3,
    eps_v1x, eps_v1y, eps_v1z,
    eps_v2x, eps_v2y, eps_v2z,
    eps_v3x, eps_v3y, eps_v3z);

  FLAMEGPU->setVariable<float>("eps_eig_1", eps_l1);
  FLAMEGPU->setVariable<float>("eps_eig_2", eps_l2);
  FLAMEGPU->setVariable<float>("eps_eig_3", eps_l3);
  FLAMEGPU->setVariable<float>("eps_eigvec1_x", eps_v1x);
  FLAMEGPU->setVariable<float>("eps_eigvec1_y", eps_v1y);
  FLAMEGPU->setVariable<float>("eps_eigvec1_z", eps_v1z);
  FLAMEGPU->setVariable<float>("eps_eigvec2_x", eps_v2x);
  FLAMEGPU->setVariable<float>("eps_eigvec2_y", eps_v2y);
  FLAMEGPU->setVariable<float>("eps_eigvec2_z", eps_v2z);
  FLAMEGPU->setVariable<float>("eps_eigvec3_x", eps_v3x);
  FLAMEGPU->setVariable<float>("eps_eigvec3_y", eps_v3y);
  FLAMEGPU->setVariable<float>("eps_eigvec3_z", eps_v3z);

  // ---------------------------------------------------------------------------
  // Update orientation toward max principal direction (stress or strain)
  // ---------------------------------------------------------------------------
  if (INCLUDE_ORIENTATION_ALIGN) {
    float target_x = sig_v1x;
    float target_y = sig_v1y;
    float target_z = sig_v1z;
    if (ORIENTATION_ALIGN_USE_STRESS == 0) {
      target_x = eps_v1x;
      target_y = eps_v1y;
      target_z = eps_v1z;
    }

    // Avoid sign flips: choose target with positive dot to current orientation
    const float dot = agent_orx*target_x + agent_ory*target_y + agent_orz*target_z;
    if (dot < 0.0f) {
      target_x = -target_x;
      target_y = -target_y;
      target_z = -target_z;
    }

    const float alpha = clampf(ORIENTATION_ALIGN_RATE * TIME_STEP, 0.0f, 1.0f);

    agent_orx = (1.0f - alpha) * agent_orx + alpha * target_x;
    agent_ory = (1.0f - alpha) * agent_ory + alpha * target_y;
    agent_orz = (1.0f - alpha) * agent_orz + alpha * target_z;
    normalize3(agent_orx, agent_ory, agent_orz);
  }
  FLAMEGPU->setVariable<float>("orx", agent_orx);
  FLAMEGPU->setVariable<float>("ory", agent_ory);
  FLAMEGPU->setVariable<float>("orz", agent_orz);

  // -------------------------
  // Update nucleus anchors:
  // x_i = x_c + R (I + eps) u_ref_i
  //
  // Here u_ref_* are unit vectors defining reference anchor directions on the unit sphere (fixed at agent initialization).
  // The mapping (I + eps) turns the sphere into an ellipsoid consistent with small strain.
  // -------------------------
  const float lead_x = agent_x + agent_radius * agent_orx;
  const float lead_y = agent_y + agent_radius * agent_ory;
  const float lead_z = agent_z + agent_radius * agent_orz;
  float best_anchor_x = agent_x;
  float best_anchor_y = agent_y;
  float best_anchor_z = agent_z;
  int best_anchor_id = 0;
  float best_anchor_r2 = 1e30f;

  for (unsigned int a = 0; a < N_ANCHOR_POINTS; ++a) {
    // Reference directions (CELL array variables)
    const float ux = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_x_i", a);
    const float uy = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_y_i", a);
    const float uz = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_z_i", a);

    // d = (I + eps) u
    const float dux = ux + agent_eps_xx * ux + agent_eps_xy * uy + agent_eps_xz * uz;
    const float duy = uy + agent_eps_xy * ux + agent_eps_yy * uy + agent_eps_yz * uz;
    const float duz = uz + agent_eps_xz * ux + agent_eps_yz * uy + agent_eps_zz * uz;

    // Anchor positions on the deformed nucleus surface
    const float anchor_x = agent_x + agent_nucleus_radius * dux;
    const float anchor_y = agent_y + agent_nucleus_radius * duy;
    const float anchor_z = agent_z + agent_nucleus_radius * duz;

    // Store updated anchors (CELL array variables)
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("x_i", a, anchor_x);
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("y_i", a, anchor_y);
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("z_i", a, anchor_z);

    const float dax = anchor_x - lead_x;
    const float day = anchor_y - lead_y;
    const float daz = anchor_z - lead_z;
    const float dr2 = dax * dax + day * day + daz * daz;
    if (dr2 < best_anchor_r2) {
      best_anchor_r2 = dr2;
      best_anchor_x = anchor_x;
      best_anchor_y = anchor_y;
      best_anchor_z = anchor_z;
      best_anchor_id = static_cast<int>(a);
    }
  }

  // -------------------------
  // FOCAD birth from CELL (bounded, stress+concentration gated)
  // -------------------------
  if (agent_focad_birth_cooldown > 0.0f) {
    agent_focad_birth_cooldown = fmaxf(0.0f, agent_focad_birth_cooldown - TIME_STEP);
  }

  if (ENABLE_FOCAD_BIRTH != 0 && FOCAD_BIRTH_N_MAX > 0) {
    const uint32_t idx_sp = FOCAD_BIRTH_SPECIES_INDEX < N_SPECIES ? FOCAD_BIRTH_SPECIES_INDEX : 0;
    const float c_raw = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", idx_sp);
    const float c = fmaxf(0.0f, c_raw);
    const float sigma_pos = fmaxf(0.0f, sig_l1);

    const float hill_sigma = fmaxf(1.0f, FOCAD_BIRTH_HILL_SIGMA);
    const float sigma_pow = powf(sigma_pos, hill_sigma);
    const float ks_pow = powf(fmaxf(1e-12f, FOCAD_BIRTH_K_SIGMA), hill_sigma);
    const float hs_denom = fmaxf(1e-12f, ks_pow + sigma_pow);
    const float h_sigma = sigma_pow / hs_denom;

    const float hill_conc = fmaxf(1.0f, FOCAD_BIRTH_HILL_CONC);
    const float c_pow = powf(c, hill_conc);
    const float kc_pow = powf(fmaxf(1e-12f, FOCAD_BIRTH_K_C), hill_conc);
    const float hc_denom = fmaxf(1e-12f, kc_pow + c_pow);
    const float h_c = c_pow / hc_denom;

    const uint32_t n_min = FOCAD_BIRTH_N_MIN;
    const uint32_t n_max = FOCAD_BIRTH_N_MAX;
    const float h_birth = h_sigma * h_c;
    const float target_f = static_cast<float>(n_min) + static_cast<float>(n_max - n_min) * h_birth;
    uint32_t target_n = static_cast<uint32_t>(target_f + 0.5f);
    if (target_n < n_min) target_n = n_min;
    if (target_n > n_max) target_n = n_max;

    const int can_birth = ((current_focad_count < target_n) && (current_focad_count < n_max) && (agent_focad_birth_cooldown <= 0.0f)) ? 1 : 0;
    
    if (can_birth != 0) {
      const float k_birth = fmaxf(0.0f, FOCAD_BIRTH_K_0 + FOCAD_BIRTH_K_MAX * h_birth);
      const float p_birth = 1.0f - expf(-k_birth * TIME_STEP);
      const float r_birth = FLAMEGPU->random.uniform<float>(0.0f, 1.0f);
      if (r_birth < p_birth) {
        //printf("Cell %d -- Birth FOCAD: count=%d target_n=%d sigma=%.3e h_sigma=%.3f c=%.3e h_c=%.3f h_birth=%.3f k_birth=%.3f p_birth=%.3f\n", 
        //       agent_id, current_focad_count, target_n, sigma_pos, h_sigma, c, h_c, h_birth, k_birth, p_birth);
        const int new_focad_id = FLAMEGPU->agent_out.getID();
        FLAMEGPU->agent_out.setVariable<int>("id", new_focad_id);
        FLAMEGPU->agent_out.setVariable<int>("cell_id", agent_id);
        FLAMEGPU->agent_out.setVariable<int>("fnode_id", -1);
        FLAMEGPU->agent_out.setVariable<float>("x", lead_x);
        FLAMEGPU->agent_out.setVariable<float>("y", lead_y);
        FLAMEGPU->agent_out.setVariable<float>("z", lead_z);
        FLAMEGPU->agent_out.setVariable<float>("vx", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("vy", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("vz", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("fx", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("fy", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("fz", 0.0f);
        FLAMEGPU->agent_out.setVariable<int>("anchor_id", best_anchor_id);
        FLAMEGPU->agent_out.setVariable<float>("x_i", best_anchor_x);
        FLAMEGPU->agent_out.setVariable<float>("y_i", best_anchor_y);
        FLAMEGPU->agent_out.setVariable<float>("z_i", best_anchor_z);
        FLAMEGPU->agent_out.setVariable<float>("x_c", agent_x);
        FLAMEGPU->agent_out.setVariable<float>("y_c", agent_y);
        FLAMEGPU->agent_out.setVariable<float>("z_c", agent_z);
        FLAMEGPU->agent_out.setVariable<float>("orx", agent_orx);
        FLAMEGPU->agent_out.setVariable<float>("ory", agent_ory);
        FLAMEGPU->agent_out.setVariable<float>("orz", agent_orz);
        FLAMEGPU->agent_out.setVariable<float>("rest_length_0", FOCAD_REST_LENGTH_0);
        FLAMEGPU->agent_out.setVariable<float>("rest_length", FOCAD_REST_LENGTH_0);
        FLAMEGPU->agent_out.setVariable<float>("k_fa", FOCAD_K_FA);
        FLAMEGPU->agent_out.setVariable<float>("f_max", FOCAD_F_MAX);
        FLAMEGPU->agent_out.setVariable<int>("attached", 0);
        FLAMEGPU->agent_out.setVariable<uint8_t>("active", 1);
        FLAMEGPU->agent_out.setVariable<float>("v_c", FOCAD_V_C);
        FLAMEGPU->agent_out.setVariable<uint8_t>("fa_state", 1);
        FLAMEGPU->agent_out.setVariable<float>("age", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("detached_age", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_on", FOCAD_K_ON);
        FLAMEGPU->agent_out.setVariable<float>("k_off_0", FOCAD_K_OFF_0);
        FLAMEGPU->agent_out.setVariable<float>("f_c", FOCAD_F_C);
        FLAMEGPU->agent_out.setVariable<float>("k_reinf", FOCAD_K_REINF);
        FLAMEGPU->agent_out.setVariable<float>("f_mag", 0.0f);
        FLAMEGPU->agent_out.setVariable<int>("is_front", 0);
        FLAMEGPU->agent_out.setVariable<int>("is_rear", 0);
        FLAMEGPU->agent_out.setVariable<int>("attached_front", 0);
        FLAMEGPU->agent_out.setVariable<int>("attached_rear", 0);
        FLAMEGPU->agent_out.setVariable<float>("frontness_front", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("frontness_rear", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_on_eff_front", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_on_eff_rear", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_off_0_eff_front", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_off_0_eff_rear", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("linc_prev_total_length", 0.0f);
        agent_focad_birth_cooldown = fmaxf(0.0f, FOCAD_BIRTH_REFRACTORY);
      }
    }
  }

  FLAMEGPU->setVariable<float>("focad_birth_cooldown", agent_focad_birth_cooldown);

  return flamegpu::ALIVE;
}
