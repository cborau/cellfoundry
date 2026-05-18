/**
 * ccs_clampf
 *
 * Clamps a scalar to the closed interval [lo, hi].
 */
FLAMEGPU_DEVICE_FUNCTION float ccs_clampf(const float x, const float lo, const float hi) {
  return fminf(hi, fmaxf(lo, x));
}

/**
 * ccs_safeInv
 *
 * Returns 1/x when |x| > eps, otherwise returns 0.
 */
FLAMEGPU_DEVICE_FUNCTION float ccs_safeInv(const float x, const float eps) {
  return (fabsf(x) > eps) ? (1.0f / x) : 0.0f;
}

/**
 * ccs_normalize3
 *
 * Normalizes a 3D vector in-place; if near-zero, sets a default unit vector.
 */
FLAMEGPU_DEVICE_FUNCTION void ccs_normalize3(float &x, float &y, float &z) {
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
 * ccs_swapf
 *
 * Swaps two floats by reference.
 */
FLAMEGPU_DEVICE_FUNCTION void ccs_swapf(float &a, float &b) {
  const float t = a;
  a = b;
  b = t;
}

/**
 * ccs_swap_col3
 *
 * Swaps two columns of a 3x3 matrix (used for eigenvector column reordering).
 */
FLAMEGPU_DEVICE_FUNCTION void ccs_swap_col3(float V[3][3], const int c1, const int c2) {
  ccs_swapf(V[0][c1], V[0][c2]);
  ccs_swapf(V[1][c1], V[1][c2]);
  ccs_swapf(V[2][c1], V[2][c2]);
}

/**
 * ccs_eig_sym_3x3
 *
 * Jacobi eigendecomposition for a real symmetric 3x3 matrix.
 * Outputs eigenvalues sorted in descending order (l1 >= l2 >= l3)
 * with corresponding eigenvectors.
 */
FLAMEGPU_DEVICE_FUNCTION void ccs_eig_sym_3x3(
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
    ccs_swapf(eval[0], eval[1]);
    ccs_swap_col3(V, 0, 1);
  }
  if (eval[0] < eval[2]) {
    ccs_swapf(eval[0], eval[2]);
    ccs_swap_col3(V, 0, 2);
  }
  if (eval[1] < eval[2]) {
    ccs_swapf(eval[1], eval[2]);
    ccs_swap_col3(V, 1, 2);
  }

  l1 = eval[0];
  l2 = eval[1];
  l3 = eval[2];

  v1x = V[0][0]; v1y = V[1][0]; v1z = V[2][0];
  v2x = V[0][1]; v2y = V[1][1]; v2z = V[2][1];
  v3x = V[0][2]; v3y = V[1][2]; v3z = V[2][2];
}

/**
 * cell_stress_state_update
 *
 * Unified constitutive pipeline for ALL nucleus stress sources.
 *
 * Sums stresslet accumulators from every interaction source:
 *   focad_S_*  -- written by cell_focad_update  (FOCAD lever-arm stresslets)
 *   cc_S_*     -- written by cell_cell_interaction
 *   cf_S_*     -- written by cell_fnode_repulsion
 *   cl_S_*     -- written by cell_lumen_interaction
 *
 * Applies the isotropic viscoelastic constitutive law to produce a single,
 * consistent sig_eig_1 (and associated eigenvalues/vectors, strain state)
 * for all downstream mechanics regardless of which sources are active.
 *
 * When INCLUDE_FOCAL_ADHESIONS is true, also updates the nucleus anchor
 * point positions (x_i/y_i/z_i) from the reference directions (u_ref_*_i)
 * and the current strain state.
 *
 * Inputs:
 *   - focad_S_*, cc_S_*, cf_S_*, cl_S_*   (agent stresslet accumulators)
 *   - eps_*                                (persistent viscoelastic strain)
 *   - nucleus_radius, x, y, z, cell_type, orx/y/z
 *   - u_ref_x_i/y_i/z_i                   (reference anchor directions, if FA)
 *   - Environment: NUCLEUS_E, NUCLEUS_NU, NUCLEUS_TAU, NUCLEUS_EPS_CLAMP,
 *                  TIME_STEP, INCLUDE_FOCAL_ADHESIONS,
 *                  INCLUDE_ORIENTATION_ALIGN, ORIENTATION_ALIGN_RATE,
 *                  ORIENTATION_ALIGN_USE_STRESS
 *
 * Outputs:
 *   - sig_*         (stress tensor)
 *   - eps_*         (updated viscoelastic strain)
 *   - sig_eig_*     (principal stresses + eigenvectors)
 *   - eps_eig_*     (principal strains + eigenvectors)
 *   - orx/y/z       (updated orientation when INCLUDE_ORIENTATION_ALIGN)
 *   - x_i/y_i/z_i  (anchor positions, when INCLUDE_FOCAL_ADHESIONS)
 *
 * Units: length [um], force [nN], stress [nN/um² = kPa]
 */
FLAMEGPU_AGENT_FUNCTION(cell_stress_state_update, flamegpu::MessageNone, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    return flamegpu::ALIVE;
  }

  // -------------------------
  // Read agent state
  // -------------------------
  const int agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");
  const float agent_nucleus_radius = FLAMEGPU->getVariable<float>("nucleus_radius");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  float agent_orx = FLAMEGPU->getVariable<float>("orx");
  float agent_ory = FLAMEGPU->getVariable<float>("ory");
  float agent_orz = FLAMEGPU->getVariable<float>("orz");

  float agent_eps_xx = FLAMEGPU->getVariable<float>("eps_xx");
  float agent_eps_yy = FLAMEGPU->getVariable<float>("eps_yy");
  float agent_eps_zz = FLAMEGPU->getVariable<float>("eps_zz");
  float agent_eps_xy = FLAMEGPU->getVariable<float>("eps_xy");
  float agent_eps_xz = FLAMEGPU->getVariable<float>("eps_xz");
  float agent_eps_yz = FLAMEGPU->getVariable<float>("eps_yz");

  // -------------------------
  // Sum stresslet contributions from all interaction sources
  // -------------------------
  const float S_xx = FLAMEGPU->getVariable<float>("focad_S_xx")
                   + FLAMEGPU->getVariable<float>("cc_S_xx")
                   + FLAMEGPU->getVariable<float>("cf_S_xx")
                   + FLAMEGPU->getVariable<float>("cl_S_xx");
  const float S_yy = FLAMEGPU->getVariable<float>("focad_S_yy")
                   + FLAMEGPU->getVariable<float>("cc_S_yy")
                   + FLAMEGPU->getVariable<float>("cf_S_yy")
                   + FLAMEGPU->getVariable<float>("cl_S_yy");
  const float S_zz = FLAMEGPU->getVariable<float>("focad_S_zz")
                   + FLAMEGPU->getVariable<float>("cc_S_zz")
                   + FLAMEGPU->getVariable<float>("cf_S_zz")
                   + FLAMEGPU->getVariable<float>("cl_S_zz");
  const float S_xy = FLAMEGPU->getVariable<float>("focad_S_xy")
                   + FLAMEGPU->getVariable<float>("cc_S_xy")
                   + FLAMEGPU->getVariable<float>("cf_S_xy")
                   + FLAMEGPU->getVariable<float>("cl_S_xy");
  const float S_xz = FLAMEGPU->getVariable<float>("focad_S_xz")
                   + FLAMEGPU->getVariable<float>("cc_S_xz")
                   + FLAMEGPU->getVariable<float>("cf_S_xz")
                   + FLAMEGPU->getVariable<float>("cl_S_xz");
  const float S_yz = FLAMEGPU->getVariable<float>("focad_S_yz")
                   + FLAMEGPU->getVariable<float>("cc_S_yz")
                   + FLAMEGPU->getVariable<float>("cf_S_yz")
                   + FLAMEGPU->getVariable<float>("cl_S_yz");

  // -------------------------
  // Material and numerical parameters
  // -------------------------
  const uint8_t N_CELL_TYPES = 3; // WARNING: must match model.py
  const float NUCLEUS_E         = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("NUCLEUS_E", agent_cell_type);
  const float NUCLEUS_NU        = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("NUCLEUS_NU", agent_cell_type);
  const float NUCLEUS_TAU       = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("NUCLEUS_TAU", agent_cell_type);
  const float NUCLEUS_EPS_CLAMP = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("NUCLEUS_EPS_CLAMP", agent_cell_type);
  const float TIME_STEP         = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const int INCLUDE_FOCAL_ADHESIONS      = FLAMEGPU->environment.getProperty<int>("INCLUDE_FOCAL_ADHESIONS");
  const int INCLUDE_ORIENTATION_ALIGN    = FLAMEGPU->environment.getProperty<int>("INCLUDE_ORIENTATION_ALIGN");
  const float ORIENTATION_ALIGN_RATE     = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("ORIENTATION_ALIGN_RATE", agent_cell_type);
  const int ORIENTATION_ALIGN_USE_STRESS = FLAMEGPU->environment.getProperty<int>("ORIENTATION_ALIGN_USE_STRESS");

  // -------------------------
  // Average stress: sigma = (1/V_nuc) * S
  // V_nuc = (4/3) pi R_nuc^3  [um^3]
  // Stress units: nN/um^2 (kPa)
  // -------------------------
  const float PI = 3.14159265358979323846f;
  const float agent_V = (4.0f / 3.0f) * PI
                        * agent_nucleus_radius * agent_nucleus_radius * agent_nucleus_radius;
  const float invV = ccs_safeInv(agent_V, 1e-20f);

  const float agent_sig_xx = invV * S_xx;
  const float agent_sig_yy = invV * S_yy;
  const float agent_sig_zz = invV * S_zz;
  const float agent_sig_xy = invV * S_xy;
  const float agent_sig_xz = invV * S_xz;
  const float agent_sig_yz = invV * S_yz;

  // -------------------------
  // Isotropic compliance inversion:
  //   eps = (1/(2G)) sigma - alpha * tr(sigma) * I
  //   G = E / (2(1+nu)),  lambda = E*nu / ((1+nu)(1-2nu))
  //   alpha = lambda / (2G(3lambda + 2G))
  // -------------------------
  const float nu = ccs_clampf(NUCLEUS_NU, 0.0f, 0.499f);
  const float E  = fmaxf(NUCLEUS_E, 1e-12f);

  const float G = E / (2.0f * (1.0f + nu));
  const float lambda = (E * nu) / ((1.0f + nu) * (1.0f - 2.0f * nu));

  const float tr_sig = agent_sig_xx + agent_sig_yy + agent_sig_zz;
  const float denom  = 3.0f * lambda + 2.0f * G;
  const float alpha  = (denom > 0.0f) ? (lambda / (2.0f * G * denom)) : 0.0f;
  const float inv2G  = 1.0f / (2.0f * G);

  float eps_el_xx = inv2G * agent_sig_xx - alpha * tr_sig;
  float eps_el_yy = inv2G * agent_sig_yy - alpha * tr_sig;
  float eps_el_zz = inv2G * agent_sig_zz - alpha * tr_sig;
  float eps_el_xy = inv2G * agent_sig_xy;
  float eps_el_xz = inv2G * agent_sig_xz;
  float eps_el_yz = inv2G * agent_sig_yz;

  // -------------------------
  // Nearly incompressible deviatoric projection:
  //   eps_tilde = eps_el - (tr(eps_el)/3) I
  // -------------------------
  const float tr_eps_el = eps_el_xx + eps_el_yy + eps_el_zz;
  const float m = tr_eps_el / 3.0f;

  const float eps_tilde_xx = eps_el_xx - m;
  const float eps_tilde_yy = eps_el_yy - m;
  const float eps_tilde_zz = eps_el_zz - m;
  const float eps_tilde_xy = eps_el_xy;
  const float eps_tilde_xz = eps_el_xz;
  const float eps_tilde_yz = eps_el_yz;

  // -------------------------
  // Viscoelastic relaxation (first-order):
  //   eps <- eps + (dt/tau) * (eps_tilde - eps)
  // -------------------------
  const float tau_s  = fmaxf(NUCLEUS_TAU, 1e-6f);
  const float rel    = fminf(TIME_STEP / tau_s, 1.0f);

  agent_eps_xx += rel * (eps_tilde_xx - agent_eps_xx);
  agent_eps_yy += rel * (eps_tilde_yy - agent_eps_yy);
  agent_eps_zz += rel * (eps_tilde_zz - agent_eps_zz);
  agent_eps_xy += rel * (eps_tilde_xy - agent_eps_xy);
  agent_eps_xz += rel * (eps_tilde_xz - agent_eps_xz);
  agent_eps_yz += rel * (eps_tilde_yz - agent_eps_yz);

  // -------------------------
  // Small-strain clamp
  // -------------------------
  agent_eps_xx = ccs_clampf(agent_eps_xx, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_yy = ccs_clampf(agent_eps_yy, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_zz = ccs_clampf(agent_eps_zz, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_xy = ccs_clampf(agent_eps_xy, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_xz = ccs_clampf(agent_eps_xz, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);
  agent_eps_yz = ccs_clampf(agent_eps_yz, -NUCLEUS_EPS_CLAMP, NUCLEUS_EPS_CLAMP);

  // Write updated strain
  FLAMEGPU->setVariable<float>("eps_xx", agent_eps_xx);
  FLAMEGPU->setVariable<float>("eps_yy", agent_eps_yy);
  FLAMEGPU->setVariable<float>("eps_zz", agent_eps_zz);
  FLAMEGPU->setVariable<float>("eps_xy", agent_eps_xy);
  FLAMEGPU->setVariable<float>("eps_xz", agent_eps_xz);
  FLAMEGPU->setVariable<float>("eps_yz", agent_eps_yz);

  // Write stress
  FLAMEGPU->setVariable<float>("sig_xx", agent_sig_xx);
  FLAMEGPU->setVariable<float>("sig_yy", agent_sig_yy);
  FLAMEGPU->setVariable<float>("sig_zz", agent_sig_zz);
  FLAMEGPU->setVariable<float>("sig_xy", agent_sig_xy);
  FLAMEGPU->setVariable<float>("sig_xz", agent_sig_xz);
  FLAMEGPU->setVariable<float>("sig_yz", agent_sig_yz);

  // -------------------------
  // Eigendecomposition: stress tensor
  // -------------------------
  float sig_l1, sig_l2, sig_l3;
  float sig_v1x, sig_v1y, sig_v1z;
  float sig_v2x, sig_v2y, sig_v2z;
  float sig_v3x, sig_v3y, sig_v3z;
  ccs_eig_sym_3x3(
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

  // -------------------------
  // Eigendecomposition: strain tensor
  // -------------------------
  float eps_l1, eps_l2, eps_l3;
  float eps_v1x, eps_v1y, eps_v1z;
  float eps_v2x, eps_v2y, eps_v2z;
  float eps_v3x, eps_v3y, eps_v3z;
  ccs_eig_sym_3x3(
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

  // -------------------------
  // Orientation alignment toward max principal direction
  //
  // Guard: skip when the stress/strain tensor is near-isotropic.
  // For symmetric packing (hydrostatic compression), all eigenvalues are
  // nearly equal and the Jacobi solver returns the identity columns as
  // eigenvectors (v1 = (1,0,0)), which would introduce a spurious x-bias.
  // Only apply alignment when the normalized anisotropy exceeds 5 %.
  // -------------------------
  if (INCLUDE_ORIENTATION_ALIGN) {
    const float l1_a  = ORIENTATION_ALIGN_USE_STRESS ? sig_l1 : eps_l1;
    const float l3_a  = ORIENTATION_ALIGN_USE_STRESS ? sig_l3 : eps_l3;
    const float l2_a  = ORIENTATION_ALIGN_USE_STRESS ? sig_l2 : eps_l2;
    const float mag_a = fabsf(l1_a) + fabsf(l2_a) + fabsf(l3_a);
    const float aniso = (mag_a > 1e-12f) ? (l1_a - l3_a) / mag_a : 0.0f;

    if (aniso > 0.05f) { // minimum anisotropy threshold to avoid alignment to noise
      float target_x = ORIENTATION_ALIGN_USE_STRESS ? sig_v1x : eps_v1x;
      float target_y = ORIENTATION_ALIGN_USE_STRESS ? sig_v1y : eps_v1y;
      float target_z = ORIENTATION_ALIGN_USE_STRESS ? sig_v1z : eps_v1z;

      // Avoid sign flips: choose target with positive dot to current orientation
      const float dot = agent_orx*target_x + agent_ory*target_y + agent_orz*target_z;
      if (dot < 0.0f) {
        target_x = -target_x;
        target_y = -target_y;
        target_z = -target_z;
      }

      const float a = ccs_clampf(ORIENTATION_ALIGN_RATE * TIME_STEP, 0.0f, 1.0f);
      agent_orx = (1.0f - a) * agent_orx + a * target_x;
      agent_ory = (1.0f - a) * agent_ory + a * target_y;
      agent_orz = (1.0f - a) * agent_orz + a * target_z;
      ccs_normalize3(agent_orx, agent_ory, agent_orz);
    }
  }
  FLAMEGPU->setVariable<float>("orx", agent_orx);
  FLAMEGPU->setVariable<float>("ory", agent_ory);
  FLAMEGPU->setVariable<float>("orz", agent_orz);

  // -------------------------
  // Nucleus anchor position update (only when INCLUDE_FOCAL_ADHESIONS)
  //
  // Apply deformation map to reference anchor directions:
  //   du = (I + eps) * u_ref
  //   anchor_i = cell_centre + R_nuc * du
  //
  // Uses the freshly updated eps_* from this step.
  // -------------------------
  if (INCLUDE_FOCAL_ADHESIONS) {
    const uint8_t N_ANCHOR_POINTS = 50; // WARNING: must match model.py
    for (unsigned int a = 0; a < N_ANCHOR_POINTS; ++a) {
      const float ux = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_x_i", a);
      const float uy = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_y_i", a);
      const float uz = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_z_i", a);

      // du = (I + eps) * u_ref   (eps is symmetric: eps_yx = eps_xy, etc.)
      const float dux = ux + agent_eps_xx * ux + agent_eps_xy * uy + agent_eps_xz * uz;
      const float duy = uy + agent_eps_xy * ux + agent_eps_yy * uy + agent_eps_yz * uz;
      const float duz = uz + agent_eps_xz * ux + agent_eps_yz * uy + agent_eps_zz * uz;

      FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("x_i", a, agent_x + agent_nucleus_radius * dux);
      FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("y_i", a, agent_y + agent_nucleus_radius * duy);
      FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("z_i", a, agent_z + agent_nucleus_radius * duz);
    }
  }

  return flamegpu::ALIVE;
}
