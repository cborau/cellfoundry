FLAMEGPU_DEVICE_FUNCTION float clampf(const float x, const float lo, const float hi) {
  return fminf(hi, fmaxf(lo, x));
}

FLAMEGPU_DEVICE_FUNCTION float safeInv(const float x, const float eps) {
  return (fabsf(x) > eps) ? (1.0f / x) : 0.0f;
}

/**
 * cell_update_stress
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
FLAMEGPU_AGENT_FUNCTION(cell_update_stress, flamegpu::MessageBucket, flamegpu::MessageNone) {
  // -------------------------
  // Read CELL agent state
  // -------------------------
  const int agent_id = FLAMEGPU->getVariable<int>("id");

  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  const float CELL_RADIUS = FLAMEGPU->getVariable<float>("radius");
  const float CELL_NUCLEUS_RADIUS = FLAMEGPU->environment.getProperty<float>("CELL_NUCLEUS_RADIUS");

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

  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  // -------------------------
  // Accumulate stresslet S = sum_i sym(r_i ⊗ f_i)
  // where r_i = x_i - x_c
  // -------------------------
  float agent_S_xx = 0.0f, agent_S_yy = 0.0f, agent_S_zz = 0.0f;
  float agent_S_xy = 0.0f, agent_S_xz = 0.0f, agent_S_yz = 0.0f;

  // Iterate over all FOCAD messages addressed to this cell id
  for (const auto &message : FLAMEGPU->message_in(agent_id)) {
    // Anchor position on nucleus and force at the anchor (message variables)
    const float message_x_i = message.getVariable<float>("x_i"); // [um]
    const float message_y_i = message.getVariable<float>("y_i"); // [um]
    const float message_z_i = message.getVariable<float>("z_i"); // [um]

    const float message_fx  = message.getVariable<float>("fx");  // [nN]
    const float message_fy  = message.getVariable<float>("fy");  // [nN]
    const float message_fz  = message.getVariable<float>("fz");  // [nN]

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
  }

  // -------------------------
  // Average stress: sigma = (1/V) * S
  // V is nucleus volume (sphere)
  // Stress units: nN/um^2 (kPa)
  // -------------------------
  const float PI = 3.14159265358979323846f;
  const float agent_V = (4.0f / 3.0f) * PI * CELL_RADIUS * CELL_RADIUS * CELL_RADIUS; // [um^3]
  const float invV = safeInv(agent_V, 1e-20f);

  const float agent_sig_xx = invV * agent_S_xx;
  const float agent_sig_yy = invV * agent_S_yy;
  const float agent_sig_zz = invV * agent_S_zz;
  const float agent_sig_xy = invV * agent_S_xy;
  const float agent_sig_xz = invV * agent_S_xz;
  const float agent_sig_yz = invV * agent_S_yz;

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

  // -------------------------
  // Update nucleus anchors:
  // x_i = x_c + R (I + eps) u_ref_i
  //
  // Here u_ref_* are unit vectors defining reference anchor directions on the unit sphere (fixed at agent initialization).
  // The mapping (I + eps) turns the sphere into an ellipsoid consistent with small strain.
  // -------------------------
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
    const float anchor_x = agent_x + CELL_NUCLEUS_RADIUS * dux;
    const float anchor_y = agent_y + CELL_NUCLEUS_RADIUS * duy;
    const float anchor_z = agent_z + CELL_NUCLEUS_RADIUS * duz;

    // Store updated anchors (CELL array variables)
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("x_i", a, anchor_x);
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("y_i", a, anchor_y);
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("z_i", a, anchor_z);
  }

  return flamegpu::ALIVE;
}
