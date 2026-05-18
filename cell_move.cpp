// -----------------------------------------------------------------------------
// Device helper functions 
// -----------------------------------------------------------------------------
FLAMEGPU_DEVICE_FUNCTION float clampf(const float x, const float lo, const float hi) {
  return fminf(hi, fmaxf(lo, x));
}

FLAMEGPU_DEVICE_FUNCTION int clampi(const int x, const int lo, const int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Wraps x into the interval [lo, hi).
FLAMEGPU_DEVICE_FUNCTION float wrapf(const float x, const float lo, const float hi) {
  const float L = hi - lo;
  float r = fmodf(x - lo, L);
  if (r < 0.0f) r += L;
  return lo + r;
}

// WARNING: Ensure ECM agents were created with z as the fastest index, then y, then x:
// grid_lin_id = i*(Ny*Nz) + j*(Nz) + k
FLAMEGPU_DEVICE_FUNCTION uint32_t macro_lin_idx(const int ii, const int jj, const int kk, const int Ny, const int Nz) {
  return (uint32_t)(ii * (Ny * Nz) + jj * Nz + kk);
}

// Normalizes a 3D vector in-place; if near-zero, sets a default unit vector.
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

// Bounded Hill gate in [0, 1].
FLAMEGPU_DEVICE_FUNCTION float hill01(const float c, const float K, const float n) {
  const float c_pos = fmaxf(c, 0.0f);
  const float K_pos = fmaxf(K, 1e-12f);
  const float nn = fmaxf(n, 1.0f);

  const float cn = powf(c_pos, nn);
  const float Kn = powf(K_pos, nn);
  return cn / (Kn + cn + 1e-12f);
}

// Soft saturation of a non-negative sensed signal.
// Formulation:
//   s_sat = s_max * s / (s_max + s)
// For s << s_max, this is approximately linear.
// For s >> s_max, this approaches s_max.
FLAMEGPU_DEVICE_FUNCTION float saturate_signal(const float signal, const float sat_level) {
  const float s = fmaxf(signal, 0.0f);
  const float smax = fmaxf(sat_level, 1e-12f);
  return smax * s / (smax + s);
}

/**
 * cell_move
 *
 * Purpose:
 *   Update CELL velocity/orientation-driven migration by combining Brownian,
 *   chemotactic, chemokinetic and durotactic components, then advance position.
 *
 * Inputs:
 *   - CELL kinematic state, stress/strain eigensystem, chemotaxis sensitivities
 *   - Environment controls for chemotaxis/durotaxis and timestep
 *
 * Outputs:
 *   - Updated CELL position, velocity, orientation-aligned motion state
 */
FLAMEGPU_AGENT_FUNCTION(cell_move, flamegpu::MessageNone, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    // Note: if DEAD_CELLS_DISAPPEAR = True, a dead CELL agent remains ALIVE for flamegpu purposes and may still interact with other agents.
    FLAMEGPU->setVariable<float>("vx", 0.0f);
    FLAMEGPU->setVariable<float>("vy", 0.0f);
    FLAMEGPU->setVariable<float>("vz", 0.0f);
    return flamegpu::ALIVE;
  }

  // Get agent variables
  int agent_id = FLAMEGPU->getVariable<int>("id");
  int agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  const float agent_x_prev = agent_x;
  const float agent_y_prev = agent_y;
  const float agent_z_prev = agent_z;
  float trajectory_length = FLAMEGPU->getVariable<float>("trajectory_length");
  float trajectory_time = FLAMEGPU->getVariable<float>("trajectory_time");

  // Velocity contributions are accumulated and applied at the end
  float agent_vx = 0.0f;
  float agent_vy = 0.0f;
  float agent_vz = 0.0f;
  const float agent_cc_dvx = FLAMEGPU->getVariable<float>("cc_dvx"); // cell-cell
  const float agent_cc_dvy = FLAMEGPU->getVariable<float>("cc_dvy");
  const float agent_cc_dvz = FLAMEGPU->getVariable<float>("cc_dvz");
  const float agent_cf_dvx = FLAMEGPU->getVariable<float>("cf_dvx"); // cell-fnode
  const float agent_cf_dvy = FLAMEGPU->getVariable<float>("cf_dvy");
  const float agent_cf_dvz = FLAMEGPU->getVariable<float>("cf_dvz");
  const float agent_cl_dvx = FLAMEGPU->getVariable<float>("cl_dvx"); // cell-lumen
  const float agent_cl_dvy = FLAMEGPU->getVariable<float>("cl_dvy");
  const float agent_cl_dvz = FLAMEGPU->getVariable<float>("cl_dvz");

  const uint8_t N_ANCHOR_POINTS = 50; // WARNING: must match main python
  float agent_x_i[N_ANCHOR_POINTS] = {};
  for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    agent_x_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("x_i", i);
  }
  float agent_y_i[N_ANCHOR_POINTS] = {};
  for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    agent_y_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("y_i", i);
  }
  float agent_z_i[N_ANCHOR_POINTS] = {};
  for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    agent_z_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("z_i", i);
  }

  // Orientation
  float agent_orx = FLAMEGPU->getVariable<float>("orx");
  float agent_ory = FLAMEGPU->getVariable<float>("ory");
  float agent_orz = FLAMEGPU->getVariable<float>("orz");
  normalize3(agent_orx, agent_ory, agent_orz);

  // Stress tensor
  const float agent_sig_xx = FLAMEGPU->getVariable<float>("sig_xx");
  const float agent_sig_yy = FLAMEGPU->getVariable<float>("sig_yy");
  const float agent_sig_zz = FLAMEGPU->getVariable<float>("sig_zz");
  const float agent_sig_xy = FLAMEGPU->getVariable<float>("sig_xy");
  const float agent_sig_xz = FLAMEGPU->getVariable<float>("sig_xz");
  const float agent_sig_yz = FLAMEGPU->getVariable<float>("sig_yz");

  // Strain tensor
  const float agent_eps_xx = FLAMEGPU->getVariable<float>("eps_xx");
  const float agent_eps_yy = FLAMEGPU->getVariable<float>("eps_yy");
  const float agent_eps_zz = FLAMEGPU->getVariable<float>("eps_zz");
  const float agent_eps_xy = FLAMEGPU->getVariable<float>("eps_xy");
  const float agent_eps_xz = FLAMEGPU->getVariable<float>("eps_xz");
  const float agent_eps_yz = FLAMEGPU->getVariable<float>("eps_yz");

  // Precomputed principal values/vectors (stress)
  const float sig_l1 = FLAMEGPU->getVariable<float>("sig_eig_1");
  const float sig_l2 = FLAMEGPU->getVariable<float>("sig_eig_2");
  const float sig_l3 = FLAMEGPU->getVariable<float>("sig_eig_3");
  float sig_v1x = FLAMEGPU->getVariable<float>("sig_eigvec1_x");
  float sig_v1y = FLAMEGPU->getVariable<float>("sig_eigvec1_y");
  float sig_v1z = FLAMEGPU->getVariable<float>("sig_eigvec1_z");
  normalize3(sig_v1x, sig_v1y, sig_v1z);

  // Precomputed principal values/vectors (strain)
  const float eps_l1 = FLAMEGPU->getVariable<float>("eps_eig_1");
  const float eps_l2 = FLAMEGPU->getVariable<float>("eps_eig_2");
  const float eps_l3 = FLAMEGPU->getVariable<float>("eps_eig_3");
  float eps_v1x = FLAMEGPU->getVariable<float>("eps_eigvec1_x");
  float eps_v1y = FLAMEGPU->getVariable<float>("eps_eigvec1_y");
  float eps_v1z = FLAMEGPU->getVariable<float>("eps_eigvec1_z");
  normalize3(eps_v1x, eps_v1y, eps_v1z);

  // ---------------------------------------------------------------------------
  // Environment config
  // ---------------------------------------------------------------------------
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");

  const uint8_t N_SPECIES = 2; // WARNING: must match main python
  const uint32_t ECM_POPULATION_SIZE = 9261; // WARNING: must match Nx*Ny*Nz
  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");

  // Get number of agents per direction
  const int Nx = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 0);
  const int Ny = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 1);
  const int Nz = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 2);

  // Get position of the boundaries
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 5);

  const uint8_t N_CELL_TYPES = 3; // WARNING: must match main python model N_CELL_TYPES

  // Chemotaxis controls
  const int INCLUDE_CHEMOTAXIS = FLAMEGPU->environment.getProperty<int>("INCLUDE_CHEMOTAXIS");
  const int CHEMOTAXIS_ONLY_DIR = FLAMEGPU->environment.getProperty<int>("CHEMOTAXIS_ONLY_DIR");
  const float CHEMOTAXIS_CHI = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CHEMOTAXIS_CHI", agent_cell_type);

  // Chemokinesis controls
  const int INCLUDE_CHEMOKINESIS = FLAMEGPU->environment.getProperty<int>("INCLUDE_CHEMOKINESIS");
  const float CHEMOKINESIS_ALPHA = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CHEMOKINESIS_ALPHA", agent_cell_type);
  const float CHEMOKINESIS_K = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CHEMOKINESIS_K", agent_cell_type);
  const float CHEMOKINESIS_HILL_N = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CHEMOKINESIS_HILL_N", agent_cell_type);
  const float CHEMOKINESIS_ADAPT_TAU = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CHEMOKINESIS_ADAPT_TAU", agent_cell_type);
  const float CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER", agent_cell_type);

  float chemokinesis_signal_sat_base[N_SPECIES] = {};
  for (int s = 0; s < N_SPECIES; s++) {
    chemokinesis_signal_sat_base[s] = FLAMEGPU->environment.getProperty<float, N_SPECIES>("CHEMOKINESIS_SIGNAL_SAT", s);
  }

  // Durotaxis controls
  const int INCLUDE_DUROTAXIS = FLAMEGPU->environment.getProperty<int>("INCLUDE_DUROTAXIS");
  const int DUROTAXIS_ONLY_DIR = FLAMEGPU->environment.getProperty<int>("DUROTAXIS_ONLY_DIR");
  const float FOCAD_MOBILITY_MU = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_MOBILITY_MU", agent_cell_type);

  // Recommended additional controls for the blended model
  const float DUROTAXIS_BLEND_BETA = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("DUROTAXIS_BLEND_BETA", agent_cell_type);
  const int DUROTAXIS_USE_STRESS = FLAMEGPU->environment.getProperty<int>("DUROTAXIS_USE_STRESS");

  // ---------------------------------------------------------------------------
  // Intermediate velocity accumulation
  // ---------------------------------------------------------------------------
  float v_base_x = 0.0f, v_base_y = 0.0f, v_base_z = 0.0f;
  float steer_x  = 0.0f, steer_y  = 0.0f, steer_z  = 0.0f;
  float dv_x     = 0.0f, dv_y     = 0.0f, dv_z     = 0.0f;

  // Persistent self-propulsion along current orientation
  const float agent_speed_ref = FLAMEGPU->getVariable<float>("speed_ref");

  // Brownian motion
  const float BROWNIAN_MOTION_STRENGTH = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("BROWNIAN_MOTION_STRENGTH", agent_cell_type);

  // Rotational diffusion
  const float ROTATIONAL_DIFFUSION_RATE = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("ROTATIONAL_DIFFUSION_RATE", agent_cell_type);

  float chemotaxis_sensitivity[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    chemotaxis_sensitivity[i] = FLAMEGPU->environment.getProperty<float, N_SPECIES>("CHEMOTAXIS_SENSITIVITY", i);
  }

  // Unified chemokinesis sensitivity per species, shared by all cell types:
  // > 0 contributes to the promotive channel
  // < 0 contributes to the inhibitory channel
  float chemokinesis_sensitivity[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    chemokinesis_sensitivity[i] =  FLAMEGPU->environment.getProperty<float, N_SPECIES>("CHEMOKINESIS_SENSITIVITY", i);
  }

  // Separate adaptation states are kept for promotive and inhibitory signalling,
  // both stored as arrays over species
  float chemokinesis_promotive_adapt_state[N_SPECIES] = {};
  float chemokinesis_inhibitory_adapt_state[N_SPECIES] = {};
  for (int s = 0; s < N_SPECIES; s++) {
    chemokinesis_promotive_adapt_state[s] =
        FLAMEGPU->getVariable<float, N_SPECIES>("chemokinesis_promotive_adapt_state", s);
    chemokinesis_inhibitory_adapt_state[s] =
        FLAMEGPU->getVariable<float, N_SPECIES>("chemokinesis_inhibitory_adapt_state", s);
  }

  // ---------------------------------------------------------------------------
  // Transform x,y,z positions to i,j,k grid positions
  // ---------------------------------------------------------------------------
  int agent_grid_i = roundf(((agent_x - COORD_BOUNDARY_X_NEG) / (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG)) * (Nx - 1));
  int agent_grid_j = roundf(((agent_y - COORD_BOUNDARY_Y_NEG) / (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG)) * (Ny - 1));
  int agent_grid_k = roundf(((agent_z - COORD_BOUNDARY_Z_NEG) / (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG)) * (Nz - 1));

  agent_grid_i = clampi(agent_grid_i, 0, Nx - 1);
  agent_grid_j = clampi(agent_grid_j, 0, Ny - 1);
  agent_grid_k = clampi(agent_grid_k, 0, Nz - 1);

  // Current ECM voxel index, used by both chemotaxis and chemokinesis
  const uint32_t c_idx = macro_lin_idx(agent_grid_i, agent_grid_j, agent_grid_k, Ny, Nz);

  // ---------------------------------------------------------------------------
  // CHEMOKINESIS: absolute local concentration modulates persistent speed
  // Separate promotive and inhibitory signals are built from the sign of the
  // unified per-species sensitivity, but both channels share the same response
  // parameters (alpha, K, Hill exponent, adaptation timescale).
  //
  // Saturation is now applied per species and per cell:
  //   sat_level(s, cell_type) =
  //       CHEMOKINESIS_SIGNAL_SAT[s] *
  //       CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER[cell_type]
  //
  // Adaptation is also tracked per species and per channel, then the effective
  // desensitized signals are summed into the promotive and inhibitory channels.
  // ---------------------------------------------------------------------------
  float chemokinesis_factor = 1.0f;
  if (INCLUDE_CHEMOKINESIS) {
    float promo_signal_eff_sum = 0.0f;
    float inhib_signal_eff_sum = 0.0f;

    for (int s = 0; s < N_SPECIES; s++) {
      const float sens = chemokinesis_sensitivity[s];
      const float c_local = fmaxf(C_SP_MACRO[s][c_idx], 0.0f);
      const float sat_level =
          chemokinesis_signal_sat_base[s] * CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER;

      if (sens > 0.0f) {
        const float promo_signal_raw = sens * c_local;
        const float promo_signal_pos = fmaxf(promo_signal_raw, 0.0f);
        const float promo_signal_sat = saturate_signal(promo_signal_pos, sat_level);

        if (CHEMOKINESIS_ADAPT_TAU > 1e-12f) {
          chemokinesis_promotive_adapt_state[s] +=
              (TIME_STEP / CHEMOKINESIS_ADAPT_TAU) *
              (promo_signal_sat - chemokinesis_promotive_adapt_state[s]);
        } else {
          chemokinesis_promotive_adapt_state[s] = promo_signal_sat;
        }
        chemokinesis_promotive_adapt_state[s] =
            fmaxf(chemokinesis_promotive_adapt_state[s], 0.0f);

        // Unused inhibitory branch for this species
        chemokinesis_inhibitory_adapt_state[s] = 0.0f;

        const float promo_signal_eff =
            fmaxf(promo_signal_sat - chemokinesis_promotive_adapt_state[s], 0.0f);
        promo_signal_eff_sum += promo_signal_eff;

      } else if (sens < 0.0f) {
        const float inhib_signal_raw = (-sens) * c_local;
        const float inhib_signal_pos = fmaxf(inhib_signal_raw, 0.0f);
        const float inhib_signal_sat = saturate_signal(inhib_signal_pos, sat_level);

        if (CHEMOKINESIS_ADAPT_TAU > 1e-12f) {
          chemokinesis_inhibitory_adapt_state[s] +=
              (TIME_STEP / CHEMOKINESIS_ADAPT_TAU) *
              (inhib_signal_sat - chemokinesis_inhibitory_adapt_state[s]);
        } else {
          chemokinesis_inhibitory_adapt_state[s] = inhib_signal_sat;
        }
        chemokinesis_inhibitory_adapt_state[s] =
            fmaxf(chemokinesis_inhibitory_adapt_state[s], 0.0f);

        // Unused promotive branch for this species
        chemokinesis_promotive_adapt_state[s] = 0.0f;

        const float inhib_signal_eff =
            fmaxf(inhib_signal_sat - chemokinesis_inhibitory_adapt_state[s], 0.0f);
        inhib_signal_eff_sum += inhib_signal_eff;

      } else {
        chemokinesis_promotive_adapt_state[s] = 0.0f;
        chemokinesis_inhibitory_adapt_state[s] = 0.0f;
      }
    }

    const float h_promo = hill01(promo_signal_eff_sum, CHEMOKINESIS_K, CHEMOKINESIS_HILL_N);
    const float h_inhib = hill01(inhib_signal_eff_sum, CHEMOKINESIS_K, CHEMOKINESIS_HILL_N);

    // Shared alpha for both channels:
    // promotive signal increases speed, inhibitory signal decreases speed
    chemokinesis_factor = 1.0f + CHEMOKINESIS_ALPHA * (h_promo - h_inhib);

    // Prevent negative persistent speed
    chemokinesis_factor = fmaxf(chemokinesis_factor, 0.0f);
  }

  // ---------------------------------------------------------------------------
  // ROTATIONAL DIFFUSION: stochastic reorientation of the cell polarity vector.
  // Perturbs orientation by adding noise scaled by sqrt(2 * D_rot * dt) to each
  // component and renormalizing, which is a standard first-order discretization
  // of rotational Brownian motion on the unit sphere.
  // ---------------------------------------------------------------------------
  if (ROTATIONAL_DIFFUSION_RATE > 0.0f) {
    const float sigma_rot = sqrtf(2.0f * ROTATIONAL_DIFFUSION_RATE * TIME_STEP);
    agent_orx += sigma_rot * FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
    agent_ory += sigma_rot * FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
    agent_orz += sigma_rot * FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
    normalize3(agent_orx, agent_ory, agent_orz);
  }

  // Persistent self-propulsion along current orientation, modulated by chemokinesis
  const float persistent_speed = agent_speed_ref * chemokinesis_factor;
  v_base_x += persistent_speed * agent_orx;
  v_base_y += persistent_speed * agent_ory;
  v_base_z += persistent_speed * agent_orz;

  v_base_x += BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0f, 1.0f));
  v_base_y += BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0f, 1.0f));
  v_base_z += BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0f, 1.0f));

  // ---------------------------------------------------------------------------
  // CHEMOTAXIS: compute direction and add to steer or dv
  // ---------------------------------------------------------------------------
  if (INCLUDE_CHEMOTAXIS) {
    const float dx = (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG) / (Nx - 1);
    const float dy = (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG) / (Ny - 1);
    const float dz = (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG) / (Nz - 1);

    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

    for (int dk = -1; dk <= 1; dk++) {
      for (int dj = -1; dj <= 1; dj++) {
        for (int di = -1; di <= 1; di++) {
          if (di == 0 && dj == 0 && dk == 0) continue;

          const int ni = clampi(agent_grid_i + di, 0, Nx - 1);
          const int nj = clampi(agent_grid_j + dj, 0, Ny - 1);
          const int nk = clampi(agent_grid_k + dk, 0, Nz - 1);

          const uint32_t n_idx = macro_lin_idx(ni, nj, nk, Ny, Nz);

          const float ddx = (float)di * dx;
          const float ddy = (float)dj * dy;
          const float ddz = (float)dk * dz;

          const float dist2 = ddx*ddx + ddy*ddy + ddz*ddz + 1e-12f;
          const float inv_dist = rsqrtf(dist2);

          const float ux_n = ddx * inv_dist;
          const float uy_n = ddy * inv_dist;
          const float uz_n = ddz * inv_dist;

          float dC_total = 0.0f;
          for (int s = 0; s < N_SPECIES; s++) {
            const float sens = chemotaxis_sensitivity[s];
            if (sens == 0.0f) continue;
            const float Cn = C_SP_MACRO[s][n_idx];
            const float C0 = C_SP_MACRO[s][c_idx];
            dC_total += sens * (Cn - C0);
          }

          const float w = dC_total * inv_dist;
          grad_x += w * ux_n;
          grad_y += w * uy_n;
          grad_z += w * uz_n;
        }
      }
    }

    const float g2 = grad_x*grad_x + grad_y*grad_y + grad_z*grad_z;
    if (g2 > 0.0f) {
      const float inv_g = rsqrtf(g2 + 1e-12f);
      const float chemo_dir_x = grad_x * inv_g;
      const float chemo_dir_y = grad_y * inv_g;
      const float chemo_dir_z = grad_z * inv_g;

      if (CHEMOTAXIS_ONLY_DIR == 1) {
        steer_x += CHEMOTAXIS_CHI * chemo_dir_x;
        steer_y += CHEMOTAXIS_CHI * chemo_dir_y;
        steer_z += CHEMOTAXIS_CHI * chemo_dir_z;
      } else {
        dv_x += CHEMOTAXIS_CHI * chemo_dir_x;
        dv_y += CHEMOTAXIS_CHI * chemo_dir_y;
        dv_z += CHEMOTAXIS_CHI * chemo_dir_z;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // DUROTAXIS: blended direction between traction and principal direction
  // ---------------------------------------------------------------------------
  if (INCLUDE_DUROTAXIS) {

    // Traction direction: t_dir = normalize( sigma * ori )
    const float t_x = agent_sig_xx*agent_orx + agent_sig_xy*agent_ory + agent_sig_xz*agent_orz;
    const float t_y = agent_sig_xy*agent_orx + agent_sig_yy*agent_ory + agent_sig_yz*agent_orz;
    const float t_z = agent_sig_xz*agent_orx + agent_sig_yz*agent_ory + agent_sig_zz*agent_orz;

    float tdir_x = t_x, tdir_y = t_y, tdir_z = t_z;
    normalize3(tdir_x, tdir_y, tdir_z);

    // Principal direction and eigenvalues source selection
    float pdir_x = sig_v1x, pdir_y = sig_v1y, pdir_z = sig_v1z;
    float l1 = sig_l1, l2 = sig_l2, l3 = sig_l3;

    if (DUROTAXIS_USE_STRESS == 0) {
      pdir_x = eps_v1x; pdir_y = eps_v1y; pdir_z = eps_v1z;
      l1 = eps_l1; l2 = eps_l2; l3 = eps_l3;
    }
    normalize3(pdir_x, pdir_y, pdir_z);

    // Sign continuity relative to orientation
    const float dot_or = agent_orx*pdir_x + agent_ory*pdir_y + agent_orz*pdir_z;
    if (dot_or < 0.0f) { pdir_x = -pdir_x; pdir_y = -pdir_y; pdir_z = -pdir_z; }

    // Blend direction: dir = normalize( (1-beta)*tdir + beta*pdir )
    const float beta = clampf(DUROTAXIS_BLEND_BETA, 0.0f, 1.0f);
    float duro_dir_x = (1.0f - beta) * tdir_x + beta * pdir_x;
    float duro_dir_y = (1.0f - beta) * tdir_y + beta * pdir_y;
    float duro_dir_z = (1.0f - beta) * tdir_z + beta * pdir_z;
    normalize3(duro_dir_x, duro_dir_y, duro_dir_z);

    // Strength scaling
    float scale_energy = agent_sig_xx*agent_eps_xx + agent_sig_yy*agent_eps_yy + agent_sig_zz*agent_eps_zz
                       + 2.0f*(agent_sig_xy*agent_eps_xy + agent_sig_xz*agent_eps_xz + agent_sig_yz*agent_eps_yz);
    if (scale_energy < 0.0f) scale_energy = 0.0f;

    // Unitless anisotropy factor from principal values
    const float aniso_den = fabsf(l1) + fabsf(l2) + fabsf(l3) + 1e-12f;
    float A = (l1 - l3) / aniso_den;

    // Final durotaxis strength
    float duro_strength = FOCAD_MOBILITY_MU * (scale_energy + A);
    if (duro_strength < 1e-12f) {
      duro_strength = FOCAD_MOBILITY_MU * fabsf(l1);
    }

    if (DUROTAXIS_ONLY_DIR == 1) {
      steer_x += duro_strength * duro_dir_x;
      steer_y += duro_strength * duro_dir_y;
      steer_z += duro_strength * duro_dir_z;
    } else {
      dv_x += duro_strength * duro_dir_x;
      dv_y += duro_strength * duro_dir_y;
      dv_z += duro_strength * duro_dir_z;
    }
  }

  // ---------------------------------------------------------------------------
  // Apply steering once, then add speed-changing dv
  // ---------------------------------------------------------------------------
  agent_vx = v_base_x;
  agent_vy = v_base_y;
  agent_vz = v_base_z;

  // Steering modifies direction while preserving current speed
  const float steer2 = steer_x*steer_x + steer_y*steer_y + steer_z*steer_z;
  if (steer2 > 1e-20f) {
    const float v2 = agent_vx*agent_vx + agent_vy*agent_vy + agent_vz*agent_vz;
    float vmag = sqrtf(v2 + 1e-12f);

    // If base speed is tiny, use speed_ref so steering can still produce motion
    if (vmag < 1e-6f) vmag = persistent_speed;

    float vdir_x = agent_vx;
    float vdir_y = agent_vy;
    float vdir_z = agent_vz;

    // If base direction is tiny, use steering direction
    const float vdir2 = vdir_x*vdir_x + vdir_y*vdir_y + vdir_z*vdir_z;
    if (vdir2 > 1e-20f) {
      normalize3(vdir_x, vdir_y, vdir_z);
    } else {
      vdir_x = steer_x; vdir_y = steer_y; vdir_z = steer_z;
      normalize3(vdir_x, vdir_y, vdir_z);
    }

    // Blend and renormalize
    float ndir_x = vdir_x + steer_x;
    float ndir_y = vdir_y + steer_y;
    float ndir_z = vdir_z + steer_z;
    normalize3(ndir_x, ndir_y, ndir_z);

    agent_vx = vmag * ndir_x;
    agent_vy = vmag * ndir_y;
    agent_vz = vmag * ndir_z;
  }

  // Add speed-changing contributions
  agent_vx += dv_x;
  agent_vy += dv_y;
  agent_vz += dv_z;

  // Add short-range interaction contributions
  agent_vx += agent_cc_dvx + agent_cf_dvx + agent_cl_dvx;
  agent_vy += agent_cc_dvy + agent_cf_dvy + agent_cl_dvy;
  agent_vz += agent_cc_dvz + agent_cf_dvz + agent_cl_dvz;

  // ---------------------------------------------------------------------------
  // Update agent position based on velocity
  // ---------------------------------------------------------------------------
  agent_x += agent_vx * TIME_STEP;
  agent_y += agent_vy * TIME_STEP;
  agent_z += agent_vz * TIME_STEP;

  // Boundary handling: periodic wrapping or simple clamping
  const unsigned int PERIODIC_BOUNDARIES_FOR_CELLS = FLAMEGPU->environment.getProperty<int>("PERIODIC_BOUNDARIES_FOR_CELLS");
  const unsigned int INCLUDE_FOCAL_ADHESIONS = FLAMEGPU->environment.getProperty<int>("INCLUDE_FOCAL_ADHESIONS");

  if (PERIODIC_BOUNDARIES_FOR_CELLS == 1 && INCLUDE_FOCAL_ADHESIONS == 0) {
    // Periodic wrapping for cell position
    agent_x = wrapf(agent_x, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_X_POS);
    agent_y = wrapf(agent_y, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Y_POS);
    agent_z = wrapf(agent_z, COORD_BOUNDARY_Z_NEG, COORD_BOUNDARY_Z_POS);

    // Move anchor points by the raw displacement and wrap individually
    const float raw_dx = agent_vx * TIME_STEP;
    const float raw_dy = agent_vy * TIME_STEP;
    const float raw_dz = agent_vz * TIME_STEP;
    for (int i = 0; i < N_ANCHOR_POINTS; i++) {
      agent_x_i[i] = wrapf(agent_x_i[i] + raw_dx, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_X_POS);
      agent_y_i[i] = wrapf(agent_y_i[i] + raw_dy, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Y_POS);
      agent_z_i[i] = wrapf(agent_z_i[i] + raw_dz, COORD_BOUNDARY_Z_NEG, COORD_BOUNDARY_Z_POS);
    }
  } else {
    // Simple clamp to domain
    agent_x = clampf(agent_x, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_X_POS);
    agent_y = clampf(agent_y, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Y_POS);
    agent_z = clampf(agent_z, COORD_BOUNDARY_Z_NEG, COORD_BOUNDARY_Z_POS);

    // Move anchor points with the same actual cell displacement (after clamp)
    const float dx_cell = agent_x - agent_x_prev;
    const float dy_cell = agent_y - agent_y_prev;
    const float dz_cell = agent_z - agent_z_prev;
    for (int i = 0; i < N_ANCHOR_POINTS; i++) {
      agent_x_i[i] += dx_cell;
      agent_y_i[i] += dy_cell;
      agent_z_i[i] += dz_cell;
    }
  }

  float dx_track = agent_x - agent_x_prev;
  float dy_track = agent_y - agent_y_prev;
  float dz_track = agent_z - agent_z_prev;
  if (PERIODIC_BOUNDARIES_FOR_CELLS == 1 && INCLUDE_FOCAL_ADHESIONS == 0) {
    dx_track = agent_vx * TIME_STEP;
    dy_track = agent_vy * TIME_STEP;
    dz_track = agent_vz * TIME_STEP;
  }
  trajectory_length += sqrtf(dx_track * dx_track + dy_track * dy_track + dz_track * dz_track);
  trajectory_time += TIME_STEP;

  // Set agent variables
  FLAMEGPU->setVariable<int>("id", agent_id);
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);
  for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("x_i", i, agent_x_i[i]);
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("y_i", i, agent_y_i[i]);
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("z_i", i, agent_z_i[i]);
  }
  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);
  FLAMEGPU->setVariable<float>("trajectory_length", trajectory_length);
  FLAMEGPU->setVariable<float>("trajectory_time", trajectory_time);
  FLAMEGPU->setVariable<float>("orx", agent_orx);
  FLAMEGPU->setVariable<float>("ory", agent_ory);
  FLAMEGPU->setVariable<float>("orz", agent_orz);

  // Persist separate adaptation states for promotive and inhibitory chemokinesis
  for (int s = 0; s < N_SPECIES; s++) {
    FLAMEGPU->setVariable<float, N_SPECIES>(
        "chemokinesis_promotive_adapt_state", s, chemokinesis_promotive_adapt_state[s]);
    FLAMEGPU->setVariable<float, N_SPECIES>(
        "chemokinesis_inhibitory_adapt_state", s, chemokinesis_inhibitory_adapt_state[s]);
  }

  return flamegpu::ALIVE;
}