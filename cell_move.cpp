// -----------------------------------------------------------------------------// Device helper functions (no lambdas, no auto)
// -----------------------------------------------------------------------------
FLAMEGPU_DEVICE_FUNCTION float clampf(const float x, const float lo, const float hi) {
  return fminf(hi, fmaxf(lo, x));
}

FLAMEGPU_DEVICE_FUNCTION int clampi(const int x, const int lo, const int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
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

// -----------------------------------------------------------------------------
// computes CELL agent movement
// -----------------------------------------------------------------------------
FLAMEGPU_AGENT_FUNCTION(cell_move, flamegpu::MessageNone, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  const float agent_x_prev = agent_x;
  const float agent_y_prev = agent_y;
  const float agent_z_prev = agent_z;

  // Velocity contributions are accumulated and applied at the end
  float agent_vx = 0.0f;
  float agent_vy = 0.0f;
  float agent_vz = 0.0f;

  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: must match main python
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

  // Orientation (assumed updated elsewhere)
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
  const int Nx = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",0);
  const int Ny = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",1);
  const int Nz = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",2);

  // Get position of the boundaries
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);

  // Chemotaxis controls
  const int INCLUDE_CHEMOTAXIS = FLAMEGPU->environment.getProperty<int>("INCLUDE_CHEMOTAXIS");
  const int CHEMOTAXIS_ONLY_DIR = FLAMEGPU->environment.getProperty<int>("CHEMOTAXIS_ONLY_DIR"); // 1: change direction only
  const float CHEMOTAXIS_CHI = FLAMEGPU->environment.getProperty<float>("CHEMOTAXIS_CHI");

  // Durotaxis controls
  const int INCLUDE_DUROTAXIS = FLAMEGPU->environment.getProperty<int>("INCLUDE_DUROTAXIS");
  const int DUROTAXIS_ONLY_DIR = FLAMEGPU->environment.getProperty<int>("DUROTAXIS_ONLY_DIR"); // 1: change direction only
  const float FOCAD_MOBILITY_MU = FLAMEGPU->environment.getProperty<float>("FOCAD_MOBILITY_MU");

  // Recommended additional controls for the blended model
  const float DUROTAXIS_BLEND_BETA = FLAMEGPU->environment.getProperty<float>("DUROTAXIS_BLEND_BETA"); // 0..1
  const int DUROTAXIS_USE_STRESS = FLAMEGPU->environment.getProperty<int>("DUROTAXIS_USE_STRESS");     // 1: stress, 0: strain

  // ---------------------------------------------------------------------------
  // Intermediate velocity accumulation
  // ---------------------------------------------------------------------------
  float v_base_x = 0.0f, v_base_y = 0.0f, v_base_z = 0.0f;     // base velocity (Brownian)
  float steer_x  = 0.0f, steer_y  = 0.0f, steer_z  = 0.0f;     // direction-only accumulators
  float dv_x     = 0.0f, dv_y     = 0.0f, dv_z     = 0.0f;     // speed-changing accumulators

  // Brownian motion (base component)
  const float agent_speed_ref = FLAMEGPU->getVariable<float>("speed_ref");
  const float BROWNIAN_MOTION_STRENGTH = FLAMEGPU->environment.getProperty<float>("BROWNIAN_MOTION_STRENGTH");
  v_base_x += agent_speed_ref * BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0, 1.0));
  v_base_y += agent_speed_ref * BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0, 1.0));
  v_base_z += agent_speed_ref * BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0, 1.0));

  float chemotaxis_sensitivity[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    chemotaxis_sensitivity[i] = FLAMEGPU->getVariable<float, N_SPECIES>("chemotaxis_sensitivity", i);
  }

  // ---------------------------------------------------------------------------
  // transform x,y,z positions to i,j,k grid positions
  // ---------------------------------------------------------------------------
  int agent_grid_i = roundf(((agent_x - COORD_BOUNDARY_X_NEG) / (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG)) * (Nx - 1));
  int agent_grid_j = roundf(((agent_y - COORD_BOUNDARY_Y_NEG) / (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG)) * (Ny - 1));
  int agent_grid_k = roundf(((agent_z - COORD_BOUNDARY_Z_NEG) / (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG)) * (Nz - 1));

  agent_grid_i = clampi(agent_grid_i, 0, Nx - 1);
  agent_grid_j = clampi(agent_grid_j, 0, Ny - 1);
  agent_grid_k = clampi(agent_grid_k, 0, Nz - 1);

  // ---------------------------------------------------------------------------
  // CHEMOTAXIS: compute direction and add to steer or dv
  // ---------------------------------------------------------------------------
  if (INCLUDE_CHEMOTAXIS) {
    const float dx = (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG) / (Nx - 1);
    const float dy = (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG) / (Ny - 1);
    const float dz = (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG) / (Nz - 1);

    const uint32_t c_idx = macro_lin_idx(agent_grid_i, agent_grid_j, agent_grid_k, Ny, Nz);

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
        printf("Agent %d: chemotaxis steer=(%.3f, %.3f, %.3f)\n", agent_id, CHEMOTAXIS_CHI * chemo_dir_x, CHEMOTAXIS_CHI * chemo_dir_y, CHEMOTAXIS_CHI * chemo_dir_z);
      } else {
        dv_x += CHEMOTAXIS_CHI * chemo_dir_x;
        dv_y += CHEMOTAXIS_CHI * chemo_dir_y;
        dv_z += CHEMOTAXIS_CHI * chemo_dir_z;
        printf("Agent %d: chemotaxis dv=(%.3f, %.3f, %.3f)\n", agent_id, CHEMOTAXIS_CHI * chemo_dir_x, CHEMOTAXIS_CHI * chemo_dir_y, CHEMOTAXIS_CHI * chemo_dir_z);
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
    printf("agent_sig=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f) eps=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)\n", agent_sig_xx, agent_sig_yy, agent_sig_zz, agent_sig_xy, agent_sig_xz, agent_sig_yz,
                                                                                                                        agent_eps_xx, agent_eps_yy, agent_eps_zz, agent_eps_xy, agent_eps_xz, agent_eps_yz);
    float scale_energy = agent_sig_xx*agent_eps_xx + agent_sig_yy*agent_eps_yy + agent_sig_zz*agent_eps_zz
                       + 2.0f*(agent_sig_xy*agent_eps_xy + agent_sig_xz*agent_eps_xz + agent_sig_yz*agent_eps_yz);
    if (scale_energy < 0.0f) scale_energy = 0.0f;

    // Unitless anisotropy factor from principal values (vanishes when nearly isotropic)
    const float aniso_den = fabsf(l1) + fabsf(l2) + fabsf(l3) + 1e-12f;
    float A = (l1 - l3) / aniso_den; // unitless, 0..1, vanishes when l1~l2~l3, increases as l1>>l2,l3; 

    // Final durotaxis strength
    printf("Agent %d: durotaxis scale_energy=%.3f A=%.3f\n", agent_id, scale_energy, A);
    float duro_strength = FOCAD_MOBILITY_MU * (scale_energy + A);
    // If both are tiny, keep a minimal fallback based on |l1|
    if (duro_strength < 1e-12f) {
      duro_strength = FOCAD_MOBILITY_MU * fabsf(l1);
    }

    if (DUROTAXIS_ONLY_DIR == 1) {
      steer_x += duro_strength * duro_dir_x;
      steer_y += duro_strength * duro_dir_y;
      steer_z += duro_strength * duro_dir_z;
      printf("Agent %d: durotaxis steer=(%.3f, %.3f, %.3f)\n", agent_id, duro_strength * duro_dir_x, duro_strength * duro_dir_y, duro_strength * duro_dir_z);
    } else {
      dv_x += duro_strength * duro_dir_x;
      dv_y += duro_strength * duro_dir_y;
      dv_z += duro_strength * duro_dir_z;
      printf("Agent %d: durotaxis dv=(%.3f, %.3f, %.3f)\n", agent_id, duro_strength * duro_dir_x, duro_strength * duro_dir_y, duro_strength * duro_dir_z);
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
    if (vmag < 1e-6f) vmag = agent_speed_ref;

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
    printf("Agent %d: steer=(%.3f, %.3f, %.3f) v=(%.3f, %.3f, %.3f)\n", agent_id, steer_x, steer_y, steer_z, agent_vx, agent_vy, agent_vz);
  }

  // Add speed-changing contributions
  agent_vx += dv_x;
  agent_vy += dv_y;
  agent_vz += dv_z;

  // ---------------------------------------------------------------------------
  // Update agent position based on velocity
  // ---------------------------------------------------------------------------
  agent_x += agent_vx * TIME_STEP;
  agent_y += agent_vy * TIME_STEP;
  agent_z += agent_vz * TIME_STEP;

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

  //Set agent variables
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

  return flamegpu::ALIVE;
}
