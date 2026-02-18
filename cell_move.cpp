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
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");

  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
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
  const int CHEMOTAXIS_ONLY_DIR = FLAMEGPU->environment.getProperty<int>("CHEMOTAXIS_ONLY_DIR"); // if true, chemotaxis does not affect speed
  const float CHEMOTAXIS_CHI = FLAMEGPU->environment.getProperty<float>("CHEMOTAXIS_CHI");

  // Brownian motion
  const float agent_speed_ref = FLAMEGPU->getVariable<float>("speed_ref");
  const float BROWNIAN_MOTION_STRENGTH = FLAMEGPU->environment.getProperty<float>("BROWNIAN_MOTION_STRENGTH");
  agent_vx += agent_speed_ref * BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0, 1.0));
  agent_vy += agent_speed_ref * BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0, 1.0));
  agent_vz += agent_speed_ref * BROWNIAN_MOTION_STRENGTH * (FLAMEGPU->random.uniform<float>(-1.0, 1.0));



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

  // ---------------------------------------------------------------------------
  // CHEMOTAXIS: 26-neighborhood, distance-weighted, multi-species
  // ---------------------------------------------------------------------------
  if (INCLUDE_CHEMOTAXIS) {
    // Physical spacing of the macro grid
    const float dx = (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG) / (Nx - 1);
    const float dy = (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG) / (Ny - 1);
    const float dz = (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG) / (Nz - 1);

    // Center linear index 
    const uint32_t c_idx = macro_lin_idx(agent_grid_i, agent_grid_j, agent_grid_k, Ny, Nz);

    float grad_x = 0.0f;
    float grad_y = 0.0f;
    float grad_z = 0.0f;

    for (int dk = -1; dk <= 1; dk++) {
      for (int dj = -1; dj <= 1; dj++) {
        for (int di = -1; di <= 1; di++) {

          if (di == 0 && dj == 0 && dk == 0) continue;

          const int ni = clampi(agent_grid_i + di, 0, Nx - 1);
          const int nj = clampi(agent_grid_j + dj, 0, Ny - 1);
          const int nk = clampi(agent_grid_k + dk, 0, Nz - 1);

          const uint32_t n_idx = macro_lin_idx(ni, nj, nk, Ny, Nz);

          // Physical displacement to neighbor
          const float ddx = (float)di * dx;
          const float ddy = (float)dj * dy;
          const float ddz = (float)dk * dz;

          // Distance weighting: weight ~ 1/dist
          const float dist2 = ddx*ddx + ddy*ddy + ddz*ddz + 1e-12f;
          const float inv_dist = rsqrtf(dist2); // rsqrtf is faster than 1/sqrtf

          // Unit direction
          const float ux_n = ddx * inv_dist;
          const float uy_n = ddy * inv_dist;
          const float uz_n = ddz * inv_dist;

          // Combine species contributions with per-species sensitivity in [-1, +1]
          float dC_total = 0.0f;
          for (int s = 0; s < N_SPECIES; s++) {
            const float sens = chemotaxis_sensitivity[s];
            if (sens == 0.0f) continue;

            const float Cn = C_SP_MACRO[s][n_idx];
            const float C0 = C_SP_MACRO[s][c_idx];
            dC_total += sens * (Cn - C0);
          }

          // Contribution: (dC_total / dist) along direction to neighbor
          const float w = dC_total * inv_dist;

          grad_x += w * ux_n;
          grad_y += w * uy_n;
          grad_z += w * uz_n;
        }
      }
    }
    printf("CELL %d at grid (%d, %d, %d) has chemotactic gradient (%.4f, %.4f, %.4f)\n", agent_id, agent_grid_i, agent_grid_j, agent_grid_k, grad_x, grad_y, grad_z);
    // Turn gradient into a direction (unit vector)
    const float g2 = grad_x*grad_x + grad_y*grad_y + grad_z*grad_z;
    if (g2 > 0.0f) {
      const float inv_g = rsqrtf(g2 + 1e-12f);
      const float chemo_dir_x = grad_x * inv_g;
      const float chemo_dir_y = grad_y * inv_g;
      const float chemo_dir_z = grad_z * inv_g;

      if (CHEMOTAXIS_ONLY_DIR == 1) {
        // Preserve current speed, steer direction only
        const float v2 = agent_vx*agent_vx + agent_vy*agent_vy + agent_vz*agent_vz;
        const float vmag = sqrtf(v2 + 1e-12f);

        // Current direction (if vmag is tiny, fall back to chemo direction)
        float vdir_x = agent_vx / (vmag + 1e-12f);
        float vdir_y = agent_vy / (vmag + 1e-12f);
        float vdir_z = agent_vz / (vmag + 1e-12f);

        if (vmag < 1e-6f) {
          vdir_x = chemo_dir_x;
          vdir_y = chemo_dir_y;
          vdir_z = chemo_dir_z;
        }

        // Blend directions, then renormalize, then restore speed
        float ndir_x = vdir_x + CHEMOTAXIS_CHI * chemo_dir_x;
        float ndir_y = vdir_y + CHEMOTAXIS_CHI * chemo_dir_y;
        float ndir_z = vdir_z + CHEMOTAXIS_CHI * chemo_dir_z;

        const float n2 = ndir_x*ndir_x + ndir_y*ndir_y + ndir_z*ndir_z;
        const float inv_n = rsqrtf(n2 + 1e-12f);

        ndir_x *= inv_n;
        ndir_y *= inv_n;
        ndir_z *= inv_n;

        agent_vx = vmag * ndir_x;
        agent_vy = vmag * ndir_y;
        agent_vz = vmag * ndir_z;

      } else {
        // Chemotaxis affects velocity (can change speed)
        agent_vx += CHEMOTAXIS_CHI * chemo_dir_x;
        agent_vy += CHEMOTAXIS_CHI * chemo_dir_y;
        agent_vz += CHEMOTAXIS_CHI * chemo_dir_z;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Update agent position based on velocity
  // ---------------------------------------------------------------------------
  agent_x += agent_vx * TIME_STEP;
  agent_y += agent_vy * TIME_STEP;
  agent_z += agent_vz * TIME_STEP;

  printf("CELL %d moved to (%.4f, %.4f, %.4f) with velocity (%.4f, %.4f, %.4f)\n", agent_id, agent_x, agent_y, agent_z, agent_vx, agent_vy, agent_vz);

  // Simple clamp to domain (first version)
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
