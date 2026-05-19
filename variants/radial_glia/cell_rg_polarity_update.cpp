/**
 * cell_rg_polarity_update  [RG variant — new function]
 *
 * Purpose:
 *   Maintain each cell's apical vector (apx, apy, apz) by relaxing it toward
 *   the local gradient direction of ECM morphogen species 2.  The 26-neighbour
 *   finite-difference stencil mirrors the chemotaxis stencil in cell_move.cpp.
 *
 * Layer: L6b_RG_Polarity_Update — after L6 (ECM boundary; fully updated
 *        diffusion field for this step).
 *
 * Message input: none (reads ECM macro-property directly).
 *
 * New env properties (registered in variants/radial_glia/__init__.py):
 *   RG_POLARITY_TAU                  [s]       — exponential blend timescale
 *   RG_POLARITY_GRADIENT_THRESHOLD   [a.u./µm] — minimum gradient magnitude
 *                                                 required to update polarity
 */
FLAMEGPU_AGENT_FUNCTION(cell_rg_polarity_update, flamegpu::MessageNone, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    return flamegpu::ALIVE;
  }

  // -------------------------------------------------------------------------
  // Own state
  // -------------------------------------------------------------------------
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  float apx = FLAMEGPU->getVariable<float>("apx");
  float apy = FLAMEGPU->getVariable<float>("apy");
  float apz = FLAMEGPU->getVariable<float>("apz");

  // -------------------------------------------------------------------------
  // Environment
  // -------------------------------------------------------------------------
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");

  const uint8_t N_SPECIES = 3;   // WARNING: must match main python
  const uint32_t ECM_POPULATION_SIZE = 9261; // WARNING: must match Nx*Ny*Nz

  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");

  const int Nx = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 0);
  const int Ny = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 1);
  const int Nz = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 2);

  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 5);

  const float RG_POLARITY_TAU                = FLAMEGPU->environment.getProperty<float>("RG_POLARITY_TAU");
  const float RG_POLARITY_GRADIENT_THRESHOLD = FLAMEGPU->environment.getProperty<float>("RG_POLARITY_GRADIENT_THRESHOLD");

  // Voxel spacings
  const float dx = (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG) / (float)(Nx - 1);
  const float dy = (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG) / (float)(Ny - 1);
  const float dz = (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG) / (float)(Nz - 1);

  // -------------------------------------------------------------------------
  // Map agent position to ECM voxel
  // -------------------------------------------------------------------------
  int gi = (int)roundf(((agent_x - COORD_BOUNDARY_X_NEG) / (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG)) * (Nx - 1));
  int gj = (int)roundf(((agent_y - COORD_BOUNDARY_Y_NEG) / (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG)) * (Ny - 1));
  int gk = (int)roundf(((agent_z - COORD_BOUNDARY_Z_NEG) / (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG)) * (Nz - 1));
  gi = (gi < 0) ? 0 : (gi >= Nx ? Nx - 1 : gi);
  gj = (gj < 0) ? 0 : (gj >= Ny ? Ny - 1 : gj);
  gk = (gk < 0) ? 0 : (gk >= Nz ? Nz - 1 : gk);

  const uint32_t c_idx = (uint32_t)(gi * (Ny * Nz) + gj * Nz + gk);
  const float C0 = (float)C_SP_MACRO[2][c_idx];

  // -------------------------------------------------------------------------
  // 26-neighbour gradient stencil (same as chemotaxis in cell_move.cpp)
  // -------------------------------------------------------------------------
  float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

  for (int dk = -1; dk <= 1; dk++) {
    for (int dj = -1; dj <= 1; dj++) {
      for (int di = -1; di <= 1; di++) {
        if (di == 0 && dj == 0 && dk == 0) continue;

        const int ni = (gi + di < 0) ? 0 : (gi + di >= Nx ? Nx - 1 : gi + di);
        const int nj = (gj + dj < 0) ? 0 : (gj + dj >= Ny ? Ny - 1 : gj + dj);
        const int nk = (gk + dk < 0) ? 0 : (gk + dk >= Nz ? Nz - 1 : gk + dk);

        const uint32_t n_idx = (uint32_t)(ni * (Ny * Nz) + nj * Nz + nk);

        const float ddx = (float)di * dx;
        const float ddy = (float)dj * dy;
        const float ddz = (float)dk * dz;

        const float dist2 = ddx*ddx + ddy*ddy + ddz*ddz + 1e-12f;
        const float inv_dist = rsqrtf(dist2);

        const float Cn = (float)C_SP_MACRO[2][n_idx];
        const float dC = Cn - C0;
        const float w  = dC * inv_dist;   // [a.u./µm]  weight per unit direction

        grad_x += w * (ddx * inv_dist);
        grad_y += w * (ddy * inv_dist);
        grad_z += w * (ddz * inv_dist);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Decide target direction
  // -------------------------------------------------------------------------
  const float g2 = grad_x*grad_x + grad_y*grad_y + grad_z*grad_z;
  const float g_mag = sqrtf(g2 + 1e-20f);

  float target_x = apx;
  float target_y = apy;
  float target_z = apz;

  if (g_mag > RG_POLARITY_GRADIENT_THRESHOLD) {
    const float inv_g = 1.0f / g_mag;
    target_x = grad_x * inv_g;
    target_y = grad_y * inv_g;
    target_z = grad_z * inv_g;
  }

  // -------------------------------------------------------------------------
  // Exponential blend: ap_new = lerp(ap, target, alpha)
  //   alpha = 1 - exp(-dt / tau)
  // -------------------------------------------------------------------------
  const float tau = fmaxf(RG_POLARITY_TAU, 1e-6f);
  const float alpha = 1.0f - expf(-TIME_STEP / tau);

  float new_apx = (1.0f - alpha) * apx + alpha * target_x;
  float new_apy = (1.0f - alpha) * apy + alpha * target_y;
  float new_apz = (1.0f - alpha) * apz + alpha * target_z;

  // Renormalize
  const float n2 = new_apx*new_apx + new_apy*new_apy + new_apz*new_apz;
  if (n2 > 1e-20f) {
    const float inv_n = rsqrtf(n2);
    new_apx *= inv_n;
    new_apy *= inv_n;
    new_apz *= inv_n;
  } else {
    // Degenerate: keep current polarity
    new_apx = apx;
    new_apy = apy;
    new_apz = apz;
  }

  // -------------------------------------------------------------------------
  // Write back
  // -------------------------------------------------------------------------
  FLAMEGPU->setVariable<float>("apx", new_apx);
  FLAMEGPU->setVariable<float>("apy", new_apy);
  FLAMEGPU->setVariable<float>("apz", new_apz);

  return flamegpu::ALIVE;
}
