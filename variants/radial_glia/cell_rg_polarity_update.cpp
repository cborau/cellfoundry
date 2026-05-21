/**
 * cell_rg_polarity_update  [RG variant]
 *
 * Purpose:
 *   Update each cell's apical unit vector (apx, apy, apz).
 *
 *   Two-component update for NPC/RG cells inside the morphogen field:
 *
 *   1. Vertical (Z) component: exponential blend toward (0,0,1).
 *      alpha_z = RG_INTRINSIC_APICAL_Z * commit_level
 *      (NPC: multiplied by epi_level * 0.1)
 *
 *   2. Lateral (XY) lumen cue: if >= RG_LUMEN_MIN_NEIGHBOURS alive RG (type-2)
 *      cells are found within RG_LUMEN_SEARCH_RADIUS, the apical XY component
 *      is steered toward the centroid of those neighbours (rosette centre proxy).
 *      beta_xy = RG_LUMEN_BIAS_STRENGTH * commit_level
 *      (NPC: multiplied by epi_level * 0.1)
 *      Falls back to XY suppression (existing behaviour) when no RG neighbours.
 *
 *   iPSC / cells outside morphogen gate: random XY noise (unchanged).
 *
 * Layer: L6b_RG_Polarity_Update — after ECM diffusion and L1 spatial broadcast.
 *
 * Message input: cell_spatial_location_message (MessageSpatial3D)
 *
 * Env properties used:
 *   RG_POLARITY_SP2_THRESHOLD  [uM]     — sp2 concentration gate
 *   RG_INTRINSIC_APICAL_Z      [-/step] — Z blend alpha per step
 *   RG_APICAL_NOISE_AMP        [-/step] — XY noise std dev outside gate
 *   RG_LUMEN_BIAS_STRENGTH     [-/step] — XY lumen-cue blend strength per step
 *   RG_LUMEN_SEARCH_RADIUS     [um]     — search radius for RG centroid
 *   RG_LUMEN_MIN_NEIGHBOURS    [-]      — minimum RG neighbours to activate lumen cue
 */
FLAMEGPU_AGENT_FUNCTION(cell_rg_polarity_update, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    return flamegpu::ALIVE;
  }

  // -------------------------------------------------------------------------
  // Own state
  // -------------------------------------------------------------------------
  const int   agent_id        = FLAMEGPU->getVariable<int>("id");
  const float agent_x         = FLAMEGPU->getVariable<float>("x");
  const float agent_y         = FLAMEGPU->getVariable<float>("y");
  const float agent_z         = FLAMEGPU->getVariable<float>("z");
  const int   agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");
  float apx = FLAMEGPU->getVariable<float>("apx");
  float apy = FLAMEGPU->getVariable<float>("apy");
  float apz = FLAMEGPU->getVariable<float>("apz");

  // -------------------------------------------------------------------------
  // Environment
  // -------------------------------------------------------------------------
  const uint8_t  N_SPECIES           = 3;     // WARNING: must match main python
  const uint32_t ECM_POPULATION_SIZE = 9261;  // WARNING: must match Nx*Ny*Nz

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

  // -------------------------------------------------------------------------
  // Map agent position to nearest ECM voxel and sample local sp2
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
  // Env properties
  // -------------------------------------------------------------------------
  const float RG_POLARITY_SP2_THRESHOLD = FLAMEGPU->environment.getProperty<float>("RG_POLARITY_SP2_THRESHOLD");
  const float RG_INTRINSIC_APICAL_Z     = FLAMEGPU->environment.getProperty<float>("RG_INTRINSIC_APICAL_Z");
  const float RG_LUMEN_BIAS_STRENGTH    = FLAMEGPU->environment.getProperty<float>("RG_LUMEN_BIAS_STRENGTH");
  const float RG_LUMEN_SEARCH_RADIUS    = FLAMEGPU->environment.getProperty<float>("RG_LUMEN_SEARCH_RADIUS");
  const int   RG_LUMEN_MIN_NEIGHBOURS   = (int)FLAMEGPU->environment.getProperty<float>("RG_LUMEN_MIN_NEIGHBOURS");

  float new_apx = apx;
  float new_apy = apy;
  float new_apz = apz;

  if (RG_INTRINSIC_APICAL_Z > 1e-9f) {
    if (agent_cell_type >= 1 && C0 > RG_POLARITY_SP2_THRESHOLD) {
      // -------------------------------------------------------------------
      // Collect nearby alive RG (type-2) cells to estimate the local
      // rosette centre as their XY centroid (lumen proxy).
      // -------------------------------------------------------------------
      float sum_x = 0.0f;
      float sum_y = 0.0f;
      int   n_rg  = 0;
      const float r2_max = RG_LUMEN_SEARCH_RADIUS * RG_LUMEN_SEARCH_RADIUS;

      for (const auto& msg : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        if (msg.getVariable<int>("dead")      != 0)        continue;
        if (msg.getVariable<int>("id")        == agent_id) continue;
        if (msg.getVariable<int>("cell_type") != 2)        continue;

        const float dx = msg.getVariable<float>("x") - agent_x;
        const float dy = msg.getVariable<float>("y") - agent_y;
        const float dz = msg.getVariable<float>("z") - agent_z;
        if (dx*dx + dy*dy + dz*dz > r2_max) continue;

        sum_x += msg.getVariable<float>("x");
        sum_y += msg.getVariable<float>("y");
        n_rg++;
      }

      // -------------------------------------------------------------------
      // Blend factors (commit-scaled; NPC gets 10x weaker signal)
      // -------------------------------------------------------------------
      const float epi    = FLAMEGPU->getVariable<float>("epithelialization_level");
      const float commit = FLAMEGPU->getVariable<float>("rg_commit_level");
      const float scale  = (agent_cell_type == 2) ? commit : epi * commit * 0.1f;
      const float alpha_z = RG_INTRINSIC_APICAL_Z  * scale;
      const float beta_xy = RG_LUMEN_BIAS_STRENGTH * scale;

      if (n_rg >= RG_LUMEN_MIN_NEIGHBOURS) {
        // Lumen cue active: steer XY toward centroid, blend Z upward.
        const float cx = sum_x / (float)n_rg - agent_x;
        const float cy = sum_y / (float)n_rg - agent_y;
        const float l2 = cx*cx + cy*cy;

        if (l2 > 1e-12f) {
          const float inv_l = rsqrtf(l2);
          new_apx = (1.0f - beta_xy) * apx + beta_xy * (cx * inv_l);
          new_apy = (1.0f - beta_xy) * apy + beta_xy * (cy * inv_l);
        } else {
          // Cell is at exact centroid (lumen centre): suppress XY.
          new_apx *= (1.0f - alpha_z);
          new_apy *= (1.0f - alpha_z);
        }
        new_apz = (1.0f - alpha_z) * apz + alpha_z;
      } else {
        // No qualifying RG neighbours: fall back to pure Z-blend.
        new_apx *= (1.0f - alpha_z);
        new_apy *= (1.0f - alpha_z);
        new_apz  = (1.0f - alpha_z) * apz + alpha_z;
      }
    } else {
      // Random XY drift for iPSC and cells outside the morphogen gate.
      const float RG_APICAL_NOISE_AMP = FLAMEGPU->environment.getProperty<float>("RG_APICAL_NOISE_AMP");
      new_apx += RG_APICAL_NOISE_AMP * FLAMEGPU->random.normal<float>();
      new_apy += RG_APICAL_NOISE_AMP * FLAMEGPU->random.normal<float>();
    }

    // Renormalize
    const float n2 = new_apx*new_apx + new_apy*new_apy + new_apz*new_apz;
    if (n2 > 1e-20f) {
      const float inv_n = rsqrtf(n2);
      new_apx *= inv_n;
      new_apy *= inv_n;
      new_apz *= inv_n;
    }
  }

  // -------------------------------------------------------------------------
  // Write back
  // -------------------------------------------------------------------------
  FLAMEGPU->setVariable<float>("apx", new_apx);
  FLAMEGPU->setVariable<float>("apy", new_apy);
  FLAMEGPU->setVariable<float>("apz", new_apz);

  return flamegpu::ALIVE;
}
