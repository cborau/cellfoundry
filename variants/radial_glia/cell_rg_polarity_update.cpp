/**
 * cell_rg_polarity_update  [RG variant]
 *
 * Purpose:
 *   Update each cell's apical unit vector (apx, apy, apz).
 *   NPC/RG cells inside the morphogen field undergo a gradual exponential blend
 *   toward (0,0,1), scaled by epithelialization level.
 *   All other cells (iPSC, or outside field) receive small random xy drift.
 *
 * Layer: L6b_RG_Polarity_Update - runs after ECM diffusion (fully updated field).
 *
 * Message input: none (reads ECM macro-property directly).
 *
 * Env properties used:
 *   RG_POLARITY_SP2_THRESHOLD  [uM]     - sp2 concentration gate
 *   RG_INTRINSIC_APICAL_Z      [-/step] - blend alpha per step (half-life = ln2/alpha steps)
 *   RG_APICAL_NOISE_AMP        [-/step] - xy noise std dev for cells outside the gate
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
  const int   agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");
  float apx = FLAMEGPU->getVariable<float>("apx");
  float apy = FLAMEGPU->getVariable<float>("apy");
  float apz = FLAMEGPU->getVariable<float>("apz");

  // -------------------------------------------------------------------------
  // Environment
  // -------------------------------------------------------------------------
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
  // Polarity update gated on local sp2 concentration and cell type.
  // NPC (cell_type==1) are included: they have pseudo-stratified polarity and
  // RG_APICAL_BIAS_NPC in cell_move already applies a matching small lift.
  // -------------------------------------------------------------------------
  const float RG_POLARITY_SP2_THRESHOLD = FLAMEGPU->environment.getProperty<float>("RG_POLARITY_SP2_THRESHOLD");
  const float RG_INTRINSIC_APICAL_Z     = FLAMEGPU->environment.getProperty<float>("RG_INTRINSIC_APICAL_Z");

  float new_apx = apx;
  float new_apy = apy;
  float new_apz = apz;

  if (RG_INTRINSIC_APICAL_Z > 1e-9f) {
    if (agent_cell_type >= 1 && C0 > RG_POLARITY_SP2_THRESHOLD) {
      // Exponential blend toward (0,0,1), alpha per step = RG_INTRINSIC_APICAL_Z * epi.
      // With default 3e-4 and epi=1: half-life = ln(2)/3e-4 ~ 2310 steps ~ 38 h.
      const float epi   = FLAMEGPU->getVariable<float>("epithelialization_level");
      const float alpha = RG_INTRINSIC_APICAL_Z * epi;
      new_apx *= (1.0f - alpha);
      new_apy *= (1.0f - alpha);
      new_apz  = (1.0f - alpha) * new_apz + alpha;
    } else {
      // Random xy drift for iPSC and cells outside the morphogen field.
      const float RG_APICAL_NOISE_AMP = FLAMEGPU->environment.getProperty<float>("RG_APICAL_NOISE_AMP");
      new_apx += RG_APICAL_NOISE_AMP * FLAMEGPU->random.normal<float>();
      new_apy += RG_APICAL_NOISE_AMP * FLAMEGPU->random.normal<float>();
    }
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
