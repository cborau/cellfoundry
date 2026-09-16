/**
 * multiscale_ecm_diffusion_prepare
 *
 * Purpose:
 *   Cache boundary concentrations for the upcoming diffusion substeps and
 *   restore imposed concentrations after the main cellular exchange step.
 *
 * Inputs:
 *   - Agent x, y, z, C_sp and diffusion_vascular_floor
 *   - COORDS_BOUNDARIES and ECM_BOUNDARY_INTERACTION_RADIUS
 *   - BOUNDARY_CONC_INIT_MULTI and BOUNDARY_CONC_FIXED_MULTI macro properties
 *
 * Outputs:
 *   - Agent diffusion_boundary: prescribed concentration, or -1 if unconstrained
 *   - Agent C_sp with vascular floors and prescribed boundary values applied
 *
 * Notes:
 *   Boundary geometry and sources remain fixed during a diffusion submodel.
 *   Cache them once per main timestep. At intersecting faces the maximum
 *   prescribed concentration wins, matching ecm_boundary_concentration_conditions.
 */
FLAMEGPU_AGENT_FUNCTION(multiscale_ecm_diffusion_prepare, flamegpu::MessageNone, flamegpu::MessageNone) {
  // Agent array variables
  const uint8_t N_SPECIES = 3; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float C_sp[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
  }

  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  const float ECM_BOUNDARY_INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("ECM_BOUNDARY_INTERACTION_RADIUS");
  auto BOUNDARY_CONC_INIT_MULTI = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, 6>("BOUNDARY_CONC_INIT_MULTI");
  auto BOUNDARY_CONC_FIXED_MULTI = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, 6>("BOUNDARY_CONC_FIXED_MULTI");

  // Face order is +X, -X, +Y, -Y, +Z, -Z, as in the existing boundary function.
  const float positions[6] = {agent_x, agent_x, agent_y, agent_y, agent_z, agent_z};
  float separations[6] = {};
  for (int j = 0; j < 6; j++) {
    separations[j] = positions[j] - FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", j);
  }

  for (int i = 0; i < N_SPECIES; i++) {
    float max_conc = -1.0f; // -1 means no prescribed concentration for this species.
    for (int j = 0; j < 6; j++) {
      if (fabsf(separations[j]) < ECM_BOUNDARY_INTERACTION_RADIUS) {
        max_conc = fmaxf(max_conc, (float)BOUNDARY_CONC_INIT_MULTI[i][j]);
        max_conc = fmaxf(max_conc, (float)BOUNDARY_CONC_FIXED_MULTI[i][j]);
      }
    }
    FLAMEGPU->setVariable<float, N_SPECIES>("diffusion_boundary", i, max_conc);

    const float vascular_floor = FLAMEGPU->getVariable<float, N_SPECIES>("diffusion_vascular_floor", i);
    if (vascular_floor >= 0.0f) {
      C_sp[i] = fmaxf(C_sp[i], vascular_floor);
    }
    // A prescribed boundary concentration takes precedence over a vessel floor.
    if (max_conc >= 0.0f) {
      C_sp[i] = max_conc;
    }
    FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, C_sp[i]);
  }
  return flamegpu::ALIVE;
}
