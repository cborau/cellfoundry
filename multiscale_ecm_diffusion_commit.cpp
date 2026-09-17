/**
 * multiscale_ecm_diffusion_commit
 *
 * Purpose:
 *   Copy final subcycled ECM concentrations into the shared macro property.
 *
 * Inputs:
 *   - Agent grid_lin_id and C_sp after every diffusion group has finished
 *
 * Outputs:
 *   - C_SP_MACRO updated for every species at this agent's grid position
 *
 * Notes:
 *   Runs once per main timestep. This is the reverse direction to
 *   ecm_Csp_update, which imports cellular exchange from C_SP_MACRO into C_sp.
 */
FLAMEGPU_AGENT_FUNCTION(multiscale_ecm_diffusion_commit, flamegpu::MessageNone, flamegpu::MessageNone) {
  // Agent array variables and macro-property dimensions
  const uint8_t N_SPECIES = 3; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  const uint32_t ECM_POPULATION_SIZE = 61206; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  const int agent_grid_lin_id = FLAMEGPU->getVariable<int>("grid_lin_id");
  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");
  for (int i = 0; i < N_SPECIES; i++) {
    const float agent_C_sp = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
    // Each ECM agent owns one macro-property column. 
    C_SP_MACRO[i][agent_grid_lin_id].exchange(agent_C_sp);
  }
  return flamegpu::ALIVE;
}
