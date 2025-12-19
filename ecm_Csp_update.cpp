// updates internal values by reading a Macro property, previously modified by CELL agents
FLAMEGPU_AGENT_FUNCTION(ecm_Csp_update, flamegpu::MessageNone, flamegpu::MessageNone) {
  
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  int agent_grid_lin_id = FLAMEGPU->getVariable<int>("grid_lin_id");

  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  const uint8_t ECM_POPULATION_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");

  for (int i = 0; i < N_SPECIES; i++) {
    FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, (float)C_SP_MACRO[i][agent_grid_lin_id]);
  }

  return flamegpu::ALIVE;
}