// interacts with boundaries if these are moving
FLAMEGPU_AGENT_FUNCTION(ecm_boundary_interaction, flamegpu::MessageNone, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  uint8_t agent_grid_i = FLAMEGPU->getVariable<uint8_t>("grid_i");
  uint8_t agent_grid_j = FLAMEGPU->getVariable<uint8_t>("grid_j");
  uint8_t agent_grid_k = FLAMEGPU->getVariable<uint8_t>("grid_k");
  // Agent array variables
  const uint8_t C_sp_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float C_sp[C_sp_ARRAY_SIZE] = {};
  for (int i = 0; i < C_sp_ARRAY_SIZE; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, C_sp_ARRAY_SIZE>("C_sp", i);
  }
  float agent_k_elast = FLAMEGPU->getVariable<float>("k_elast");
  uint8_t agent_d_dumping = FLAMEGPU->getVariable<uint8_t>("d_dumping");
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");



  //Set agent variables
  FLAMEGPU->setVariable<int>("id", agent_id);
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);
  FLAMEGPU->setVariable<uint8_t>("grid_i", agent_grid_i);
  FLAMEGPU->setVariable<uint8_t>("grid_j", agent_grid_j);
  FLAMEGPU->setVariable<uint8_t>("grid_k", agent_grid_k);
  // Agent array variables
  const uint8_t C_sp_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < C_sp_ARRAY_SIZE; i++) {
    FLAMEGPU->setVariable<float, C_sp_ARRAY_SIZE>("C_sp", i, C_sp[i]);
  }
  FLAMEGPU->setVariable<float>("k_elast", agent_k_elast);
  FLAMEGPU->setVariable<uint8_t>("d_dumping", agent_d_dumping);
  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);


  return flamegpu::ALIVE;
}