// computes CELL agent movement
FLAMEGPU_AGENT_FUNCTION(cell_move, flamegpu::MessageNone, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");
  // Agent array variables
  const uint8_t k_consumption_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float k_consumption[k_consumption_ARRAY_SIZE] = {};
  for (int i = 0; i < k_consumption_ARRAY_SIZE; i++) {
    k_consumption[i] = FLAMEGPU->getVariable<float, k_consumption_ARRAY_SIZE>("k_consumption", i);
  }
  // Agent array variables
  const uint8_t k_production_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float k_production[k_production_ARRAY_SIZE] = {};
  for (int i = 0; i < k_production_ARRAY_SIZE; i++) {
    k_production[i] = FLAMEGPU->getVariable<float, k_production_ARRAY_SIZE>("k_production", i);
  }
  // Agent array variables
  const uint8_t C_sp_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float C_sp[C_sp_ARRAY_SIZE] = {};
  for (int i = 0; i < C_sp_ARRAY_SIZE; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, C_sp_ARRAY_SIZE>("C_sp", i);
  }



  //Set agent variables
  FLAMEGPU->setVariable<int>("id", agent_id);
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);
  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);
  // Agent array variables
  const uint8_t k_consumption_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < k_consumption_ARRAY_SIZE; i++) {
    FLAMEGPU->setVariable<float, k_consumption_ARRAY_SIZE>("k_consumption", i, k_consumption[i]);
  }
  // Agent array variables
  const uint8_t k_production_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < k_production_ARRAY_SIZE; i++) {
    FLAMEGPU->setVariable<float, k_production_ARRAY_SIZE>("k_production", i, k_production[i]);
  }
  // Agent array variables
  const uint8_t C_sp_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < C_sp_ARRAY_SIZE; i++) {
    FLAMEGPU->setVariable<float, C_sp_ARRAY_SIZE>("C_sp", i, C_sp[i]);
  }


  return flamegpu::ALIVE;
}