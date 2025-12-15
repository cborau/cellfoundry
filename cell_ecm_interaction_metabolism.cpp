// defines interactions with ECM agents and computes metabolism of species
FLAMEGPU_AGENT_FUNCTION(cell_ecm_interaction_metabolism, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
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

  //Define message variables (agent sending the input message)
  int message_id = 0;
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;
  uint8_t message_grid_i = 0;
  uint8_t message_grid_j = 0;
  uint8_t message_grid_k = 0;
  const uint8_t message_C_sp_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float message_C_sp[message_C_sp_ARRAY_SIZE] = {};
  float message_k_elast = 0.0;
  uint8_t message_d_dumping = 0;
  float message_vx = 0.0;
  float message_vy = 0.0;
  float message_vz = 0.0;

  //Loop through all agents sending input messages
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    message_id = message.getVariable<int>("id");
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_z = message.getVariable<float>("z");
    message_grid_i = message.getVariable<uint8_t>("grid_i");
    message_grid_j = message.getVariable<uint8_t>("grid_j");
    message_grid_k = message.getVariable<uint8_t>("grid_k");
    for (int i = 0; i < message_C_sp_ARRAY_SIZE; i++) {
      message_C_sp[i] = message.getVariable<float, message_C_sp_ARRAY_SIZE>("C_sp", i);
    }
    message_k_elast = message.getVariable<float>("k_elast");
    message_d_dumping = message.getVariable<uint8_t>("d_dumping");
    message_vx = message.getVariable<float>("vx");
    message_vy = message.getVariable<float>("vy");
    message_vz = message.getVariable<float>("vz");
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