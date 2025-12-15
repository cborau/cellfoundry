// exposes CELL agents location data
FLAMEGPU_AGENT_FUNCTION(cell_output_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
  FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
  FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));
  // Agent array variables
  const uint8_t k_consumption_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < k_consumption_ARRAY_SIZE; i++) {
    float ncol = FLAMEGPU->getVariable<float, k_consumption_ARRAY_SIZE>("k_consumption", i);
    FLAMEGPU->message_out.setVariable<float, k_consumption_ARRAY_SIZE>("k_consumption", i, ncol);
  }
  // Agent array variables
  const uint8_t k_production_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < k_production_ARRAY_SIZE; i++) {
    float ncol = FLAMEGPU->getVariable<float, k_production_ARRAY_SIZE>("k_production", i);
    FLAMEGPU->message_out.setVariable<float, k_production_ARRAY_SIZE>("k_production", i, ncol);
  }
  // Agent array variables
  const uint8_t C_sp_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < C_sp_ARRAY_SIZE; i++) {
    float ncol = FLAMEGPU->getVariable<float, C_sp_ARRAY_SIZE>("C_sp", i);
    FLAMEGPU->message_out.setVariable<float, C_sp_ARRAY_SIZE>("C_sp", i, ncol);
  }  

  return flamegpu::ALIVE;
}