// exposes x,y,z position of the ECM grid agents
FLAMEGPU_AGENT_FUNCTION(ecm_output_spatial_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<uint8_t>("grid_i", FLAMEGPU->getVariable<uint8_t>("grid_i"));
  FLAMEGPU->message_out.setVariable<uint8_t>("grid_j", FLAMEGPU->getVariable<uint8_t>("grid_j"));
  FLAMEGPU->message_out.setVariable<uint8_t>("grid_k", FLAMEGPU->getVariable<uint8_t>("grid_k"));
  // Agent array variables
  const uint8_t C_sp_ARRAY_SIZE = ?; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < C_sp_ARRAY_SIZE; i++) {
    float ncol = FLAMEGPU->getVariable<float, C_sp_ARRAY_SIZE>("C_sp", i);
    FLAMEGPU->message_out.setVariable<float, C_sp_ARRAY_SIZE>("C_sp", i, ncol);
  }
  FLAMEGPU->message_out.setVariable<float>("k_elast", FLAMEGPU->getVariable<float>("k_elast"));
  FLAMEGPU->message_out.setVariable<uint8_t>("d_dumping", FLAMEGPU->getVariable<uint8_t>("d_dumping"));
  FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
  FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
  FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));  

  return flamegpu::ALIVE;
}