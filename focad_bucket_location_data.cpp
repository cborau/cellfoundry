FLAMEGPU_AGENT_FUNCTION(focad_bucket_location_data, flamegpu::MessageNone, flamegpu::MessageBucket) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<int>("fnode_id", FLAMEGPU->getVariable<int>("fnode_id"));
  FLAMEGPU->message_out.setVariable<int>("cell_id", FLAMEGPU->getVariable<int>("cell_id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
  FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
  FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));
  FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
  FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
  FLAMEGPU->message_out.setVariable<float>("fz", FLAMEGPU->getVariable<float>("fz"));
  FLAMEGPU->message_out.setVariable<float>("x_i", FLAMEGPU->getVariable<float>("x_i"));
  FLAMEGPU->message_out.setVariable<float>("y_i", FLAMEGPU->getVariable<float>("y_i"));
  FLAMEGPU->message_out.setVariable<float>("z_i", FLAMEGPU->getVariable<float>("z_i"));
  FLAMEGPU->message_out.setVariable<float>("x_c", FLAMEGPU->getVariable<float>("x_c"));
  FLAMEGPU->message_out.setVariable<float>("y_c", FLAMEGPU->getVariable<float>("y_c"));
  FLAMEGPU->message_out.setVariable<float>("z_c", FLAMEGPU->getVariable<float>("z_c"));
  FLAMEGPU->message_out.setVariable<float>("rest_length_0", FLAMEGPU->getVariable<float>("rest_length_0"));
  FLAMEGPU->message_out.setVariable<float>("rest_length", FLAMEGPU->getVariable<float>("rest_length"));
  FLAMEGPU->message_out.setVariable<float>("k_fa", FLAMEGPU->getVariable<float>("k_fa"));
  FLAMEGPU->message_out.setVariable<float>("f_max", FLAMEGPU->getVariable<float>("f_max"));
  FLAMEGPU->message_out.setVariable<uint8_t>("attached", FLAMEGPU->getVariable<uint8_t>("attached"));
  FLAMEGPU->message_out.setVariable<uint8_t>("active", FLAMEGPU->getVariable<uint8_t>("active"));
  FLAMEGPU->message_out.setVariable<float>("v_c", FLAMEGPU->getVariable<float>("v_c"));
  FLAMEGPU->message_out.setVariable<uint8_t>("fa_state", FLAMEGPU->getVariable<uint8_t>("fa_state"));
  FLAMEGPU->message_out.setVariable<float>("age", FLAMEGPU->getVariable<float>("age"));
  FLAMEGPU->message_out.setVariable<float>("k_on", FLAMEGPU->getVariable<float>("k_on"));
  FLAMEGPU->message_out.setVariable<float>("k_off_0", FLAMEGPU->getVariable<float>("k_off_0"));
  FLAMEGPU->message_out.setVariable<float>("f_c", FLAMEGPU->getVariable<float>("f_c"));
  FLAMEGPU->message_out.setVariable<float>("k_reinf", FLAMEGPU->getVariable<float>("k_reinf"));
  
  FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<int>("cell_id"));
    

  return flamegpu::ALIVE;
}