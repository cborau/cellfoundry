FLAMEGPU_AGENT_FUNCTION(cell_update_stress, flamegpu::MessageBucket, flamegpu::MessageNone) {
  
  /*
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");

  //Define message variables (agent sending the input message)
  int message_id = 0;
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;
  float message_vx = 0.0;
  float message_vy = 0.0;
  float message_vz = 0.0;
  float message_fx = 0.0;
  float message_fy = 0.0;
  float message_fz = 0.0;
  float message_x_i = 0.0;
  float message_y_i = 0.0;
  float message_z_i = 0.0;
  float message_x_c = 0.0;
  float message_y_c = 0.0;
  float message_z_c = 0.0;
  int message_id = 0;
  int message_cell_id = 0;
  float message_rest_length_0 = 0.0;
  float message_rest_length = 0.0;
  float message_k_fa = 0.0;
  float message_f_max = 0.0;
  uint8_t message_active = 0;
  float message_v_c = 0.0;
  uint8_t message_state = 0;
  float message_age = 0.0;
  float message_k_on = 0.0;
  float message_k_off_0 = 0.0;
  float message_f_c = 0.0;
  float message_k_reinf = 0.0;
  int message_fnode_id = 0;
  uint8_t message_attached = 0;

  //Loop through all agents sending input messages
  for (const auto &message : FLAMEGPU->message_in( TODO: provide bucket index )) {
    message_id = message.getVariable<int>("id");
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_z = message.getVariable<float>("z");
    message_vx = message.getVariable<float>("vx");
    message_vy = message.getVariable<float>("vy");
    message_vz = message.getVariable<float>("vz");
    message_fx = message.getVariable<float>("fx");
    message_fy = message.getVariable<float>("fy");
    message_fz = message.getVariable<float>("fz");
    message_x_i = message.getVariable<float>("x_i");
    message_y_i = message.getVariable<float>("y_i");
    message_z_i = message.getVariable<float>("z_i");
    message_x_c = message.getVariable<float>("x_c");
    message_y_c = message.getVariable<float>("y_c");
    message_z_c = message.getVariable<float>("z_c");
    message_id = message.getVariable<int>("id");
    message_cell_id = message.getVariable<int>("cell_id");
    message_rest_length_0 = message.getVariable<float>("rest_length_0");
    message_rest_length = message.getVariable<float>("rest_length");
    message_k_fa = message.getVariable<float>("k_fa");
    message_f_max = message.getVariable<float>("f_max");
    message_active = message.getVariable<uint8_t>("active");
    message_v_c = message.getVariable<float>("v_c");
    message_state = message.getVariable<uint8_t>("state");
    message_age = message.getVariable<float>("age");
    message_k_on = message.getVariable<float>("k_on");
    message_k_off_0 = message.getVariable<float>("k_off_0");
    message_f_c = message.getVariable<float>("f_c");
    message_k_reinf = message.getVariable<float>("k_reinf");
    message_fnode_id = message.getVariable<int>("fnode_id");
    message_attached = message.getVariable<uint8_t>("attached");
  }

  //Set agent variables
  FLAMEGPU->setVariable<int>("id", agent_id);
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);
  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);
  */

  return flamegpu::ALIVE;
}