FLAMEGPU_AGENT_FUNCTION(focad_anchor_update, flamegpu::MessageBucket, flamegpu::MessageNone) {
  
  /*//Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");
  float agent_fx = FLAMEGPU->getVariable<float>("fx");
  float agent_fy = FLAMEGPU->getVariable<float>("fy");
  float agent_fz = FLAMEGPU->getVariable<float>("fz");
  float agent_x_i = FLAMEGPU->getVariable<float>("x_i");
  float agent_y_i = FLAMEGPU->getVariable<float>("y_i");
  float agent_z_i = FLAMEGPU->getVariable<float>("z_i");
  float agent_x_c = FLAMEGPU->getVariable<float>("x_c");
  float agent_y_c = FLAMEGPU->getVariable<float>("y_c");
  float agent_z_c = FLAMEGPU->getVariable<float>("z_c");
  int agent_id = FLAMEGPU->getVariable<int>("id");
  int agent_cell_id = FLAMEGPU->getVariable<int>("cell_id");
  float agent_rest_length_0 = FLAMEGPU->getVariable<float>("rest_length_0");
  float agent_rest_length = FLAMEGPU->getVariable<float>("rest_length");
  float agent_k_fa = FLAMEGPU->getVariable<float>("k_fa");
  float agent_f_max = FLAMEGPU->getVariable<float>("f_max");
  uint8_t agent_active = FLAMEGPU->getVariable<uint8_t>("active");
  float agent_v_c = FLAMEGPU->getVariable<float>("v_c");
  uint8_t agent_fa_state = FLAMEGPU->getVariable<uint8_t>("fa_state");
  float agent_age = FLAMEGPU->getVariable<float>("age");
  float agent_k_on = FLAMEGPU->getVariable<float>("k_on");
  float agent_k_off_0 = FLAMEGPU->getVariable<float>("k_off_0");
  float agent_f_c = FLAMEGPU->getVariable<float>("f_c");
  float agent_k_reinf = FLAMEGPU->getVariable<float>("k_reinf");
  int agent_fnode_id = FLAMEGPU->getVariable<int>("fnode_id");
  uint8_t agent_attached = FLAMEGPU->getVariable<uint8_t>("attached");

  //Define message variables (agent sending the input message)
  int message_id = 0;
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;
  float message_vx = 0.0;
  float message_vy = 0.0;
  float message_vz = 0.0;

  //Loop through all agents sending input messages
  for (const auto &message : FLAMEGPU->message_in(TODO: provide bucket index )) {
    message_id = message.getVariable<int>("id");
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_z = message.getVariable<float>("z");
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
  FLAMEGPU->setVariable<float>("fx", agent_fx);
  FLAMEGPU->setVariable<float>("fy", agent_fy);
  FLAMEGPU->setVariable<float>("fz", agent_fz);
  FLAMEGPU->setVariable<float>("x_i", agent_x_i);
  FLAMEGPU->setVariable<float>("y_i", agent_y_i);
  FLAMEGPU->setVariable<float>("z_i", agent_z_i);
  FLAMEGPU->setVariable<float>("x_c", agent_x_c);
  FLAMEGPU->setVariable<float>("y_c", agent_y_c);
  FLAMEGPU->setVariable<float>("z_c", agent_z_c);
  FLAMEGPU->setVariable<int>("id", agent_id);
  FLAMEGPU->setVariable<int>("cell_id", agent_cell_id);
  FLAMEGPU->setVariable<float>("rest_length_0", agent_rest_length_0);
  FLAMEGPU->setVariable<float>("rest_length", agent_rest_length);
  FLAMEGPU->setVariable<float>("k_fa", agent_k_fa);
  FLAMEGPU->setVariable<float>("f_max", agent_f_max);
  FLAMEGPU->setVariable<uint8_t>("active", agent_active);
  FLAMEGPU->setVariable<float>("v_c", agent_v_c);
  FLAMEGPU->setVariable<uint8_t>("fa_state", agent_state);
  FLAMEGPU->setVariable<float>("age", agent_age);
  FLAMEGPU->setVariable<float>("k_on", agent_k_on);
  FLAMEGPU->setVariable<float>("k_off_0", agent_k_off_0);
  FLAMEGPU->setVariable<float>("f_c", agent_f_c);
  FLAMEGPU->setVariable<float>("k_reinf", agent_k_reinf);
  FLAMEGPU->setVariable<int>("fnode_id", agent_fnode_id);
  FLAMEGPU->setVariable<uint8_t>("attached", agent_attached);
 */

  return flamegpu::ALIVE;
}