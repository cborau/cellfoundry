/**
 * fnode_bucket_location_data
 *
 * Purpose:
 *   Export FNODE state and connectivity arrays into a bucket message keyed by node id.
 *
 * Inputs:
 *   - FNODE kinematics and material parameters
 *   - Connectivity arrays: linked_nodes, equilibrium_distance
 *
 * Outputs:
 *   - MessageBucket record for direct id-based neighbor access
 */
FLAMEGPU_AGENT_FUNCTION(fnode_bucket_location_data, flamegpu::MessageNone, flamegpu::MessageBucket) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
  FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
  FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));
  FLAMEGPU->message_out.setVariable<float>("k_elast", FLAMEGPU->getVariable<float>("k_elast"));
  FLAMEGPU->message_out.setVariable<float>("d_dumping", FLAMEGPU->getVariable<float>("d_dumping"));
  const uint8_t MAX_CONNECTIVITY = 8; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function. TODO: declare it somehow at compile time
  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
      int idx = FLAMEGPU->getVariable<int, MAX_CONNECTIVITY>("linked_nodes", i);
      FLAMEGPU->message_out.setVariable<int, MAX_CONNECTIVITY>("linked_nodes", i, idx);
  }
  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
      float ncol = FLAMEGPU->getVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i);
      FLAMEGPU->message_out.setVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i, ncol);
  }
  FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<int>("id"));

  return flamegpu::ALIVE;
}