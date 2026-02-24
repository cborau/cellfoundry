/**
 * cell_spatial_location_data
 *
 * Purpose:
 *   Broadcast CELL kinematics and metabolic parameters over a spatial message list.
 *
 * Inputs:
 *   - CELL variables: id, x,y,z, vx,vy,vz
 *   - Species arrays: k_consumption, k_production, k_reaction, C_sp, M_sp
 *
 * Outputs:
 *   - MessageSpatial3D record for nearby agent queries
 */
FLAMEGPU_AGENT_FUNCTION(cell_spatial_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {

  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
  FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
  FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));
  // Agent array variables
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < N_SPECIES; i++) {
    float ncol = FLAMEGPU->getVariable<float, N_SPECIES>("k_consumption", i);
    FLAMEGPU->message_out.setVariable<float, N_SPECIES>("k_consumption", i, ncol);
  }
  
  for (int i = 0; i < N_SPECIES; i++) {
    float ncol = FLAMEGPU->getVariable<float, N_SPECIES>("k_production", i);
    FLAMEGPU->message_out.setVariable<float, N_SPECIES>("k_production", i, ncol);
  }

  for (int i = 0; i < N_SPECIES; i++) {
    float ncol = FLAMEGPU->getVariable<float, N_SPECIES>("k_reaction", i);
    FLAMEGPU->message_out.setVariable<float, N_SPECIES>("k_reaction", i, ncol);
  }

  for (int i = 0; i < N_SPECIES; i++) {
    float ncol = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
    FLAMEGPU->message_out.setVariable<float, N_SPECIES>("C_sp", i, ncol);
  }  

  for (int i = 0; i < N_SPECIES; i++) {
    float ncol = FLAMEGPU->getVariable<float, N_SPECIES>("M_sp", i);
    FLAMEGPU->message_out.setVariable<float, N_SPECIES>("M_sp", i, ncol);
  }

  FLAMEGPU->message_out.setVariable<int>("dead", FLAMEGPU->getVariable<int>("dead"));
  FLAMEGPU->message_out.setVariable<int>("dead_by", FLAMEGPU->getVariable<int>("dead_by"));


  return flamegpu::ALIVE;
}