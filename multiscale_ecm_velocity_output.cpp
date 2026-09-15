/**
 * multiscale_ecm_velocity_output
 *
 * Purpose:
 *   Publish updated ECM velocities for vascular advection after ECM movement.
 *
 * Inputs:
 *   - Agent grid_i, grid_j, grid_k and velocity vx, vy, vz
 *
 * Outputs:
 *   - MessageArray3D entry containing vx, vy and vz for vasc_move
 *
 * Notes:
 *   Runs once per main timestep, outside diffusion submodels. The separate
 *   message list avoids writing twice to the original ECM array message in
 *   one step, which raises ArrayMessageWriteConflict in the installed rc.5.
 */
FLAMEGPU_AGENT_FUNCTION(multiscale_ecm_velocity_output, flamegpu::MessageNone, flamegpu::MessageArray3D) {
  const uint8_t agent_grid_i = FLAMEGPU->getVariable<uint8_t>("grid_i");
  const uint8_t agent_grid_j = FLAMEGPU->getVariable<uint8_t>("grid_j");
  const uint8_t agent_grid_k = FLAMEGPU->getVariable<uint8_t>("grid_k");
  FLAMEGPU->message_out.setIndex(agent_grid_i, agent_grid_j, agent_grid_k);
  FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
  FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
  FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));
  return flamegpu::ALIVE;
}
