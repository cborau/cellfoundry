/**
 * multiscale_ecm_diffusion_output
 *
 * Purpose:
 *   Publish the ECM state needed by the next explicit diffusion substep.
 *
 * Inputs:
 *   - Agent grid_i, grid_j, grid_k and spatial position x, y, z
 *   - Agent concentration C_sp and local diffusion coefficient D_sp
 *
 * Outputs:
 *   - One MessageArray3D entry per ECM grid position
 *
 * Notes:
 *   Run before each diffusion update so all agents read the same previous
 *   concentration snapshot. Mechanical variables are published separately
 *   by ecm_grid_location_data on the main simulation clock.
 */
FLAMEGPU_AGENT_FUNCTION(multiscale_ecm_diffusion_output, flamegpu::MessageNone, flamegpu::MessageArray3D) {
  // Agent array variables
  const uint8_t N_SPECIES = 3; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  const uint8_t agent_grid_i = FLAMEGPU->getVariable<uint8_t>("grid_i");
  const uint8_t agent_grid_j = FLAMEGPU->getVariable<uint8_t>("grid_j");
  const uint8_t agent_grid_k = FLAMEGPU->getVariable<uint8_t>("grid_k");
  FLAMEGPU->message_out.setIndex(agent_grid_i, agent_grid_j, agent_grid_k);
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));

  for (int i = 0; i < N_SPECIES; i++) {
    const float agent_C_sp = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
    const float agent_D_sp = FLAMEGPU->getVariable<float, N_SPECIES>("D_sp", i);
    FLAMEGPU->message_out.setVariable<float, N_SPECIES>("C_sp", i, agent_C_sp);
    FLAMEGPU->message_out.setVariable<float, N_SPECIES>("D_sp", i, agent_D_sp);
  }
  return flamegpu::ALIVE;
}
