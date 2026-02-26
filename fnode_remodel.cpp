/**
 * fnode_remodel
 *
 * Purpose:
 *   Update FNODE degradation/deposition state from nearby CELLs, scale
 *   stiffness, and register removal requests when degradation reaches 1.
 *
 * Inputs:
 *   - CELL spatial messages (x, y, z, dead)
 *   - FNODE state (degradation, id)
 *   - Remodeling environment properties and removal macro buffers
 *
 * Outputs:
 *   - Updated FNODE `degradation`, `k_elast`, `marked_for_removal`
 */
FLAMEGPU_AGENT_FUNCTION(fnode_remodel, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  const uint32_t INCLUDE_NETWORK_REMODELING = FLAMEGPU->environment.getProperty<uint32_t>("INCLUDE_NETWORK_REMODELING");
  if (INCLUDE_NETWORK_REMODELING == 0) {
    return flamegpu::ALIVE;
  }

  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float FNODE_DEGRADATION_RATE = FLAMEGPU->environment.getProperty<float>("FNODE_DEGRADATION_RATE");
  const float FNODE_DEPOSITION_RATE = FLAMEGPU->environment.getProperty<float>("FNODE_DEPOSITION_RATE");
  const float FNODE_CELL_DEGRADATION_RADIUS = FLAMEGPU->environment.getProperty<float>("FNODE_CELL_DEGRADATION_RADIUS");
  const float FIBRE_SEGMENT_K_ELAST = FLAMEGPU->environment.getProperty<float>("FIBRE_SEGMENT_K_ELAST");

  const int id = FLAMEGPU->getVariable<int>("id");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  float degradation = FLAMEGPU->getVariable<float>("degradation");

  int n_live_cells = 0;
  const float r2max = FNODE_CELL_DEGRADATION_RADIUS * FNODE_CELL_DEGRADATION_RADIUS;
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    if (message.getVariable<int>("dead") == 1) {
      continue;
    }
    const float dx = message.getVariable<float>("x") - agent_x;
    const float dy = message.getVariable<float>("y") - agent_y;
    const float dz = message.getVariable<float>("z") - agent_z;
    const float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= r2max) {
      n_live_cells += 1;
    }
  }

  const float d_inc = TIME_STEP * fmaxf(0.0f, FNODE_DEGRADATION_RATE) * static_cast<float>(n_live_cells);
  const float d_dec = TIME_STEP * fmaxf(0.0f, FNODE_DEPOSITION_RATE);
  degradation += d_inc;
  degradation -= d_dec;
  degradation = fminf(1.0f, fmaxf(0.0f, degradation));

  FLAMEGPU->setVariable<float>("degradation", degradation);
  FLAMEGPU->setVariable<float>("k_elast", fmaxf(0.0f, FIBRE_SEGMENT_K_ELAST * (1.0f - degradation)));

  if (degradation >= 1.0f) {
    FLAMEGPU->setVariable<int>("marked_for_removal", 1);
    printf("FNODE %d at (%f, %f, %f) is marked for removal due to degradation\n", id, agent_x, agent_y, agent_z);
  } else {
    FLAMEGPU->setVariable<int>("marked_for_removal", 0);
  }

  return flamegpu::ALIVE;
}
