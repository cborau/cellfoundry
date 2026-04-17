/**
 * fnode_remodel
 *
 * Purpose:
 *   Update FNODE degradation/reinforcement state from nearby CELLs and register removal requests when net degradation reaches 1.
 *
 * Inputs:
 *   - CELL spatial messages (x, y, z, dead)
 *   - FNODE state (degradation, reinforcement, id)
 *   - Remodeling environment properties and removal macro buffers
 *
 * Outputs:
 *   - Updated FNODE `degradation`, `reinforcement`, `marked_for_removal`
 */
FLAMEGPU_AGENT_FUNCTION(fnode_remodel, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  const uint32_t INCLUDE_NETWORK_REMODELING = FLAMEGPU->environment.getProperty<uint32_t>("INCLUDE_NETWORK_REMODELING");
  if (INCLUDE_NETWORK_REMODELING == 0) {
    return flamegpu::ALIVE;
  }
  int DEBUG_PRINTING = FLAMEGPU->environment.getProperty<int>("DEBUG_PRINTING");
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const uint8_t N_CELL_TYPES = 3; // WARNING: must match the value in the main python script.
  // Per-cell-type degradation / deposition rates and cell radii
  float FNODE_DEGRADATION_RATE[N_CELL_TYPES];
  float FNODE_DEPOSITION_RATE[N_CELL_TYPES];
  float CELL_RADIUS[N_CELL_TYPES];
  for (int ct = 0; ct < N_CELL_TYPES; ct++) {
    FNODE_DEGRADATION_RATE[ct] = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FNODE_DEGRADATION_RATE", ct);
    FNODE_DEPOSITION_RATE[ct]  = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FNODE_DEPOSITION_RATE", ct);
    CELL_RADIUS[ct]            = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_RADIUS", ct);
  }
  const float FNODE_CELL_DEGRADATION_RADIUS = FLAMEGPU->environment.getProperty<float>("FNODE_CELL_DEGRADATION_RADIUS");

  const int id = FLAMEGPU->getVariable<int>("id");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  float degradation = FLAMEGPU->getVariable<float>("degradation");
  float reinforcement = FLAMEGPU->getVariable<float>("reinforcement");

  // Accumulate per-cell-type degradation and deposition contributions
  float deg_sum = 0.0f;
  float dep_sum = 0.0f;
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
      const int ct = message.getVariable<int>("cell_type");
      // Distance-dependent weight: full effect at distance <= CELL_RADIUS, zero at FNODE_CELL_DEGRADATION_RADIUS
      const float dist = sqrtf(r2);
      const float cr = CELL_RADIUS[ct];
      float weight = 1.0f;
      if (dist > cr && FNODE_CELL_DEGRADATION_RADIUS > cr) {
        weight = (FNODE_CELL_DEGRADATION_RADIUS - dist) / (FNODE_CELL_DEGRADATION_RADIUS - cr);
      }
      weight = fminf(1.0f, fmaxf(0.0f, weight));
      deg_sum += weight * fmaxf(0.0f, FNODE_DEGRADATION_RATE[ct]);
      dep_sum += weight * fmaxf(0.0f, FNODE_DEPOSITION_RATE[ct]);
    }
  }

  // Degradation 
  const float d_inc = TIME_STEP * deg_sum;
  degradation += d_inc;
  degradation = fminf(1.0f, fmaxf(0.0f, degradation));

  // Reinforcement 
  const float r_inc = TIME_STEP * dep_sum;
  reinforcement += r_inc;
  reinforcement = fmaxf(0.0f, reinforcement); // No upper bound on reinforcement

  FLAMEGPU->setVariable<float>("degradation", degradation);
  FLAMEGPU->setVariable<float>("reinforcement", reinforcement);

  // Net degradation: if degradation minus reinforcement reaches 1, mark for removal
  const float net_degradation = degradation - reinforcement;
  if (net_degradation >= 1.0f) {
    FLAMEGPU->setVariable<int>("marked_for_removal", 1);
    if (DEBUG_PRINTING == 1) {
      printf("[fnode_remodel] FNODE %d at (%.3f, %.3f, %.3f) marked for removal (deg=%.3f, reinf=%.3f)\n", id, agent_x, agent_y, agent_z, degradation, reinforcement);
    }
  } else {
    FLAMEGPU->setVariable<int>("marked_for_removal", 0);
  }

  return flamegpu::ALIVE;
}
