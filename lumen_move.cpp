/**
 * lumen_move
 *
 * Purpose:
 *   Advance LUMEN agent position using overdamped dynamics. Combines all
 *   interaction velocity contributions (LUMEN-LUMEN and LUMEN-CELL) and
 *   integrates position with a forward-Euler step. Resets per-step velocity
 *   accumulators to zero after applying them.
 *
 * Inputs:
 *   - LUMEN kinematic state, ll_dv*, lc_dv* contributions
 *   - Environment: TIME_STEP, domain boundary coordinates
 *
 * Outputs:
 *   - Updated LUMEN position, velocity; reset interaction velocity accumulators
 */
FLAMEGPU_AGENT_FUNCTION(lumen_move, flamegpu::MessageNone, flamegpu::MessageNone) {
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");

  const float ll_dvx = FLAMEGPU->getVariable<float>("ll_dvx");
  const float ll_dvy = FLAMEGPU->getVariable<float>("ll_dvy");
  const float ll_dvz = FLAMEGPU->getVariable<float>("ll_dvz");
  const float lc_dvx = FLAMEGPU->getVariable<float>("lc_dvx");
  const float lc_dvy = FLAMEGPU->getVariable<float>("lc_dvy");
  const float lc_dvz = FLAMEGPU->getVariable<float>("lc_dvz");

  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");

  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 5);

  // Total velocity from interaction contributions
  float vx = ll_dvx + lc_dvx;
  float vy = ll_dvy + lc_dvy;
  float vz = ll_dvz + lc_dvz;

  // Forward-Euler position update
  agent_x += vx * TIME_STEP;
  agent_y += vy * TIME_STEP;
  agent_z += vz * TIME_STEP;

  // Clamp to domain boundaries
  agent_x = fminf(COORD_BOUNDARY_X_POS, fmaxf(COORD_BOUNDARY_X_NEG, agent_x));
  agent_y = fminf(COORD_BOUNDARY_Y_POS, fmaxf(COORD_BOUNDARY_Y_NEG, agent_y));
  agent_z = fminf(COORD_BOUNDARY_Z_POS, fmaxf(COORD_BOUNDARY_Z_NEG, agent_z));

  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);
  FLAMEGPU->setVariable<float>("vx", vx);
  FLAMEGPU->setVariable<float>("vy", vy);
  FLAMEGPU->setVariable<float>("vz", vz);

  // Reset per-step velocity accumulators
  FLAMEGPU->setVariable<float>("ll_dvx", 0.0f);
  FLAMEGPU->setVariable<float>("ll_dvy", 0.0f);
  FLAMEGPU->setVariable<float>("ll_dvz", 0.0f);
  FLAMEGPU->setVariable<float>("lc_dvx", 0.0f);
  FLAMEGPU->setVariable<float>("lc_dvy", 0.0f);
  FLAMEGPU->setVariable<float>("lc_dvz", 0.0f);

  return flamegpu::ALIVE;
}
