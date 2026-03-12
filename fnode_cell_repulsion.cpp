FLAMEGPU_DEVICE_FUNCTION void fcr_normalize3(float &x, float &y, float &z) {
  const float n2 = x * x + y * y + z * z;
  if (n2 > 1e-20f) {
    const float inv = rsqrtf(n2);
    x *= inv;
    y *= inv;
    z *= inv;
  } else {
    x = 1.0f;
    y = 0.0f;
    z = 0.0f;
  }
}

/**
 * fnode_cell_repulsion
 *
 * Purpose:
 *   Prevent FNODE points from being pushed into CELL centers by adding a
 *   short-range repulsive force (Newton-3 counterpart of cell_fnode_repulsion).
 *
 * Inputs:
 *   - cell_spatial_location_message (spatial neighbors)
 *   - Environment interaction parameters (same as cell_fnode_repulsion)
 *
 * Outputs:
 *   - Updated FNODE force components (fx, fy, fz) with repulsive contributions added
 */
FLAMEGPU_AGENT_FUNCTION(fnode_cell_repulsion, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  // Read existing accumulated forces
  float agent_fx = FLAMEGPU->getVariable<float>("fx");
  float agent_fy = FLAMEGPU->getVariable<float>("fy");
  float agent_fz = FLAMEGPU->getVariable<float>("fz");

  const uint8_t N_CELL_TYPES = 3; // WARNING: must match main python model N_CELL_TYPES

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const int msg_dead = message.getVariable<int>("dead");
    if (msg_dead == 1) continue;

    const float mx = message.getVariable<float>("x");
    const float my = message.getVariable<float>("y");
    const float mz = message.getVariable<float>("z");
    const int msg_cell_type = message.getVariable<int>("cell_type");

    // Use the same exclusion parameters as cell_fnode_repulsion for this cell type
    const float CELL_FNODE_REPULSION_K = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_FNODE_REPULSION_K", msg_cell_type);
    const float CELL_FNODE_EXCLUSION_DISTANCE = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_FNODE_EXCLUSION_DISTANCE", msg_cell_type);
    const float msg_radius = message.getVariable<float>("radius");

    const float exclusion = fmaxf(1e-6f, fmaxf(CELL_FNODE_EXCLUSION_DISTANCE, msg_radius));

    // Direction: fnode - cell  (points away from cell center)
    float dx = agent_x - mx;
    float dy = agent_y - my;
    float dz = agent_z - mz;
    float dist2 = dx * dx + dy * dy + dz * dz;

    if (dist2 <= 1e-20f) {
      dx = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      dy = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      dz = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      fcr_normalize3(dx, dy, dz);
      dist2 = 1e-12f;
    }

    const float dist = sqrtf(dist2);
    if (dist >= exclusion) continue;

    float nx = dx;
    float ny = dy;
    float nz = dz;
    fcr_normalize3(nx, ny, nz);

    const float overlap = exclusion - dist;
    const float f_pair = fmaxf(0.0f, CELL_FNODE_REPULSION_K) * overlap;

    agent_fx += f_pair * nx;
    agent_fy += f_pair * ny;
    agent_fz += f_pair * nz;
  }

  FLAMEGPU->setVariable<float>("fx", agent_fx);
  FLAMEGPU->setVariable<float>("fy", agent_fy);
  FLAMEGPU->setVariable<float>("fz", agent_fz);

  return flamegpu::ALIVE;
}
