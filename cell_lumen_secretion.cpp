/**
 * cls_normalize3
 *
 * Purpose:
 *   Normalize a 3D vector in-place; if near-zero, sets a default unit vector.
 */
FLAMEGPU_DEVICE_FUNCTION void cls_normalize3(float &x, float &y, float &z) {
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
 * cell_lumen_secretion
 *
 * Purpose:
 *   Each CELL agent stochastically secretes a LUMEN droplet in the direction
 *   opposite to its orientation vector (apical side = anti-orientation).
 *   The new LUMEN droplet is placed at one contact distance (cell_radius +
 *   LUMEN_RADIUS) behind the cell. A per-cell cooldown timer prevents
 *   continuous secretion.
 *
 * Inputs:
 *   - CELL variables: x, y, z, orx, ory, orz, radius, lumen_secretion_cooldown, dead
 *   - Environment: TIME_STEP, LUMEN_RADIUS, LUMEN_SECRETION_RATE,
 *                  LUMEN_SECRETION_COOLDOWN
 *   - MACRO_MAX_GLOBAL_LUMEN_ID for unique id assignment
 *
 * Outputs:
 *   - New LUMEN agent (if secretion fires), updated lumen_secretion_cooldown
 */
FLAMEGPU_AGENT_FUNCTION(cell_lumen_secretion, flamegpu::MessageNone, flamegpu::MessageNone) {
  // Skip dead cells
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    return flamegpu::ALIVE;
  }

  const float TIME_STEP                = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float LUMEN_RADIUS             = FLAMEGPU->environment.getProperty<float>("LUMEN_RADIUS");
  const float LUMEN_SECRETION_RATE     = FLAMEGPU->environment.getProperty<float>("LUMEN_SECRETION_RATE");
  const float LUMEN_SECRETION_COOLDOWN = FLAMEGPU->environment.getProperty<float>("LUMEN_SECRETION_COOLDOWN");

  float cooldown = FLAMEGPU->getVariable<float>("lumen_secretion_cooldown");

  // Advance cooldown timer
  cooldown -= TIME_STEP;

  if (cooldown <= 0.0f) {
    // Stochastic secretion
    const float prob = 1.0f - expf(-LUMEN_SECRETION_RATE * TIME_STEP);
    if (FLAMEGPU->random.uniform<float>() < prob) {
      const float agent_x = FLAMEGPU->getVariable<float>("x");
      const float agent_y = FLAMEGPU->getVariable<float>("y");
      const float agent_z = FLAMEGPU->getVariable<float>("z");
      const float agent_r = FLAMEGPU->getVariable<float>("radius");

      // Orientation vector (cell's front direction)
      float orx = FLAMEGPU->getVariable<float>("orx");
      float ory = FLAMEGPU->getVariable<float>("ory");
      float orz = FLAMEGPU->getVariable<float>("orz");
      cls_normalize3(orx, ory, orz);

      // Place LUMEN droplet opposite to orientation (apical/inner side)
      const float offset = agent_r + LUMEN_RADIUS;
      const float lx = agent_x - orx * offset;
      const float ly = agent_y - ory * offset;
      const float lz = agent_z - orz * offset;

      // Assign a new unique id
      auto MACRO_MAX_GLOBAL_LUMEN_ID = FLAMEGPU->environment.getMacroProperty<int, 1>("MACRO_MAX_GLOBAL_LUMEN_ID");
      const int new_lumen_id = MACRO_MAX_GLOBAL_LUMEN_ID.addAtomic(1);
      

      // Create new LUMEN agent
      auto lumen_out = FLAMEGPU->agent_out;
      lumen_out.setVariable<int>("id", new_lumen_id);
      lumen_out.setVariable<float>("x", lx);
      lumen_out.setVariable<float>("y", ly);
      lumen_out.setVariable<float>("z", lz);
      lumen_out.setVariable<float>("vx", 0.0f);
      lumen_out.setVariable<float>("vy", 0.0f);
      lumen_out.setVariable<float>("vz", 0.0f);
      lumen_out.setVariable<float>("radius", LUMEN_RADIUS);
      lumen_out.setVariable<float>("ll_dvx", 0.0f);
      lumen_out.setVariable<float>("ll_dvy", 0.0f);
      lumen_out.setVariable<float>("ll_dvz", 0.0f);
      lumen_out.setVariable<float>("lc_dvx", 0.0f);
      lumen_out.setVariable<float>("lc_dvy", 0.0f);
      lumen_out.setVariable<float>("lc_dvz", 0.0f);

      cooldown = LUMEN_SECRETION_COOLDOWN;
    }
  }

  FLAMEGPU->setVariable<float>("lumen_secretion_cooldown", cooldown);
  return flamegpu::ALIVE;
}
