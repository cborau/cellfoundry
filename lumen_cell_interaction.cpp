/**
 * lc_normalize3
 *
 * Purpose:
 *   Normalize a 3D vector in-place; if near-zero, sets a default unit vector.
 */
FLAMEGPU_DEVICE_FUNCTION void lc_normalize3(float &x, float &y, float &z) {
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
 * lumen_cell_interaction
 *
 * Purpose:
 *   Compute LUMEN-CELL repulsion (hydrostatic pressure effect). When a LUMEN
 *   droplet overlaps with a cell, it is pushed away. This is the LUMEN-side
 *   reaction; the corresponding CELL-side reaction is handled in
 *   cell_lumen_interaction.cpp.
 *
 * Inputs:
 *   - cell_spatial_location_message (spatial neighbors)
 *   - Environment parameters: LUMEN_K_LUMEN_CELL_REPULSION, LUMEN_ETA,
 *                             LUMEN_LUMEN_CELL_DV_MAX
 *
 * Outputs:
 *   - Per-LUMEN interaction velocity contribution (lc_dv*) [um/s]
 */
FLAMEGPU_AGENT_FUNCTION(lumen_cell_interaction, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  const float agent_r = FLAMEGPU->getVariable<float>("radius");

  const float LUMEN_ETA                   = FLAMEGPU->environment.getProperty<float>("LUMEN_ETA");
  const float LUMEN_K_LUMEN_CELL_REPULSION = FLAMEGPU->environment.getProperty<float>("LUMEN_K_LUMEN_CELL_REPULSION");
  const float LUMEN_LUMEN_CELL_DV_MAX      = FLAMEGPU->environment.getProperty<float>("LUMEN_LUMEN_CELL_DV_MAX");

  float fx_sum = 0.0f;
  float fy_sum = 0.0f;
  float fz_sum = 0.0f;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    if (message.getVariable<int>("dead") == 1) {
      continue;
    }

    const float mx = message.getVariable<float>("x");
    const float my = message.getVariable<float>("y");
    const float mz = message.getVariable<float>("z");
    const float mr = message.getVariable<float>("radius");

    float dx = agent_x - mx;
    float dy = agent_y - my;
    float dz = agent_z - mz;
    float dist2 = dx * dx + dy * dy + dz * dz;

    if (dist2 <= 1e-20f) {
      dx = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      dy = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      dz = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      lc_normalize3(dx, dy, dz);
      dist2 = 1e-12f;
    }

    const float dist = sqrtf(dist2);
    const float r_contact = fmaxf(1e-6f, agent_r + mr);

    if (dist >= r_contact) {
      continue;
    }

    float nx = dx;
    float ny = dy;
    float nz = dz;
    lc_normalize3(nx, ny, nz);

    const float overlap = r_contact - dist;
    const float f_pair = fmaxf(0.0f, LUMEN_K_LUMEN_CELL_REPULSION) * overlap;

    fx_sum += f_pair * nx;
    fy_sum += f_pair * ny;
    fz_sum += f_pair * nz;
  }

  const float inv_drag = (LUMEN_ETA > 1e-12f) ? (1.0f / LUMEN_ETA) : 0.0f;
  float dvx = fx_sum * inv_drag;
  float dvy = fy_sum * inv_drag;
  float dvz = fz_sum * inv_drag;

  // Cap velocity contribution
  const float dv_max = fmaxf(0.0f, LUMEN_LUMEN_CELL_DV_MAX);
  if (dv_max > 0.0f) {
    const float dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
    const float dvn = sqrtf(dv2 + 1e-20f);
    if (dvn > dv_max) {
      const float scale = dv_max / dvn;
      dvx *= scale;
      dvy *= scale;
      dvz *= scale;
    }
  }

  FLAMEGPU->setVariable<float>("lc_dvx", dvx);
  FLAMEGPU->setVariable<float>("lc_dvy", dvy);
  FLAMEGPU->setVariable<float>("lc_dvz", dvz);

  return flamegpu::ALIVE;
}
