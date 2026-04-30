/**
 * ll_normalize3
 *
 * Purpose:
 *   Normalize a 3D vector in-place; if near-zero, sets a default unit vector.
 */
FLAMEGPU_DEVICE_FUNCTION void ll_normalize3(float &x, float &y, float &z) {
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
 * lumen_lumen_interaction
 *
 * Purpose:
 *   Compute short-range LUMEN-LUMEN mechanics: volume exclusion repulsion at
 *   contact and a cohesive adhesion shell beyond contact (surface-tension analogue).
 *   This keeps LUMEN droplets together as a coherent liquid mass while preventing
 *   overlap.
 *
 * Inputs:
 *   - lumen_spatial_location_message (spatial neighbors)
 *   - Environment parameters: LUMEN_K_LUMEN_LUMEN_REPULSION, LUMEN_K_LUMEN_LUMEN_ADHESION,
 *                             LUMEN_LUMEN_ADHESION_RANGE, LUMEN_ETA
 *
 * Outputs:
 *   - Per-LUMEN interaction velocity contribution (ll_dv*) [um/s]
 */
FLAMEGPU_AGENT_FUNCTION(lumen_lumen_interaction, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  const int agent_id = FLAMEGPU->getVariable<int>("id");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  const float agent_r = FLAMEGPU->getVariable<float>("radius");

  const float LUMEN_ETA                     = FLAMEGPU->environment.getProperty<float>("LUMEN_ETA");
  const float LUMEN_K_LUMEN_LUMEN_REPULSION = FLAMEGPU->environment.getProperty<float>("LUMEN_K_LUMEN_LUMEN_REPULSION");
  const float LUMEN_K_LUMEN_LUMEN_ADHESION  = FLAMEGPU->environment.getProperty<float>("LUMEN_K_LUMEN_LUMEN_ADHESION");
  const float LUMEN_LUMEN_ADHESION_RANGE    = FLAMEGPU->environment.getProperty<float>("LUMEN_LUMEN_ADHESION_RANGE");

  float fx_sum = 0.0f;
  float fy_sum = 0.0f;
  float fz_sum = 0.0f;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const int message_id = message.getVariable<int>("id");
    if (message_id == agent_id) {
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
      ll_normalize3(dx, dy, dz);
      dist2 = 1e-12f;
    }

    const float dist = sqrtf(dist2);
    const float r_contact = fmaxf(1e-6f, agent_r + mr);
    const float r_adh_end = r_contact + fmaxf(1e-6f, LUMEN_LUMEN_ADHESION_RANGE);

    float nx = dx;
    float ny = dy;
    float nz = dz;
    ll_normalize3(nx, ny, nz);

    float f_pair = 0.0f;
    if (dist < r_contact) {
      // Volume exclusion repulsion
      const float overlap = r_contact - dist;
      f_pair = fmaxf(0.0f, LUMEN_K_LUMEN_LUMEN_REPULSION) * overlap;
    } else if (dist < r_adh_end) {
      // Cohesive adhesion shell (bell-shaped, attractive)
      const float s = (dist - r_contact) / fmaxf(1e-6f, LUMEN_LUMEN_ADHESION_RANGE);
      const float bell = 4.0f * s * (1.0f - s);
      f_pair = -fmaxf(0.0f, LUMEN_K_LUMEN_LUMEN_ADHESION) * bell;
    }

    fx_sum += f_pair * nx;
    fy_sum += f_pair * ny;
    fz_sum += f_pair * nz;
  }

  const float inv_drag = (LUMEN_ETA > 1e-12f) ? (1.0f / LUMEN_ETA) : 0.0f;
  float dvx = fx_sum * inv_drag;
  float dvy = fy_sum * inv_drag;
  float dvz = fz_sum * inv_drag;

  // Cap velocity contribution to prevent numerical blow-up when LUMEN strongly overlap
  const float LUMEN_LUMEN_CELL_DV_MAX = FLAMEGPU->environment.getProperty<float>("LUMEN_LUMEN_CELL_DV_MAX");
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

  FLAMEGPU->setVariable<float>("ll_dvx", dvx);
  FLAMEGPU->setVariable<float>("ll_dvy", dvy);
  FLAMEGPU->setVariable<float>("ll_dvz", dvz);

  return flamegpu::ALIVE;
}
