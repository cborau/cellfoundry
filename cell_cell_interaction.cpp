FLAMEGPU_DEVICE_FUNCTION float cc_clampf(const float x, const float lo, const float hi) {
  return fminf(hi, fmaxf(lo, x));
}

FLAMEGPU_DEVICE_FUNCTION void cc_normalize3(float &x, float &y, float &z) {
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
 * cell_cell_interaction
 *
 * Purpose:
 *   Compute short-range CELL-CELL mechanics with strong contact repulsion and
 *   weak finite-range adhesion shell (soft cohesion) to promote aggregate
 *   compactness while allowing escape under other motility cues.
 *
 * Inputs:
 *   - cell_spatial_location_message (spatial neighbors)
 *   - Environment interaction parameters
 *
 * Outputs:
 *   - Per-cell interaction velocity contribution (cc_dv*) [um/s]
 */
FLAMEGPU_AGENT_FUNCTION(cell_cell_interaction, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    // Note: if DEAD_CELLS_DISAPPEAR = True, a dead CELL agent remains ALIVE for flamegpu purposes and may still interact with other agents.
    FLAMEGPU->setVariable<float>("cc_dvx", 0.0f);
    FLAMEGPU->setVariable<float>("cc_dvy", 0.0f);
    FLAMEGPU->setVariable<float>("cc_dvz", 0.0f);
    return flamegpu::ALIVE;
  }

  const int agent_id = FLAMEGPU->getVariable<int>("id");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  const float agent_r = FLAMEGPU->getVariable<float>("radius");

  const float CELL_D_DUMPING = FLAMEGPU->environment.getProperty<float>("CELL_D_DUMPING");
  const float CELL_CELL_REPULSION_K = FLAMEGPU->environment.getProperty<float>("CELL_CELL_REPULSION_K");
  const float CELL_CELL_ADHESION_K = FLAMEGPU->environment.getProperty<float>("CELL_CELL_ADHESION_K");
  const float CELL_CELL_ADHESION_RANGE = FLAMEGPU->environment.getProperty<float>("CELL_CELL_ADHESION_RANGE");
  const float CELL_CELL_DV_MAX = FLAMEGPU->environment.getProperty<float>("CELL_CELL_DV_MAX");

  float fx_sum = 0.0f;
  float fy_sum = 0.0f;
  float fz_sum = 0.0f;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const int message_id = message.getVariable<int>("id");
    if (message_id == agent_id) {
      continue;
    }

    if (message.getVariable<int>("dead") == 1) {
      continue; //USER-DEFINED behavior for dead cells
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
      cc_normalize3(dx, dy, dz);
      dist2 = 1e-12f;
    }

    const float dist = sqrtf(dist2);
    const float r_contact = fmaxf(1e-6f, agent_r + mr);
    const float r_adh_end = r_contact + fmaxf(1e-6f, CELL_CELL_ADHESION_RANGE);

    float nx = dx;
    float ny = dy;
    float nz = dz;
    cc_normalize3(nx, ny, nz);

    float f_pair = 0.0f;
    if (dist < r_contact) {
      const float overlap = r_contact - dist;
      f_pair = fmaxf(0.0f, CELL_CELL_REPULSION_K) * overlap;
    } else if (dist < r_adh_end) {
      const float s = (dist - r_contact) / fmaxf(1e-6f, CELL_CELL_ADHESION_RANGE);
      const float bell = 4.0f * s * (1.0f - s);
      f_pair = -fmaxf(0.0f, CELL_CELL_ADHESION_K) * bell;
    }

    fx_sum += f_pair * nx;
    fy_sum += f_pair * ny;
    fz_sum += f_pair * nz;
  }

  const float inv_drag = (CELL_D_DUMPING > 1e-12f) ? (1.0f / CELL_D_DUMPING) : 0.0f;
  float dvx = fx_sum * inv_drag;
  float dvy = fy_sum * inv_drag;
  float dvz = fz_sum * inv_drag;

  const float dv_max = fmaxf(0.0f, CELL_CELL_DV_MAX);
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

  FLAMEGPU->setVariable<float>("cc_dvx", dvx);
  FLAMEGPU->setVariable<float>("cc_dvy", dvy);
  FLAMEGPU->setVariable<float>("cc_dvz", dvz);
  return flamegpu::ALIVE;
}
