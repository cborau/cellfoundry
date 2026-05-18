FLAMEGPU_DEVICE_FUNCTION void cfr_normalize3(float &x, float &y, float &z) {
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
 * cell_fnode_repulsion
 *
 * Purpose:
 *   Prevent CELL centers from approaching FNODE points closer than an
 *   exclusion distance by adding a short-range repulsive velocity component.
 *
 * Inputs:
 *   - fnode_spatial_location_message (spatial neighbors)
 *   - Environment interaction parameters
 *
 * Outputs:
 *   - Per-cell FNODE interaction velocity contribution (cf_dv*) [um/s]
 */
FLAMEGPU_AGENT_FUNCTION(cell_fnode_repulsion, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    // Note: if DEAD_CELLS_DISAPPEAR = True, a dead CELL agent remains ALIVE for flamegpu purposes and may still interact with other agents.
    FLAMEGPU->setVariable<float>("cf_dvx", 0.0f);
    FLAMEGPU->setVariable<float>("cf_dvy", 0.0f);
    FLAMEGPU->setVariable<float>("cf_dvz", 0.0f);
    FLAMEGPU->setVariable<float>("cf_S_xx", 0.0f);
    FLAMEGPU->setVariable<float>("cf_S_yy", 0.0f);
    FLAMEGPU->setVariable<float>("cf_S_zz", 0.0f);
    FLAMEGPU->setVariable<float>("cf_S_xy", 0.0f);
    FLAMEGPU->setVariable<float>("cf_S_xz", 0.0f);
    FLAMEGPU->setVariable<float>("cf_S_yz", 0.0f);
    return flamegpu::ALIVE;
  }

  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  const float agent_radius = FLAMEGPU->getVariable<float>("radius");
  const float agent_nucleus_radius = FLAMEGPU->getVariable<float>("nucleus_radius");
  const int agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");

  const uint8_t N_CELL_TYPES = 3; // WARNING: must match main python model N_CELL_TYPES
  const float CELL_D_DUMPING = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_D_DUMPING", agent_cell_type);
  const float CELL_FNODE_REPULSION_K = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_FNODE_REPULSION_K", agent_cell_type);
  const float CELL_FNODE_EXCLUSION_DISTANCE = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_FNODE_EXCLUSION_DISTANCE", agent_cell_type);
  const float CELL_FNODE_DV_MAX = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_FNODE_DV_MAX", agent_cell_type);

  const float exclusion = fmaxf(1e-6f, fmaxf(CELL_FNODE_EXCLUSION_DISTANCE, agent_radius));

  float fx_sum = 0.0f;
  float fy_sum = 0.0f;
  float fz_sum = 0.0f;
  float cf_S_xx = 0.0f, cf_S_yy = 0.0f, cf_S_zz = 0.0f;
  float cf_S_xy = 0.0f, cf_S_xz = 0.0f, cf_S_yz = 0.0f;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const float mx = message.getVariable<float>("x");
    const float my = message.getVariable<float>("y");
    const float mz = message.getVariable<float>("z");

    float dx = agent_x - mx;
    float dy = agent_y - my;
    float dz = agent_z - mz;
    float dist2 = dx * dx + dy * dy + dz * dz;

    if (dist2 <= 1e-20f) {
      dx = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      dy = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      dz = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      cfr_normalize3(dx, dy, dz);
      dist2 = 1e-12f;
    }

    const float dist = sqrtf(dist2);
    if (dist >= exclusion) {
      continue;
    }

    float nx = dx;
    float ny = dy;
    float nz = dz;
    cfr_normalize3(nx, ny, nz);

    const float overlap = exclusion - dist;
    const float f_pair = fmaxf(0.0f, CELL_FNODE_REPULSION_K) * overlap;
    fx_sum += f_pair * nx;
    fy_sum += f_pair * ny;
    fz_sum += f_pair * nz;
    const float cf_s_coeff = -agent_nucleus_radius * f_pair;
    cf_S_xx += cf_s_coeff * nx * nx;
    cf_S_yy += cf_s_coeff * ny * ny;
    cf_S_zz += cf_s_coeff * nz * nz;
    cf_S_xy += cf_s_coeff * nx * ny;
    cf_S_xz += cf_s_coeff * nx * nz;
    cf_S_yz += cf_s_coeff * ny * nz;
  }

  const float inv_drag = (CELL_D_DUMPING > 1e-12f) ? (1.0f / CELL_D_DUMPING) : 0.0f;
  float dvx = fx_sum * inv_drag;
  float dvy = fy_sum * inv_drag;
  float dvz = fz_sum * inv_drag;

  const float dv_max = fmaxf(0.0f, CELL_FNODE_DV_MAX);
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

  FLAMEGPU->setVariable<float>("cf_dvx", dvx);
  FLAMEGPU->setVariable<float>("cf_dvy", dvy);
  FLAMEGPU->setVariable<float>("cf_dvz", dvz);
  FLAMEGPU->setVariable<float>("cf_S_xx", cf_S_xx);
  FLAMEGPU->setVariable<float>("cf_S_yy", cf_S_yy);
  FLAMEGPU->setVariable<float>("cf_S_zz", cf_S_zz);
  FLAMEGPU->setVariable<float>("cf_S_xy", cf_S_xy);
  FLAMEGPU->setVariable<float>("cf_S_xz", cf_S_xz);
  FLAMEGPU->setVariable<float>("cf_S_yz", cf_S_yz);
  return flamegpu::ALIVE;
}
