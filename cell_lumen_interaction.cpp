/**
 * cl_normalize3
 *
 * Purpose:
 *   Normalize a 3D vector in-place; if near-zero, sets a default unit vector.
 */
FLAMEGPU_DEVICE_FUNCTION void cl_normalize3(float &x, float &y, float &z) {
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
 * cell_lumen_interaction
 *
 * Purpose:
 *   Compute CELL-LUMEN repulsion (Newton's 3rd law pair of lumen_cell_interaction).
 *   When a LUMEN droplet overlaps with a cell, the cell is pushed away from the
 *   lumen (simulating hydrostatic pressure from the lumen cavity pushing the
 *   surrounding cells outward).
 *
 * Inputs:
 *   - lumen_spatial_location_message (spatial neighbors)
 *   - Environment parameters: LUMEN_K_LUMEN_CELL_REPULSION, CELL_D_DUMPING,
 *                             LUMEN_LUMEN_CELL_DV_MAX
 *
 * Outputs:
 *   - Per-CELL interaction velocity contribution (cl_dv*) [um/s]
 */
FLAMEGPU_AGENT_FUNCTION(cell_lumen_interaction, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    FLAMEGPU->setVariable<float>("cl_dvx", 0.0f);
    FLAMEGPU->setVariable<float>("cl_dvy", 0.0f);
    FLAMEGPU->setVariable<float>("cl_dvz", 0.0f);
    FLAMEGPU->setVariable<float>("cl_S_xx", 0.0f);
    FLAMEGPU->setVariable<float>("cl_S_yy", 0.0f);
    FLAMEGPU->setVariable<float>("cl_S_zz", 0.0f);
    FLAMEGPU->setVariable<float>("cl_S_xy", 0.0f);
    FLAMEGPU->setVariable<float>("cl_S_xz", 0.0f);
    FLAMEGPU->setVariable<float>("cl_S_yz", 0.0f);
    return flamegpu::ALIVE;
  }

  const int agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  const float agent_r = FLAMEGPU->getVariable<float>("radius");
  const float agent_nucleus_radius = FLAMEGPU->getVariable<float>("nucleus_radius");

  const uint8_t N_CELL_TYPES = 3; // WARNING: must match main python model N_CELL_TYPES
  const float CELL_D_DUMPING            = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_D_DUMPING", agent_cell_type);
  const float LUMEN_K_LUMEN_CELL_REPULSION = FLAMEGPU->environment.getProperty<float>("LUMEN_K_LUMEN_CELL_REPULSION");
  const float LUMEN_LUMEN_CELL_DV_MAX      = FLAMEGPU->environment.getProperty<float>("LUMEN_LUMEN_CELL_DV_MAX");

  float fx_sum = 0.0f;
  float fy_sum = 0.0f;
  float fz_sum = 0.0f;
  float cl_S_xx = 0.0f, cl_S_yy = 0.0f, cl_S_zz = 0.0f;
  float cl_S_xy = 0.0f, cl_S_xz = 0.0f, cl_S_yz = 0.0f;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const float mx = message.getVariable<float>("x");
    const float my = message.getVariable<float>("y");
    const float mz = message.getVariable<float>("z");
    const float mr = message.getVariable<float>("radius");

    // Direction: cell pushed away from lumen (opposite of lumen_cell_interaction)
    float dx = agent_x - mx;
    float dy = agent_y - my;
    float dz = agent_z - mz;
    float dist2 = dx * dx + dy * dy + dz * dz;

    if (dist2 <= 1e-20f) {
      dx = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      dy = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      dz = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
      cl_normalize3(dx, dy, dz);
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
    cl_normalize3(nx, ny, nz);

    const float overlap = r_contact - dist;
    const float f_pair = fmaxf(0.0f, LUMEN_K_LUMEN_CELL_REPULSION) * overlap;

    fx_sum += f_pair * nx;
    fy_sum += f_pair * ny;
    fz_sum += f_pair * nz;
    const float cl_s_coeff = -agent_nucleus_radius * f_pair;
    cl_S_xx += cl_s_coeff * nx * nx;
    cl_S_yy += cl_s_coeff * ny * ny;
    cl_S_zz += cl_s_coeff * nz * nz;
    cl_S_xy += cl_s_coeff * nx * ny;
    cl_S_xz += cl_s_coeff * nx * nz;
    cl_S_yz += cl_s_coeff * ny * nz;
  }

  const float inv_drag = (CELL_D_DUMPING > 1e-12f) ? (1.0f / CELL_D_DUMPING) : 0.0f;
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

  FLAMEGPU->setVariable<float>("cl_dvx", dvx);
  FLAMEGPU->setVariable<float>("cl_dvy", dvy);
  FLAMEGPU->setVariable<float>("cl_dvz", dvz);
  FLAMEGPU->setVariable<float>("cl_S_xx", cl_S_xx);
  FLAMEGPU->setVariable<float>("cl_S_yy", cl_S_yy);
  FLAMEGPU->setVariable<float>("cl_S_zz", cl_S_zz);
  FLAMEGPU->setVariable<float>("cl_S_xy", cl_S_xy);
  FLAMEGPU->setVariable<float>("cl_S_xz", cl_S_xz);
  FLAMEGPU->setVariable<float>("cl_S_yz", cl_S_yz);

  return flamegpu::ALIVE;
}
