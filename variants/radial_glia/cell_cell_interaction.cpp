/**
 * cell_cell_interaction  [RG variant override]
 *
 * Purpose:
 *   CELL-CELL mechanics with short-range contact repulsion and adhesion shell.
 *   Extends the base file to use type-pair adhesion/repulsion matrices
 *   (RG_ADHESION_MATRIX, RG_REPULSION_MATRIX) instead of per-self-type scalars,
 *   and boosts RG-RG adhesion proportionally to the geometric mean of the two
 *   cells' epithelialization levels (simulating junction belt tightening).
 *
 * Changes versus base cell_cell_interaction.cpp:
 *   1. Read epithelialization_level from agent and neighbour message.
 *   2. Look up K_adh and K_rep from 3x3 flattened matrices indexed by
 *      (agent_cell_type, neighbour_cell_type).
 *   3. Apply RG_EPITHELIAL_ADHESION_BOOST for RG-RG pairs.
 *
 * New env properties required (registered in variants/radial_glia/__init__.py):
 *   RG_ADHESION_MATRIX  [9 floats, nN/µm]
 *   RG_REPULSION_MATRIX [9 floats, nN/µm]
 *   RG_EPITHELIAL_ADHESION_BOOST [float, -]
 */

/**
 * cc_clampf
 *
 * Purpose:
 *   Clamp a scalar to the closed interval [lo, hi].
 */
FLAMEGPU_DEVICE_FUNCTION float cc_clampf(const float x, const float lo, const float hi) {
  return fminf(hi, fmaxf(lo, x));
}

/**
 * cc_normalize3
 *
 * Purpose:
 *   Normalize a 3D vector in-place; if near-zero, sets a default unit vector.
 */
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

FLAMEGPU_AGENT_FUNCTION(cell_cell_interaction, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    FLAMEGPU->setVariable<float>("cc_dvx", 0.0f);
    FLAMEGPU->setVariable<float>("cc_dvy", 0.0f);
    FLAMEGPU->setVariable<float>("cc_dvz", 0.0f);
    FLAMEGPU->setVariable<float>("cc_S_xx", 0.0f);
    FLAMEGPU->setVariable<float>("cc_S_yy", 0.0f);
    FLAMEGPU->setVariable<float>("cc_S_zz", 0.0f);
    FLAMEGPU->setVariable<float>("cc_S_xy", 0.0f);
    FLAMEGPU->setVariable<float>("cc_S_xz", 0.0f);
    FLAMEGPU->setVariable<float>("cc_S_yz", 0.0f);
    return flamegpu::ALIVE;
  }

  const int agent_id = FLAMEGPU->getVariable<int>("id");
  const int agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  const float agent_r = FLAMEGPU->getVariable<float>("radius");
  const float agent_nucleus_radius = FLAMEGPU->getVariable<float>("nucleus_radius");

  const uint8_t N_CELL_TYPES = 3; // WARNING: must match main python model N_CELL_TYPES
  const float CELL_D_DUMPING = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_D_DUMPING", agent_cell_type);
  const float CELL_CELL_ADHESION_RANGE = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_CELL_ADHESION_RANGE", agent_cell_type);
  const float CELL_CELL_DV_MAX = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_CELL_DV_MAX", agent_cell_type);

  // RG variant: type-pair adhesion/repulsion matrices (3x3, flattened row-major)
  const float RG_EPITHELIAL_ADHESION_BOOST = FLAMEGPU->environment.getProperty<float>("RG_EPITHELIAL_ADHESION_BOOST");

  // Own epithelialization level (for RG-RG boost)
  const float agent_epi = FLAMEGPU->getVariable<float>("epithelialization_level");

  float fx_sum = 0.0f;
  float fy_sum = 0.0f;
  float fz_sum = 0.0f;
  float cc_S_xx = 0.0f, cc_S_yy = 0.0f, cc_S_zz = 0.0f;
  float cc_S_xy = 0.0f, cc_S_xz = 0.0f, cc_S_yz = 0.0f;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const int message_id = message.getVariable<int>("id");
    if (message_id == agent_id) {
      continue;
    }

    if (message.getVariable<int>("dead") == 1) {
      continue;
    }

    const int nb_cell_type = message.getVariable<int>("cell_type");
    const float mx = message.getVariable<float>("x");
    const float my = message.getVariable<float>("y");
    const float mz = message.getVariable<float>("z");
    const float mr = message.getVariable<float>("radius");

    // Look up pair-specific adhesion and repulsion from matrices
    const int pair_idx = agent_cell_type * N_CELL_TYPES + nb_cell_type;
    float K_adh = FLAMEGPU->environment.getProperty<float, 9>("RG_ADHESION_MATRIX",  pair_idx);
    const float K_rep = FLAMEGPU->environment.getProperty<float, 9>("RG_REPULSION_MATRIX", pair_idx);

    // RG-RG epithelialization boost
    if (agent_cell_type == 2 && nb_cell_type == 2) {
      const float nb_epi = message.getVariable<float>("epithelialization_level");
      const float epith_min = fminf(agent_epi, nb_epi);
      K_adh *= (1.0f + (RG_EPITHELIAL_ADHESION_BOOST - 1.0f) * epith_min);
    }

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
      f_pair = fmaxf(0.0f, K_rep) * overlap;
    } else if (dist < r_adh_end) {
      const float s = (dist - r_contact) / fmaxf(1e-6f, CELL_CELL_ADHESION_RANGE);
      const float bell = 4.0f * s * (1.0f - s);
      f_pair = -fmaxf(0.0f, K_adh) * bell;
    }

    fx_sum += f_pair * nx;
    fy_sum += f_pair * ny;
    fz_sum += f_pair * nz;
    const float cc_s_coeff = -agent_nucleus_radius * f_pair;
    cc_S_xx += cc_s_coeff * nx * nx;
    cc_S_yy += cc_s_coeff * ny * ny;
    cc_S_zz += cc_s_coeff * nz * nz;
    cc_S_xy += cc_s_coeff * nx * ny;
    cc_S_xz += cc_s_coeff * nx * nz;
    cc_S_yz += cc_s_coeff * ny * nz;
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
  FLAMEGPU->setVariable<float>("cc_S_xx", cc_S_xx);
  FLAMEGPU->setVariable<float>("cc_S_yy", cc_S_yy);
  FLAMEGPU->setVariable<float>("cc_S_zz", cc_S_zz);
  FLAMEGPU->setVariable<float>("cc_S_xy", cc_S_xy);
  FLAMEGPU->setVariable<float>("cc_S_xz", cc_S_xz);
  FLAMEGPU->setVariable<float>("cc_S_yz", cc_S_yz);
  return flamegpu::ALIVE;
}
