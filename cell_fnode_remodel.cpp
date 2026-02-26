FLAMEGPU_DEVICE_FUNCTION float cfnr_length3(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}

FLAMEGPU_DEVICE_FUNCTION void cfnr_normalize3(float &x, float &y, float &z) {
  const float n = cfnr_length3(x, y, z);
  if (n > 1e-12f) {
    x /= n;
    y /= n;
    z /= n;
  } else {
    x = 1.0f;
    y = 0.0f;
    z = 0.0f;
  }
}

FLAMEGPU_DEVICE_FUNCTION float cfnr_hill(const float x, const float k, const float h) {
  const float xx = fmaxf(0.0f, x);
  const float hh = fmaxf(1.0f, h);
  const float kk = fmaxf(1e-6f, k);
  const float num = powf(xx, hh);
  const float den = powf(kk, hh) + num;
  if (den <= 1e-20f) {
    return 0.0f;
  }
  return num / den;
}

/**
 * cell_fnode_remodel
 *
 * Purpose:
 *   Probabilistically create a single FNODE around a CELL and request reciprocal
 *   parent-link update through environment macros.
 *
 * Inputs:
 *   - Nearby FNODE spatial messages (id, x, y, z, connectivity_count)
 *   - CELL state (position, stress proxy, concentration, dead)
 *   - Remodeling environment properties and macro properties
 *
 * Outputs:
 *   - Optional newborn FNODE via agent output (max 1 per CELL per step)
 *   - Updated CELL cooldown variable `fnode_birth_cooldown`
 */
FLAMEGPU_AGENT_FUNCTION(cell_fnode_remodel, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    return flamegpu::ALIVE;
  }

    const uint32_t INCLUDE_NETWORK_REMODELING = FLAMEGPU->environment.getProperty<uint32_t>("INCLUDE_NETWORK_REMODELING");
    if (INCLUDE_NETWORK_REMODELING == 0) {
    return flamegpu::ALIVE;
  }

  const uint8_t MAX_CONNECTIVITY = 8;      // must match model.py
  const uint8_t N_SPECIES = 2;             // must match model.py

  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float FNODE_BIRTH_K_0 = FLAMEGPU->environment.getProperty<float>("FNODE_BIRTH_K_0");
  const uint32_t FNODE_BIRTH_SPECIES_INDEX = FLAMEGPU->environment.getProperty<uint32_t>("FNODE_BIRTH_SPECIES_INDEX");
  const float FNODE_BIRTH_K_C = FLAMEGPU->environment.getProperty<float>("FNODE_BIRTH_K_C");
  const float FNODE_BIRTH_HILL_CONC = FLAMEGPU->environment.getProperty<float>("FNODE_BIRTH_HILL_CONC");
  const float FNODE_BIRTH_K_SIGMA = FLAMEGPU->environment.getProperty<float>("FNODE_BIRTH_K_SIGMA");
  const float FNODE_BIRTH_HILL_SIGMA = FLAMEGPU->environment.getProperty<float>("FNODE_BIRTH_HILL_SIGMA");
  const float FNODE_BIRTH_RADIUS = FLAMEGPU->environment.getProperty<float>("FNODE_BIRTH_RADIUS");
  const float FNODE_BIRTH_LINK_MAX_DISTANCE = FLAMEGPU->environment.getProperty<float>("FNODE_BIRTH_LINK_MAX_DISTANCE");
  const float FNODE_BIRTH_REFRACTORY = FLAMEGPU->environment.getProperty<float>("FNODE_BIRTH_REFRACTORY");
  const float FIBRE_SEGMENT_K_ELAST = FLAMEGPU->environment.getProperty<float>("FIBRE_SEGMENT_K_ELAST");
  const float FIBRE_SEGMENT_D_DUMPING = FLAMEGPU->environment.getProperty<float>("FIBRE_SEGMENT_D_DUMPING");
  const float FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE");

  float cooldown = FLAMEGPU->getVariable<float>("fnode_birth_cooldown");
  cooldown = fmaxf(0.0f, cooldown - TIME_STEP);
  FLAMEGPU->setVariable<float>("fnode_birth_cooldown", cooldown);
  if (cooldown > 0.0f) {
    return flamegpu::ALIVE;
  }

  float c_sp0 = 0.0f;
  if (FNODE_BIRTH_SPECIES_INDEX < N_SPECIES) {
    c_sp0 = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", FNODE_BIRTH_SPECIES_INDEX);
  }
  const float sig_l1 = fmaxf(0.0f, FLAMEGPU->getVariable<float>("sig_eig_1"));
  const float gate_c = cfnr_hill(c_sp0, FNODE_BIRTH_K_C, FNODE_BIRTH_HILL_CONC);
  const float gate_s = cfnr_hill(sig_l1, FNODE_BIRTH_K_SIGMA, FNODE_BIRTH_HILL_SIGMA);
  const float k_birth = fmaxf(0.0f, FNODE_BIRTH_K_0) * (0.25f + 0.75f * gate_c) * (0.25f + 0.75f * gate_s);
  const float p_birth = 1.0f - expf(-k_birth * TIME_STEP);
  if (FLAMEGPU->random.uniform<float>(0.0f, 1.0f) >= p_birth) {
    return flamegpu::ALIVE;
  }

  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  const float max_link_r2 = FNODE_BIRTH_LINK_MAX_DISTANCE * FNODE_BIRTH_LINK_MAX_DISTANCE;
  int closest_fnode_id = -1;
  float best_parent_x = 0.0f;
  float best_parent_y = 0.0f;
  float best_parent_z = 0.0f;
  float best_r2 = max_link_r2;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const uint8_t degree = message.getVariable<uint8_t>("connectivity_count");
    if (degree >= MAX_CONNECTIVITY) {
      continue;
    }
    const float mx = message.getVariable<float>("x");
    const float my = message.getVariable<float>("y");
    const float mz = message.getVariable<float>("z");
    const float dx = mx - agent_x;
    const float dy = my - agent_y;
    const float dz = mz - agent_z;
    const float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < best_r2) {
      best_r2 = r2;
      closest_fnode_id = message.getVariable<int>("id");
      best_parent_x = mx;
      best_parent_y = my;
      best_parent_z = mz;
    }
  }

  if (closest_fnode_id < 0) {
    return flamegpu::ALIVE;
  }

  auto MACRO_MAX_GLOBAL_FNODE_ID = FLAMEGPU->environment.getMacroProperty<int, 1>("MACRO_MAX_GLOBAL_FNODE_ID");

  const int new_fnode_id = MACRO_MAX_GLOBAL_FNODE_ID.addAtomic(1);

  float dir_x = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
  float dir_y = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
  float dir_z = FLAMEGPU->random.uniform<float>(-1.0f, 1.0f);
  cfnr_normalize3(dir_x, dir_y, dir_z);

  const float new_x = agent_x + dir_x * FNODE_BIRTH_RADIUS;
  const float new_y = agent_y + dir_y * FNODE_BIRTH_RADIUS;
  const float new_z = agent_z + dir_z * FNODE_BIRTH_RADIUS;

  const float pdx = new_x - best_parent_x;
  const float pdy = new_y - best_parent_y;
  const float pdz = new_z - best_parent_z;
  const float parent_dist = fmaxf(1e-6f, cfnr_length3(pdx, pdy, pdz));

  printf("CELL %d at (%f, %f, %f) is creating FNODE %d at (%f, %f, %f) with parent FNODE %d at (%f, %f, %f)\n",
    FLAMEGPU->getVariable<int>("id"),
    agent_x, agent_y, agent_z,
    new_fnode_id,
    new_x, new_y, new_z,
    closest_fnode_id,
    best_parent_x, best_parent_y, best_parent_z
  );

  FLAMEGPU->agent_out.setVariable<int>("id", new_fnode_id);
  FLAMEGPU->agent_out.setVariable<float>("x", new_x);
  FLAMEGPU->agent_out.setVariable<float>("y", new_y);
  FLAMEGPU->agent_out.setVariable<float>("z", new_z);
  FLAMEGPU->agent_out.setVariable<float>("vx", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("vy", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("vz", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("fx", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("fy", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("fz", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("k_elast", FIBRE_SEGMENT_K_ELAST);
  FLAMEGPU->agent_out.setVariable<float>("d_dumping", FIBRE_SEGMENT_D_DUMPING);
  FLAMEGPU->agent_out.setVariable<float>("boundary_fx", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("boundary_fy", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("boundary_fz", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bx_pos", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bx_neg", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_by_pos", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_by_neg", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bz_pos", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bz_neg", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bx_pos_y", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bx_pos_z", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bx_neg_y", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bx_neg_z", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_by_pos_x", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_by_pos_z", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_by_neg_x", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_by_neg_z", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bz_pos_x", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bz_pos_y", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bz_neg_x", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_bz_neg_y", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_extension", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("f_compression", 0.0f);
  FLAMEGPU->agent_out.setVariable<float>("elastic_energy", 0.0f);
  FLAMEGPU->agent_out.setVariable<uint8_t>("connectivity_count", 1);
  FLAMEGPU->agent_out.setVariable<float>("degradation", 0.0f);
  FLAMEGPU->agent_out.setVariable<int>("marked_for_removal", 0);
  FLAMEGPU->agent_out.setVariable<int>("closest_fnode_id", closest_fnode_id);
  FLAMEGPU->agent_out.setVariable<uint8_t>("clamped_bx_pos", 0);
  FLAMEGPU->agent_out.setVariable<uint8_t>("clamped_bx_neg", 0);
  FLAMEGPU->agent_out.setVariable<uint8_t>("clamped_by_pos", 0);
  FLAMEGPU->agent_out.setVariable<uint8_t>("clamped_by_neg", 0);
  FLAMEGPU->agent_out.setVariable<uint8_t>("clamped_bz_pos", 0);
  FLAMEGPU->agent_out.setVariable<uint8_t>("clamped_bz_neg", 0);

  FLAMEGPU->agent_out.setVariable<float, MAX_CONNECTIVITY>("linked_nodes", 0, static_cast<float>(closest_fnode_id));
  FLAMEGPU->agent_out.setVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", 0, parent_dist);
  for (int i = 1; i < MAX_CONNECTIVITY; i++) {
    FLAMEGPU->agent_out.setVariable<float, MAX_CONNECTIVITY>("linked_nodes", i, -1.0f);
    FLAMEGPU->agent_out.setVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE);
  }

  FLAMEGPU->setVariable<float>("fnode_birth_cooldown", fmaxf(0.0f, FNODE_BIRTH_REFRACTORY));

  return flamegpu::ALIVE;
}
