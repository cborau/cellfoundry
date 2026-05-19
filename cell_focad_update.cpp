/**
 * cell_focad_update
 *
 * Accumulates the FOCAD stresslet tensor from focal adhesion anchor forces
 * into focad_S_* agent variables, which are consumed each step by
 * cell_stress_state_update (along with cc_S_*, cf_S_*, cl_S_* from contact
 * interactions) to compute the combined nucleus stress and update strain state.
 *
 * Also handles FOCAD birth / refractory logic.  Birth gating uses sig_eig_1
 * from the previous timestep (the total combined stress), ensuring a
 * consistent mechanical stimulus across all stress sources.
 *
 * Units:
 *   length: um,  force: nN,  stresslet: nN·um
 */
FLAMEGPU_AGENT_FUNCTION(cell_focad_update, flamegpu::MessageBucket, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    return flamegpu::ALIVE; // Note: if DEAD_CELLS_DISAPPEAR = True, a dead CELL agent remains ALIVE for flamegpu purposes and may still interact with other agents.
  }
  // -------------------------
  // Read CELL agent state
  // -------------------------
  const int agent_id = FLAMEGPU->getVariable<int>("id");
  const int agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");

  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_focad_birth_cooldown = FLAMEGPU->getVariable<float>("focad_birth_cooldown");
  const float agent_orx = FLAMEGPU->getVariable<float>("orx");
  const float agent_ory = FLAMEGPU->getVariable<float>("ory");
  const float agent_orz = FLAMEGPU->getVariable<float>("orz");

  const float agent_radius = FLAMEGPU->getVariable<float>("radius");
  // Previous step's combined principal stress — used for FOCAD birth gating
  const float sig_eig_1_prev = FLAMEGPU->getVariable<float>("sig_eig_1");

  // -------------------------
  // Environment parameters
  // -------------------------
  const uint8_t N_CELL_TYPES = 3; // WARNING: must match model.py
  const float TIME_STEP         = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const uint32_t ENABLE_FOCAD_BIRTH = FLAMEGPU->environment.getProperty<uint32_t>("ENABLE_FOCAD_BIRTH");
  const uint32_t FOCAD_BIRTH_SPECIES_INDEX = FLAMEGPU->environment.getProperty<uint32_t>("FOCAD_BIRTH_SPECIES_INDEX");
  const uint32_t FOCAD_BIRTH_N_MIN = static_cast<uint32_t>(FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_N_MIN", agent_cell_type));
  const uint32_t FOCAD_BIRTH_N_MAX = static_cast<uint32_t>(FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_N_MAX", agent_cell_type));
  const float FOCAD_BIRTH_K_0 = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_K_0", agent_cell_type);
  const float FOCAD_BIRTH_K_MAX = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_K_MAX", agent_cell_type);
  const float FOCAD_BIRTH_K_SIGMA = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_K_SIGMA", agent_cell_type);
  const float FOCAD_BIRTH_HILL_SIGMA = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_HILL_SIGMA", agent_cell_type);
  const float FOCAD_BIRTH_K_C = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_K_C", agent_cell_type);
  const float FOCAD_BIRTH_HILL_CONC = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_HILL_CONC", agent_cell_type);
  const float FOCAD_BIRTH_REFRACTORY = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_BIRTH_REFRACTORY", agent_cell_type);
  const float FOCAD_REST_LENGTH_0 = FLAMEGPU->environment.getProperty<float>("FOCAD_REST_LENGTH_0");
  const float FOCAD_K_FA = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_K_FA", agent_cell_type);
  const float FOCAD_F_MAX = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_F_MAX", agent_cell_type);
  const float FOCAD_V_C = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_V_C", agent_cell_type);
  const float FOCAD_K_ON = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_K_ON", agent_cell_type);
  const float FOCAD_K_OFF_0 = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_K_OFF_0", agent_cell_type);
  const float FOCAD_F_C = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_F_C", agent_cell_type);
  const float FOCAD_K_REINF = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("FOCAD_K_REINF", agent_cell_type);

  const uint8_t N_ANCHOR_POINTS = 50; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  const uint8_t N_SPECIES = 3; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  // -------------------------
  // Accumulate stresslet S = sum_i sym(r_i ⊗ f_i)
  // where r_i = x_i - x_c
  // -------------------------
  float agent_S_xx = 0.0f, agent_S_yy = 0.0f, agent_S_zz = 0.0f;
  float agent_S_xy = 0.0f, agent_S_xz = 0.0f, agent_S_yz = 0.0f;
  uint32_t current_focad_count = 0;

  // Iterate over all FOCAD messages addressed to this cell id
  for (const auto &message : FLAMEGPU->message_in(agent_id)) {
    current_focad_count += 1;
    // Anchor position on nucleus and force at the anchor (message variables)
    const float message_x_i = message.getVariable<float>("x_i"); // [um]
    const float message_y_i = message.getVariable<float>("y_i"); // [um]
    const float message_z_i = message.getVariable<float>("z_i"); // [um]

    const float message_fx  = message.getVariable<float>("fx");  // [nN]
    const float message_fy  = message.getVariable<float>("fy");  // [nN]
    const float message_fz  = message.getVariable<float>("fz");  // [nN]

    //printf("cell_update_stress -- FOCAD message for CELL %d: anchor=(%.4f, %.4f, %.4f) um, force=(%.4f, %.4f, %.4f) nN\n", agent_id, message_x_i, message_y_i, message_z_i, message_fx, message_fy, message_fz);

    // Lever arm from nucleus center to adhesion location
    const float agent_rx = message_x_i - agent_x; // [um]
    const float agent_ry = message_y_i - agent_y; // [um]
    const float agent_rz = message_z_i - agent_z; // [um]

    // Symmetric stresslet contributions
    agent_S_xx += agent_rx * message_fx;                                  
    agent_S_yy += agent_ry * message_fy;                                  
    agent_S_zz += agent_rz * message_fz;                                  
    agent_S_xy += 0.5f * (agent_rx * message_fy + agent_ry * message_fx); 
    agent_S_xz += 0.5f * (agent_rx * message_fz + agent_rz * message_fx);
    agent_S_yz += 0.5f * (agent_ry * message_fz + agent_rz * message_fy);
    // printf("cell_update_stress -- message: r=(%.3f, %.3f, %.3f) f=(%.3f, %.3f, %.3f) S_contrib=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)\n", 
    //        agent_rx, agent_ry, agent_rz, message_fx, message_fy, message_fz,
    //        agent_S_xx,
    //        agent_S_yy,
    //        agent_S_zz,
    //        agent_S_xy,
    //        agent_S_xz,
    //        agent_S_yz);

  }

  // Write FOCAD stresslet accumulators for cell_stress_state_update
  FLAMEGPU->setVariable<float>("focad_S_xx", agent_S_xx);
  FLAMEGPU->setVariable<float>("focad_S_yy", agent_S_yy);
  FLAMEGPU->setVariable<float>("focad_S_zz", agent_S_zz);
  FLAMEGPU->setVariable<float>("focad_S_xy", agent_S_xy);
  FLAMEGPU->setVariable<float>("focad_S_xz", agent_S_xz);
  FLAMEGPU->setVariable<float>("focad_S_yz", agent_S_yz);

  // -------------------------
  // FOCAD birth from CELL (bounded, stress+concentration gated)
  // -------------------------
  if (agent_focad_birth_cooldown > 0.0f) {
    agent_focad_birth_cooldown = fmaxf(0.0f, agent_focad_birth_cooldown - TIME_STEP);
  }

  if (ENABLE_FOCAD_BIRTH != 0 && FOCAD_BIRTH_N_MAX > 0) {
    const uint32_t idx_sp = FOCAD_BIRTH_SPECIES_INDEX < N_SPECIES ? FOCAD_BIRTH_SPECIES_INDEX : 0;
    const float c_raw = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", idx_sp);
    const float c = fmaxf(0.0f, c_raw);
    const float sigma_pos = fmaxf(0.0f, sig_eig_1_prev);

    const float hill_sigma = fmaxf(1.0f, FOCAD_BIRTH_HILL_SIGMA);
    const float sigma_pow = powf(sigma_pos, hill_sigma);
    const float ks_pow = powf(fmaxf(1e-12f, FOCAD_BIRTH_K_SIGMA), hill_sigma);
    const float hs_denom = fmaxf(1e-12f, ks_pow + sigma_pow);
    const float h_sigma = sigma_pow / hs_denom;

    const float hill_conc = fmaxf(1.0f, FOCAD_BIRTH_HILL_CONC);
    const float c_pow = powf(c, hill_conc);
    const float kc_pow = powf(fmaxf(1e-12f, FOCAD_BIRTH_K_C), hill_conc);
    const float hc_denom = fmaxf(1e-12f, kc_pow + c_pow);
    const float h_c = c_pow / hc_denom;

    const uint32_t n_min = FOCAD_BIRTH_N_MIN;
    const uint32_t n_max = FOCAD_BIRTH_N_MAX;
    const uint32_t enforced_min = (n_min < n_max) ? n_min : n_max;

    // Enforce one-agent-per-step creation rule: when below minimum, force at most one birth this step.
    const int force_min_birth = ((current_focad_count < enforced_min) && (current_focad_count < n_max)) ? 1 : 0;

    const float h_birth = h_sigma * h_c;
    const float target_f = static_cast<float>(enforced_min) + static_cast<float>(n_max - enforced_min) * h_birth;
    uint32_t target_n = static_cast<uint32_t>(target_f + 0.5f);
    if (target_n < enforced_min) target_n = enforced_min;
    if (target_n > n_max) target_n = n_max;

    int do_birth = 0;
    if (force_min_birth != 0) {
      do_birth = 1;
    } else if ((current_focad_count < target_n) && (current_focad_count < n_max) && (agent_focad_birth_cooldown <= 0.0f)) {
      const float k_birth = fmaxf(0.0f, FOCAD_BIRTH_K_0 + FOCAD_BIRTH_K_MAX * h_birth);
      const float p_birth = 1.0f - expf(-k_birth * TIME_STEP);
      const float r_birth = FLAMEGPU->random.uniform<float>(0.0f, 1.0f);
      if (r_birth < p_birth) {
        do_birth = 1;
      }
    }

    if (do_birth != 0) {
        // Place new FOCAD at the cell leading edge
        const float lead_x = agent_x + agent_radius * agent_orx;
        const float lead_y = agent_y + agent_radius * agent_ory;
        const float lead_z = agent_z + agent_radius * agent_orz;
        // Find anchor closest to leading edge (uses stored x_i from previous step)
        float best_anchor_x = agent_x;
        float best_anchor_y = agent_y;
        float best_anchor_z = agent_z;
        int best_anchor_id = 0;
        float best_anchor_r2 = 1e30f;
        for (unsigned int a = 0; a < N_ANCHOR_POINTS; ++a) {
          const float ax = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("x_i", a);
          const float ay = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("y_i", a);
          const float az = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("z_i", a);
          const float dax = ax - lead_x;
          const float day = ay - lead_y;
          const float daz = az - lead_z;
          const float dr2 = dax * dax + day * day + daz * daz;
          if (dr2 < best_anchor_r2) {
            best_anchor_r2 = dr2;
            best_anchor_x = ax;
            best_anchor_y = ay;
            best_anchor_z = az;
            best_anchor_id = static_cast<int>(a);
          }
        }
        const int new_focad_id = FLAMEGPU->agent_out.getID();
        FLAMEGPU->agent_out.setVariable<int>("id", new_focad_id);
        FLAMEGPU->agent_out.setVariable<int>("cell_id", agent_id);
        FLAMEGPU->agent_out.setVariable<int>("cell_type", agent_cell_type);
        FLAMEGPU->agent_out.setVariable<int>("fnode_id", -1);
        FLAMEGPU->agent_out.setVariable<float>("x", lead_x);
        FLAMEGPU->agent_out.setVariable<float>("y", lead_y);
        FLAMEGPU->agent_out.setVariable<float>("z", lead_z);
        FLAMEGPU->agent_out.setVariable<float>("vx", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("vy", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("vz", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("fx", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("fy", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("fz", 0.0f);
        FLAMEGPU->agent_out.setVariable<int>("anchor_id", best_anchor_id);
        FLAMEGPU->agent_out.setVariable<float>("x_i", best_anchor_x);
        FLAMEGPU->agent_out.setVariable<float>("y_i", best_anchor_y);
        FLAMEGPU->agent_out.setVariable<float>("z_i", best_anchor_z);
        FLAMEGPU->agent_out.setVariable<float>("x_c", agent_x);
        FLAMEGPU->agent_out.setVariable<float>("y_c", agent_y);
        FLAMEGPU->agent_out.setVariable<float>("z_c", agent_z);
        FLAMEGPU->agent_out.setVariable<float>("orx", agent_orx);
        FLAMEGPU->agent_out.setVariable<float>("ory", agent_ory);
        FLAMEGPU->agent_out.setVariable<float>("orz", agent_orz);
        FLAMEGPU->agent_out.setVariable<float>("rest_length_0", FOCAD_REST_LENGTH_0);
        FLAMEGPU->agent_out.setVariable<float>("rest_length", FOCAD_REST_LENGTH_0);
        FLAMEGPU->agent_out.setVariable<float>("k_fa", FOCAD_K_FA);
        FLAMEGPU->agent_out.setVariable<float>("f_max", FOCAD_F_MAX);
        FLAMEGPU->agent_out.setVariable<int>("attached", 0);
        FLAMEGPU->agent_out.setVariable<uint8_t>("active", 1);
        FLAMEGPU->agent_out.setVariable<float>("v_c", FOCAD_V_C);
        FLAMEGPU->agent_out.setVariable<uint8_t>("fa_state", 1);
        FLAMEGPU->agent_out.setVariable<float>("age", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("detached_age", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_on", FOCAD_K_ON);
        FLAMEGPU->agent_out.setVariable<float>("k_off_0", FOCAD_K_OFF_0);
        FLAMEGPU->agent_out.setVariable<float>("f_c", FOCAD_F_C);
        FLAMEGPU->agent_out.setVariable<float>("k_reinf", FOCAD_K_REINF);
        FLAMEGPU->agent_out.setVariable<float>("f_mag", 0.0f);
        FLAMEGPU->agent_out.setVariable<int>("is_front", 0);
        FLAMEGPU->agent_out.setVariable<int>("is_rear", 0);
        FLAMEGPU->agent_out.setVariable<int>("attached_front", 0);
        FLAMEGPU->agent_out.setVariable<int>("attached_rear", 0);
        FLAMEGPU->agent_out.setVariable<float>("frontness_front", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("frontness_rear", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_on_eff_front", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_on_eff_rear", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_off_0_eff_front", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("k_off_0_eff_rear", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("linc_prev_total_length", 0.0f);
        // Keep refractory only for probabilistic births. Forced replenishment can happen in consecutive steps until minimum is reached.
        if (force_min_birth == 0) {
        agent_focad_birth_cooldown = fmaxf(0.0f, FOCAD_BIRTH_REFRACTORY);
        }
      }
  }

  FLAMEGPU->setVariable<float>("focad_birth_cooldown", agent_focad_birth_cooldown);

  return flamegpu::ALIVE;
}
