/**
 * variants/radial_glia/cell_cycle.cpp
 *
 * Purpose:
 *   Override of the base cell_cycle.cpp for the radial-glia variant.
 *
 *   Extensions vs the base function:
 *     1. RG-specific agent variables (rg_commit_level, epithelialization_level,
 *        apx/apy/apz, rosette_maturity, rg_neighbour_density, morphogen_local,
 *        substrate_anchor_x/y) are explicitly propagated to daughter
 *        cells on division — the base function leaves them at default (0 / -1).
 *     2. Asymmetric division for RG cells (cell_type == 2):
 *          Parent  — retains primary cilium, high Notch → stays RG;
 *                    commitment and epithelialization are slightly diluted by
 *                    the division event.
 *          Daughter — inherits Delta excess → re-enters as NEP (cell_type=1);
 *                    commitment is reset to just above the iPSC→NEP threshold;
 *                    polarity is scrambled (random apical angle in xy, apz=0).
 *     3. Symmetric division for iPSC and NEP (cell_type == 0/1):
 *          Both cells inherit the parent's rg_commit_level and epithelialization_level
 *          unchanged; apical vector is preserved.
 *
 * Notes:
 *   - The constants N_SPECIES, N_ANCHOR_POINTS, N_CELL_TYPES must match
 *     the values in model.py.
 *   - ECM_POPULATION_SIZE is not used in this function.
 */

FLAMEGPU_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
  x /= divisor;
  y /= divisor;
  z /= divisor;
}
FLAMEGPU_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}

FLAMEGPU_AGENT_FUNCTION(cell_cycle, flamegpu::MessageNone, flamegpu::MessageNone) {
  int id = FLAMEGPU->getVariable<int>("id");
  auto MACRO_MAX_GLOBAL_CELL_ID = FLAMEGPU->environment.getMacroProperty<int, 1>("MACRO_MAX_GLOBAL_CELL_ID");
  const uint32_t DEAD_CELLS_DISAPPEAR = FLAMEGPU->environment.getProperty<uint32_t>("DEAD_CELLS_DISAPPEAR");
  int agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");
  int agent_max_global_cell_id = FLAMEGPU->getVariable<int>("max_global_cell_id");
  const int agent_marked_for_removal = FLAMEGPU->getVariable<int>("marked_for_removal");
  if (agent_marked_for_removal == 1 && DEAD_CELLS_DISAPPEAR != 0) {
    return flamegpu::DEAD;
  }
  const int agent_dead = FLAMEGPU->getVariable<int>("dead");
  if (agent_dead == 1) {
    return flamegpu::ALIVE;
  }
  FLAMEGPU->setVariable<int>("just_divided", 0);
  FLAMEGPU->setVariable<int>("daughter_id", -1);

  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");
  // Agent orientation
  float agent_orx = FLAMEGPU->getVariable<float>("orx");
  float agent_ory = FLAMEGPU->getVariable<float>("ory");
  float agent_orz = FLAMEGPU->getVariable<float>("orz");
  float agent_alignment = FLAMEGPU->getVariable<float>("alignment");

  float agent_k_elast = FLAMEGPU->getVariable<float>("k_elast");
  float agent_d_dumping = FLAMEGPU->getVariable<float>("d_dumping");
  float agent_speed_ref = FLAMEGPU->getVariable<float>("speed_ref");
  float agent_radius = FLAMEGPU->getVariable<float>("radius");
  float agent_nucleus_radius = FLAMEGPU->getVariable<float>("nucleus_radius");
  float agent_focad_birth_cooldown = FLAMEGPU->getVariable<float>("focad_birth_cooldown");
  float agent_damage = FLAMEGPU->getVariable<float>("damage");
  float agent_sig_l1 = FLAMEGPU->getVariable<float>("sig_eig_1");
  int agent_completed_cycles = FLAMEGPU->getVariable<int>("completed_cycles");

  const uint8_t N_SPECIES = 3;        // WARNING: must match main python model
  const uint8_t N_ANCHOR_POINTS = 50; // WARNING: must match main python model

  float agent_k_consumption[N_SPECIES] = {};
  float agent_k_production[N_SPECIES] = {};
  float agent_k_reaction[N_SPECIES] = {};
  float agent_C_sp[N_SPECIES] = {};
  float agent_M_sp[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    agent_k_consumption[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_consumption", i);
    agent_k_production[i]  = FLAMEGPU->getVariable<float, N_SPECIES>("k_production", i);
    agent_k_reaction[i]    = FLAMEGPU->getVariable<float, N_SPECIES>("k_reaction", i);
    agent_C_sp[i]          = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
    agent_M_sp[i]          = FLAMEGPU->getVariable<float, N_SPECIES>("M_sp", i);
  }

  float agent_x_i[N_ANCHOR_POINTS] = {};
  float agent_y_i[N_ANCHOR_POINTS] = {};
  float agent_z_i[N_ANCHOR_POINTS] = {};
  float agent_u_ref_x_i[N_ANCHOR_POINTS] = {};
  float agent_u_ref_y_i[N_ANCHOR_POINTS] = {};
  float agent_u_ref_z_i[N_ANCHOR_POINTS] = {};
  for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    agent_x_i[i]       = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("x_i", i);
    agent_y_i[i]       = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("y_i", i);
    agent_z_i[i]       = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("z_i", i);
    agent_u_ref_x_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_x_i", i);
    agent_u_ref_y_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_y_i", i);
    agent_u_ref_z_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_z_i", i);
  }

  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");

  const uint8_t N_CELL_TYPES = 3; // WARNING: must match main python model N_CELL_TYPES
  const float CELL_RADIUS         = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_RADIUS", agent_cell_type);
  const float CELL_NUCLEUS_RADIUS = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_NUCLEUS_RADIUS", agent_cell_type);
  const float CELL_CYCLE_DURATION = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_CYCLE_DURATION", agent_cell_type);
  const float CYCLE_PHASE_G1_START   = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CYCLE_PHASE_G1_START", agent_cell_type);
  const float CYCLE_PHASE_S_START    = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CYCLE_PHASE_S_START", agent_cell_type);
  const float CYCLE_PHASE_G2_START   = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CYCLE_PHASE_G2_START", agent_cell_type);
  const float CYCLE_PHASE_M_START    = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CYCLE_PHASE_M_START", agent_cell_type);
  const float CYCLE_PHASE_G1_DURATION = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CYCLE_PHASE_G1_DURATION", agent_cell_type);
  const float CYCLE_PHASE_M_DURATION  = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CYCLE_PHASE_M_DURATION", agent_cell_type);
  const float hypoxia_threshold      = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_HYPOXIA_THRESHOLD", agent_cell_type);
  const float nutrient_threshold     = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_NUTRIENT_THRESHOLD", agent_cell_type);
  const float stress_threshold       = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_STRESS_THRESHOLD", agent_cell_type);
  const float hypoxia_damage_rate    = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_HYPOXIA_DAMAGE_RATE", agent_cell_type);
  const float nutrient_damage_rate   = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_NUTRIENT_DAMAGE_RATE", agent_cell_type);
  const float stress_damage_rate     = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_STRESS_DAMAGE_RATE", agent_cell_type);
  const float basal_damage_repair_rate = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_BASAL_DAMAGE_REPAIR_RATE", agent_cell_type);
  const float acute_hypoxia_threshold  = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_ACUTE_HYPOXIA_THRESHOLD", agent_cell_type);
  const float acute_nutrient_threshold = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_ACUTE_NUTRIENT_THRESHOLD", agent_cell_type);
  const float acute_stress_threshold   = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_ACUTE_STRESS_THRESHOLD", agent_cell_type);

  const uint32_t oxygen_idx   = FLAMEGPU->environment.getProperty<uint32_t>("OXYGEN_SPECIES_INDEX");
  const uint32_t nutrient_idx = FLAMEGPU->environment.getProperty<uint32_t>("NUTRIENT_SPECIES_INDEX");
  const float oxygen_proxy         = agent_C_sp[oxygen_idx];
  const float nutrient_proxy       = agent_C_sp[nutrient_idx];
  const float tensile_stress_proxy = fmaxf(0.0f, agent_sig_l1);

  const float division_rate_multiplier    = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("DIVISION_RATE_MULTIPLIER", agent_cell_type);
  const float damage_accumulation_multiplier = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("DAMAGE_ACCUMULATION_MULTIPLIER", agent_cell_type);
  const float damage_repair_multiplier    = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("DAMAGE_REPAIR_MULTIPLIER", agent_cell_type);
  const float damage_death_threshold      = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("DAMAGE_DEATH_THRESHOLD", agent_cell_type);

  // -------------------------------------------------------------------------
  // RG variant: read state variables needed for division inheritance
  // -------------------------------------------------------------------------
  const float rg_commit = FLAMEGPU->getVariable<float>("rg_commit_level");
  const float epi       = FLAMEGPU->getVariable<float>("epithelialization_level");
  const float apx       = FLAMEGPU->getVariable<float>("apx");
  const float apy       = FLAMEGPU->getVariable<float>("apy");
  const float apz       = FLAMEGPU->getVariable<float>("apz");
  // Threshold for iPSC→NEP transition (used to initialise RG daughter)
  const float RG_COMMIT_THRESHOLD_NEP = FLAMEGPU->environment.getProperty<float>("RG_COMMIT_THRESHOLD_NEP");

  // Biologically grounded death pathways via cumulative damage
  if (oxygen_proxy < hypoxia_threshold) {
    const float severity = (hypoxia_threshold - oxygen_proxy) / fmaxf(1e-6f, hypoxia_threshold);
    agent_damage += TIME_STEP * damage_accumulation_multiplier * hypoxia_damage_rate * severity;
  }
  if (nutrient_proxy < nutrient_threshold) {
    const float severity = (nutrient_threshold - nutrient_proxy) / fmaxf(1e-6f, nutrient_threshold);
    agent_damage += TIME_STEP * damage_accumulation_multiplier * nutrient_damage_rate * severity;
  }
  if (tensile_stress_proxy > stress_threshold) {
    const float severity = (tensile_stress_proxy - stress_threshold) / fmaxf(1e-6f, stress_threshold);
    agent_damage += TIME_STEP * damage_accumulation_multiplier * stress_damage_rate * severity;
  }

  agent_damage -= TIME_STEP * damage_repair_multiplier * basal_damage_repair_rate;
  agent_damage = fminf(1.0f, fmaxf(0.0f, agent_damage));
  FLAMEGPU->setVariable<float>("damage", agent_damage);
  int death_cause = -1;
  if (oxygen_proxy < acute_hypoxia_threshold) {
    death_cause = 0;
  } else if (nutrient_proxy < acute_nutrient_threshold) {
    death_cause = 1;
  } else if (tensile_stress_proxy > acute_stress_threshold) {
    death_cause = 2;
  } else if (agent_damage >= damage_death_threshold) {
    death_cause = 3;
  }

  if (death_cause >= 0) {
    FLAMEGPU->setVariable<int>("dead", 1);
    FLAMEGPU->setVariable<int>("dead_by", death_cause);
    FLAMEGPU->setVariable<int>("just_divided", 0);
    FLAMEGPU->setVariable<int>("daughter_id", -1);
    FLAMEGPU->setVariable<float>("vx", 0.0f);
    FLAMEGPU->setVariable<float>("vy", 0.0f);
    FLAMEGPU->setVariable<float>("vz", 0.0f);
    if (DEAD_CELLS_DISAPPEAR != 0) {
      FLAMEGPU->setVariable<int>("marked_for_removal", 1);
      return flamegpu::ALIVE;
    }
    FLAMEGPU->setVariable<int>("marked_for_removal", 0);
    return flamegpu::ALIVE;
  }
  FLAMEGPU->setVariable<int>("marked_for_removal", 0);

  float agent_clock = FLAMEGPU->getVariable<float>("clock");
  agent_clock += TIME_STEP;
  FLAMEGPU->setVariable<float>("clock", agent_clock);

  if ((agent_clock >= CYCLE_PHASE_G1_START) && (agent_clock < CYCLE_PHASE_S_START)) {
    FLAMEGPU->setVariable<int>("cycle_phase", 1);
  }
  if ((agent_clock >= CYCLE_PHASE_S_START) && (agent_clock < CYCLE_PHASE_G2_START)) {
    FLAMEGPU->setVariable<int>("cycle_phase", 2);
  }
  if ((agent_clock >= CYCLE_PHASE_G2_START) && (agent_clock < CYCLE_PHASE_M_START)) {
    FLAMEGPU->setVariable<int>("cycle_phase", 3);
  }

  // Increasing probability of division with time in M phase
  if (agent_clock >= CYCLE_PHASE_M_START) {
    float time_in_phase   = agent_clock - CYCLE_PHASE_M_START;
    float phase_n_steps   = CYCLE_PHASE_M_DURATION / TIME_STEP;
    float p_step          = 1 / phase_n_steps;
    float current_phase_step = time_in_phase / TIME_STEP;
    float p_division = p_step / ((phase_n_steps - current_phase_step + 1) / phase_n_steps);
    p_division *= division_rate_multiplier;
    p_division = fminf(1.0f, fmaxf(0.0f, p_division));
    float p = FLAMEGPU->random.uniform<float>(0.0, 1.0);
    FLAMEGPU->setVariable<int>("cycle_phase", 4);
    if (agent_clock > CELL_CYCLE_DURATION) {
      agent_clock -= CELL_CYCLE_DURATION;
      FLAMEGPU->setVariable<float>("clock", agent_clock);
    }

    if (p < p_division) {
      // Division occurs
      const float old_agent_x = agent_x;
      const float old_agent_y = agent_y;
      const float old_agent_z = agent_z;
      const float parent_new_x = old_agent_x + (agent_orx * CELL_RADIUS / 2);
      const float parent_new_y = old_agent_y + (agent_ory * CELL_RADIUS / 2);
      const float parent_new_z = old_agent_z + (agent_orz * CELL_RADIUS / 2);
      const float daughter_x = old_agent_x - (agent_orx * CELL_RADIUS / 2);
      const float daughter_y = old_agent_y - (agent_ory * CELL_RADIUS / 2);
      const float daughter_z = old_agent_z - (agent_orz * CELL_RADIUS / 2);

      FLAMEGPU->setVariable<float>("x", parent_new_x);
      FLAMEGPU->setVariable<float>("y", parent_new_y);
      FLAMEGPU->setVariable<float>("z", parent_new_z);
      FLAMEGPU->setVariable<float>("vx", 0.0f);
      FLAMEGPU->setVariable<float>("vy", 0.0f);
      FLAMEGPU->setVariable<float>("vz", 0.0f);
      FLAMEGPU->setVariable<float>("trajectory_length", 0.0f);
      FLAMEGPU->setVariable<float>("trajectory_time", 0.0f);
      FLAMEGPU->setVariable<float>("birth_x", parent_new_x);
      FLAMEGPU->setVariable<float>("birth_y", parent_new_y);
      FLAMEGPU->setVariable<float>("birth_z", parent_new_z);
      FLAMEGPU->setVariable<float>("radius", CELL_RADIUS / 2);
      FLAMEGPU->setVariable<float>("nucleus_radius", CELL_NUCLEUS_RADIUS / 2);
      FLAMEGPU->setVariable<float>("eps_xx", 0.0f);
      FLAMEGPU->setVariable<float>("eps_yy", 0.0f);
      FLAMEGPU->setVariable<float>("eps_zz", 0.0f);
      FLAMEGPU->setVariable<float>("eps_xy", 0.0f);
      FLAMEGPU->setVariable<float>("eps_xz", 0.0f);
      FLAMEGPU->setVariable<float>("eps_yz", 0.0f);
      FLAMEGPU->setVariable<float>("sig_xx", 0.0f);
      FLAMEGPU->setVariable<float>("sig_yy", 0.0f);
      FLAMEGPU->setVariable<float>("sig_zz", 0.0f);
      FLAMEGPU->setVariable<float>("sig_xy", 0.0f);
      FLAMEGPU->setVariable<float>("sig_xz", 0.0f);
      FLAMEGPU->setVariable<float>("sig_yz", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eig_1", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eig_2", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eig_3", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec1_x", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec1_y", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec1_z", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec2_x", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec2_y", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec2_z", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec3_x", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec3_y", 0.0f);
      FLAMEGPU->setVariable<float>("sig_eigvec3_z", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eig_1", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eig_2", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eig_3", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec1_x", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec1_y", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec1_z", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec2_x", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec2_y", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec2_z", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec3_x", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec3_y", 0.0f);
      FLAMEGPU->setVariable<float>("eps_eigvec3_z", 0.0f);

      const float damage_share = 0.5f * agent_damage;
      FLAMEGPU->setVariable<float>("damage", damage_share);

      for (int i = 0; i < N_SPECIES; i++) {
        const float parent_daughter_mass = 0.5f * agent_M_sp[i];
        FLAMEGPU->setVariable<float, N_SPECIES>("M_sp", i, parent_daughter_mass);
        FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, agent_C_sp[i]);
      }

      const float new_nucleus_radius = CELL_NUCLEUS_RADIUS / 2;
      for (int i = 0; i < N_ANCHOR_POINTS; i++) {
        FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("x_i", i, parent_new_x + new_nucleus_radius * agent_u_ref_x_i[i]);
        FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("y_i", i, parent_new_y + new_nucleus_radius * agent_u_ref_y_i[i]);
        FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("z_i", i, parent_new_z + new_nucleus_radius * agent_u_ref_z_i[i]);
      }

      // -----------------------------------------------------------------
      // RG variant: parent RG-state updates after division
      // -----------------------------------------------------------------
      if (agent_cell_type == 2) {
        // RG asymmetric division (Notch-Delta mechanism):
        // Parent retains primary cilium → high Notch → stays RG.
        // Commitment and epithelialization are slightly diluted by the division event.
        FLAMEGPU->setVariable<float>("rg_commit_level",         rg_commit * 0.95f);
        FLAMEGPU->setVariable<float>("epithelialization_level", epi * 0.85f);
        FLAMEGPU->setVariable<float>("apx", apx);
        FLAMEGPU->setVariable<float>("apy", apy);
        FLAMEGPU->setVariable<float>("apz", apz);
      } else {
        // iPSC / NEP: symmetric division — parent inherits own state unchanged.
        FLAMEGPU->setVariable<float>("rg_commit_level",         rg_commit);
        FLAMEGPU->setVariable<float>("epithelialization_level", epi);
        FLAMEGPU->setVariable<float>("apx", apx);
        FLAMEGPU->setVariable<float>("apy", apy);
        FLAMEGPU->setVariable<float>("apz", apz);
      }
      // rosette_maturity is recomputed each step by cell_rg_differentiation;
      // reset to zero after division so the parent does not carry a stale value.
      FLAMEGPU->setVariable<float>("rosette_maturity", 0.0f);

      agent_completed_cycles += 1;
      FLAMEGPU->setVariable<int>("completed_cycles", agent_completed_cycles);
      FLAMEGPU->setVariable<float>("clock", 0.0 + FLAMEGPU->random.uniform<float>(0.0, 0.1) * CYCLE_PHASE_G1_DURATION);
      FLAMEGPU->setVariable<int>("cycle_phase", 1);

      // New cell agent
      float rand_dir_x = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
      float rand_dir_y = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
      float rand_dir_z = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
      float rand_dir_length = vec3Length(rand_dir_x, rand_dir_y, rand_dir_z);
      if (rand_dir_length < 1e-6f) {
        rand_dir_x = 1.0f;
        rand_dir_y = 0.0f;
        rand_dir_z = 0.0f;
      } else {
        vec3Div(rand_dir_x, rand_dir_y, rand_dir_z, rand_dir_length);
      }
      const int daughter_cell_id = MACRO_MAX_GLOBAL_CELL_ID.addAtomic(1);
      FLAMEGPU->setVariable<int>("dead", 0);
      FLAMEGPU->setVariable<int>("dead_by", -1);
      FLAMEGPU->setVariable<int>("mother_id", -1);
      FLAMEGPU->setVariable<int>("just_divided", 1);
      FLAMEGPU->setVariable<int>("daughter_id", daughter_cell_id);
      FLAMEGPU->setVariable<int>("marked_for_removal", 0);

      // Daughter cell: same properties as parent; RG-specific state set below.
      FLAMEGPU->agent_out.setVariable<int>("id", daughter_cell_id);
      FLAMEGPU->agent_out.setVariable<int>("max_global_cell_id", daughter_cell_id);
      FLAMEGPU->agent_out.setVariable<float>("x", daughter_x);
      FLAMEGPU->agent_out.setVariable<float>("y", daughter_y);
      FLAMEGPU->agent_out.setVariable<float>("z", daughter_z);
      FLAMEGPU->agent_out.setVariable<float>("vx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("vy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("vz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("trajectory_length", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("trajectory_time", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("birth_x", daughter_x);
      FLAMEGPU->agent_out.setVariable<float>("birth_y", daughter_y);
      FLAMEGPU->agent_out.setVariable<float>("birth_z", daughter_z);
      FLAMEGPU->agent_out.setVariable<float>("orx", rand_dir_x);
      FLAMEGPU->agent_out.setVariable<float>("ory", rand_dir_y);
      FLAMEGPU->agent_out.setVariable<float>("orz", rand_dir_z);
      FLAMEGPU->agent_out.setVariable<float>("k_elast", agent_k_elast);
      FLAMEGPU->agent_out.setVariable<float>("d_dumping", agent_d_dumping);
      FLAMEGPU->agent_out.setVariable<float>("alignment", agent_alignment);
      for (int i = 0; i < N_SPECIES; i++) {
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_consumption", i, agent_k_consumption[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_production",  i, agent_k_production[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_reaction",    i, agent_k_reaction[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("C_sp",          i, agent_C_sp[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("M_sp",          i, 0.5f * agent_M_sp[i]);
      }
      FLAMEGPU->agent_out.setVariable<float>("speed_ref", agent_speed_ref);
      FLAMEGPU->agent_out.setVariable<float>("radius", CELL_RADIUS / 2);
      FLAMEGPU->agent_out.setVariable<float>("nucleus_radius", CELL_NUCLEUS_RADIUS / 2);
      FLAMEGPU->agent_out.setVariable<float>("cc_dvx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_dvy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_dvz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_dvx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_dvy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_dvz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_S_xx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_S_yy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_S_zz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_S_xy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_S_xz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_S_yz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_S_xx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_S_yy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_S_zz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_S_xy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_S_xz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_S_yz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cl_S_xx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cl_S_yy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cl_S_zz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cl_S_xy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cl_S_xz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cl_S_yz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("focad_S_xx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("focad_S_yy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("focad_S_zz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("focad_S_xy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("focad_S_xz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("focad_S_yz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("clock", 0.0 + FLAMEGPU->random.uniform<float>(0.0, 0.1) * CYCLE_PHASE_G1_DURATION);
      FLAMEGPU->agent_out.setVariable<int>("cycle_phase", 1);
      FLAMEGPU->agent_out.setVariable<int>("completed_cycles", 0);
      FLAMEGPU->agent_out.setVariable<int>("dead", 0);
      FLAMEGPU->agent_out.setVariable<int>("dead_by", -1);
      FLAMEGPU->agent_out.setVariable<int>("mother_id", id);
      FLAMEGPU->agent_out.setVariable<int>("just_divided", 0);
      FLAMEGPU->agent_out.setVariable<int>("daughter_id", -1);
      FLAMEGPU->agent_out.setVariable<int>("marked_for_removal", 0);
      FLAMEGPU->agent_out.setVariable<float>("focad_birth_cooldown", fmaxf(0.0f, agent_focad_birth_cooldown));
      FLAMEGPU->agent_out.setVariable<float>("damage", damage_share);

      const float daughter_nucleus_radius = CELL_NUCLEUS_RADIUS / 2;
      for (int i = 0; i < N_ANCHOR_POINTS; i++) {
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("x_i",       i, daughter_x + daughter_nucleus_radius * agent_u_ref_x_i[i]);
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("y_i",       i, daughter_y + daughter_nucleus_radius * agent_u_ref_y_i[i]);
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("z_i",       i, daughter_z + daughter_nucleus_radius * agent_u_ref_z_i[i]);
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("u_ref_x_i", i, agent_u_ref_x_i[i]);
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("u_ref_y_i", i, agent_u_ref_y_i[i]);
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("u_ref_z_i", i, agent_u_ref_z_i[i]);
      }

      FLAMEGPU->agent_out.setVariable<float>("eps_xx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_yy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_zz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_xy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_xz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_yz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_xx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_yy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_zz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_xy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_xz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_yz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eig_1", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eig_2", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eig_3", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec1_x", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec1_y", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec1_z", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec2_x", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec2_y", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec2_z", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec3_x", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec3_y", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("sig_eigvec3_z", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eig_1", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eig_2", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eig_3", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec1_x", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec1_y", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec1_z", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec2_x", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec2_y", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec2_z", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec3_x", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec3_y", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("eps_eigvec3_z", 0.0f);

      // -----------------------------------------------------------------
      // RG variant: daughter RG-state and type assignment
      // -----------------------------------------------------------------
      if (agent_cell_type == 2) {
        // RG asymmetric division: daughter inherits Delta excess → re-enters
        // as NEP (type 1) with commitment just above the iPSC→NEP threshold.
        const float daughter_commit = RG_COMMIT_THRESHOLD_NEP + 0.05f;
        const float theta = FLAMEGPU->random.uniform<float>(0.0f, 6.28318530718f);
        FLAMEGPU->agent_out.setVariable<int>("cell_type", 1);           // NEP
        FLAMEGPU->agent_out.setVariable<float>("rg_commit_level",         daughter_commit);
        FLAMEGPU->agent_out.setVariable<float>("epithelialization_level", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("apx", cosf(theta));
        FLAMEGPU->agent_out.setVariable<float>("apy", sinf(theta));
        FLAMEGPU->agent_out.setVariable<float>("apz", 0.0f);
        // Sync k_production[2] for NEP: base rate × NEP multiplier
        const float base_prod = FLAMEGPU->environment.getProperty<float, N_SPECIES>("INIT_CELL_PRODUCTION_RATES", 2);
        const float mult_nep  = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>("CELL_PRODUCTION_MULTIPLIER", 1);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_production", 2, base_prod * mult_nep);
      } else {
        // iPSC / NEP symmetric division: daughter inherits parent state.
        FLAMEGPU->agent_out.setVariable<int>("cell_type", agent_cell_type);
        FLAMEGPU->agent_out.setVariable<float>("rg_commit_level",         rg_commit);
        FLAMEGPU->agent_out.setVariable<float>("epithelialization_level", epi);
        FLAMEGPU->agent_out.setVariable<float>("apx", apx);
        FLAMEGPU->agent_out.setVariable<float>("apy", apy);
        FLAMEGPU->agent_out.setVariable<float>("apz", apz);
      }
      // Common daughter defaults: recomputed-each-step readout variables
      FLAMEGPU->agent_out.setVariable<float>("rosette_maturity",     0.0f);
      FLAMEGPU->agent_out.setVariable<float>("rg_neighbour_density", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("morphogen_local",      0.0f);
      FLAMEGPU->agent_out.setVariable<float>("substrate_anchor_x",  daughter_x);
      FLAMEGPU->agent_out.setVariable<float>("substrate_anchor_y",  daughter_y);

      agent_vx = 0.0f;
      agent_vy = 0.0f;
      agent_vz = 0.0f;
    }
  } else {
    agent_radius += ((CELL_RADIUS / 2) / CYCLE_PHASE_M_START) * TIME_STEP;
    agent_nucleus_radius += ((CELL_NUCLEUS_RADIUS / 2) / CYCLE_PHASE_M_START) * TIME_STEP;
    FLAMEGPU->setVariable<float>("radius", fminf(agent_radius, CELL_RADIUS));
    FLAMEGPU->setVariable<float>("nucleus_radius", fminf(agent_nucleus_radius, CELL_NUCLEUS_RADIUS));
  }

  // Recompute anchor positions from u_ref at current nucleus_radius.
  agent_nucleus_radius = FLAMEGPU->getVariable<float>("nucleus_radius");
  agent_x = FLAMEGPU->getVariable<float>("x");
  agent_y = FLAMEGPU->getVariable<float>("y");
  agent_z = FLAMEGPU->getVariable<float>("z");
  for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("x_i", i, agent_x + agent_nucleus_radius * agent_u_ref_x_i[i]);
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("y_i", i, agent_y + agent_nucleus_radius * agent_u_ref_y_i[i]);
    FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("z_i", i, agent_z + agent_nucleus_radius * agent_u_ref_z_i[i]);
  }

  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);
  return flamegpu::ALIVE;
}
