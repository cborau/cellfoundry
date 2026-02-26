/**
 * vec3Div
 *
 * Purpose:
 *   Divide a 3D vector (x, y, z) by a scalar divisor in-place.
 *
 * Inputs:
 *   - x, y, z: vector components (modified)
 *   - divisor: scalar value
 *
 * Outputs:
 *   - x, y, z: scaled vector components
 */
FLAMEGPU_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
  x /= divisor;
  y /= divisor;
  z /= divisor;
}
/**
 * vec3Length
 *
 * Purpose:
 *   Compute the Euclidean length of a 3D vector (x, y, z).
 *
 * Inputs:
 *   - x, y, z: vector components
 *
 * Outputs:
 *   - Returns the magnitude of the vector
 */
FLAMEGPU_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}
/**
 * cell_cycle
 *
 * Purpose:
 *   Agent function for cell cycle progression, division, and death.
 *   Handles cell phase transitions, damage accumulation, and division logic.
 *
 * Inputs:
 *   - CELL agent variables: id, cell_type, position, orientation, metabolic rates, anchor arrays, etc.
 *   - Environment properties: cell cycle timings, thresholds, rates
 *
 * Outputs:
 *   - Updated cell state variables (phase, clock, damage, division, daughter creation)
 *
 * Notes:
 *   - Dead cells may remain ALIVE for agent purposes if DEAD_CELLS_DISAPPEAR is set.
 *   - Division logic includes randomization and mass/anchor inheritance.
 */
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
    return flamegpu::ALIVE; // Note: if DEAD_CELLS_DISAPPEAR = True, a dead CELL agent remains ALIVE for flamegpu purposes and may still interact with other agents.
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

  const uint8_t N_SPECIES = 2;       // WARNING: must match main python model
  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: must match main python model

  float agent_k_consumption[N_SPECIES] = {};
  float agent_k_production[N_SPECIES] = {};
  float agent_k_reaction[N_SPECIES] = {};
  float agent_C_sp[N_SPECIES] = {};
  float agent_M_sp[N_SPECIES] = {};
  float agent_chemotaxis_sensitivity[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    agent_k_consumption[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_consumption", i);
    agent_k_production[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_production", i);
    agent_k_reaction[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_reaction", i);
    agent_C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
    agent_M_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("M_sp", i);
    agent_chemotaxis_sensitivity[i] = FLAMEGPU->getVariable<float, N_SPECIES>("chemotaxis_sensitivity", i);
  }

  float agent_x_i[N_ANCHOR_POINTS] = {};
  float agent_y_i[N_ANCHOR_POINTS] = {};
  float agent_z_i[N_ANCHOR_POINTS] = {};
  float agent_u_ref_x_i[N_ANCHOR_POINTS] = {};
  float agent_u_ref_y_i[N_ANCHOR_POINTS] = {};
  float agent_u_ref_z_i[N_ANCHOR_POINTS] = {};
  for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    agent_x_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("x_i", i);
    agent_y_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("y_i", i);
    agent_z_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("z_i", i);
    agent_u_ref_x_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_x_i", i);
    agent_u_ref_y_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_y_i", i);
    agent_u_ref_z_i[i] = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("u_ref_z_i", i);
  }
  
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float CELL_RADIUS = FLAMEGPU->environment.getProperty<float>("CELL_RADIUS");
  const float CELL_NUCLEUS_RADIUS = FLAMEGPU->environment.getProperty<float>("CELL_NUCLEUS_RADIUS");
  const float CELL_CYCLE_DURATION = FLAMEGPU->environment.getProperty<float>("CELL_CYCLE_DURATION");
  const float CYCLE_PHASE_G1_START = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_G1_START");
  const float CYCLE_PHASE_S_START = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_S_START");
  const float CYCLE_PHASE_G2_START = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_G2_START");
  const float CYCLE_PHASE_M_START = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_M_START");
  const float CYCLE_PHASE_G1_DURATION = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_G1_DURATION");
  const float CYCLE_PHASE_M_DURATION = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_M_DURATION");
  const float hypoxia_threshold = FLAMEGPU->environment.getProperty<float>("CELL_HYPOXIA_THRESHOLD");
  const float nutrient_threshold = FLAMEGPU->environment.getProperty<float>("CELL_NUTRIENT_THRESHOLD");
  const float stress_threshold = FLAMEGPU->environment.getProperty<float>("CELL_STRESS_THRESHOLD");
  const float hypoxia_damage_rate = FLAMEGPU->environment.getProperty<float>("CELL_HYPOXIA_DAMAGE_RATE");
  const float nutrient_damage_rate = FLAMEGPU->environment.getProperty<float>("CELL_NUTRIENT_DAMAGE_RATE");
  const float stress_damage_rate = FLAMEGPU->environment.getProperty<float>("CELL_STRESS_DAMAGE_RATE");
  const float basal_damage_repair_rate = FLAMEGPU->environment.getProperty<float>("CELL_BASAL_DAMAGE_REPAIR_RATE");
  const float acute_hypoxia_threshold = FLAMEGPU->environment.getProperty<float>("CELL_ACUTE_HYPOXIA_THRESHOLD");
  const float acute_nutrient_threshold = FLAMEGPU->environment.getProperty<float>("CELL_ACUTE_NUTRIENT_THRESHOLD");
  const float acute_stress_threshold = FLAMEGPU->environment.getProperty<float>("CELL_ACUTE_STRESS_THRESHOLD");

  // Proxies used for death pathways (can be remapped by user model semantics)
  const float oxygen_proxy = agent_C_sp[0];
  const float nutrient_proxy = agent_C_sp[1];
  const float tensile_stress_proxy = fmaxf(0.0f, agent_sig_l1);

  float division_rate_multiplier = 1.0f;
  float damage_accumulation_multiplier = 1.0f;
  float damage_repair_multiplier = 1.0f;
  float damage_death_threshold = 1.0f;

  // User-defined behaviour for cell_type = 0. Add more types and logic as needed.
  if (agent_cell_type == 0) {
    division_rate_multiplier = 1.00f;
    damage_accumulation_multiplier = 1.00f;
    damage_repair_multiplier = 1.00f;
    damage_death_threshold = 1.00f;
  }
  // User-defined behaviour for cell_type = 1. Add more types and logic as needed.
  else if (agent_cell_type == 1) {
    division_rate_multiplier = 1.15f;
    damage_accumulation_multiplier = 0.85f;
    damage_repair_multiplier = 1.10f;
    damage_death_threshold = 1.0f;
  }
  // User-defined behaviour for cell_type = 2. Add more types and logic as needed.
  else if (agent_cell_type == 2) {
    division_rate_multiplier = 0.85f;
    damage_accumulation_multiplier = 1.25f;
    damage_repair_multiplier = 0.85f;
    damage_death_threshold = 0.80f;
  }

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
      return flamegpu::ALIVE; // Note: if DEAD_CELLS_DISAPPEAR = True, a dead CELL agent remains ALIVE for flamegpu purposes and may still interact with other agents.
    }
    FLAMEGPU->setVariable<int>("marked_for_removal", 0);
    return flamegpu::ALIVE; // Note: if DEAD_CELLS_DISAPPEAR = True, a dead CELL agent remains ALIVE for flamegpu purposes and may still interact with other agents.
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
  printf("Cell [id: %d] at t: %g - cycle phase: %d, max cell id: %d \n", id, agent_clock, FLAMEGPU->getVariable<int>("cycle_phase"), agent_max_global_cell_id);
  // Increasing probability of division with time in M phase
  if (agent_clock >= CYCLE_PHASE_M_START) {
    float time_in_phase = agent_clock - CYCLE_PHASE_M_START;
    float phase_n_steps = CYCLE_PHASE_M_DURATION / TIME_STEP; 
    float p_step = 1 / phase_n_steps;
    float current_phase_step = time_in_phase / TIME_STEP;
    float p_division = p_step / ((phase_n_steps - current_phase_step + 1) / phase_n_steps); //actual probability in current step.
    p_division *= division_rate_multiplier;
    p_division = fminf(1.0f, fmaxf(0.0f, p_division));
    float p = FLAMEGPU->random.uniform<float>(0.0,1.0);
    FLAMEGPU->setVariable<int>("cycle_phase", 4);
    if (agent_clock > CELL_CYCLE_DURATION) { // this should never happen as the cell should divide first
      agent_clock -= CELL_CYCLE_DURATION;      
      FLAMEGPU->setVariable<float>("clock", agent_clock);  
    }
    printf("Cell [id: %d] in M phase at t: %g [time in phase: %g]- p_division: %g, random draw: %g \n", id, agent_clock, time_in_phase, p_division, p);
    if (p < p_division) {
      printf("Cell [id: %d] DIVISION OCCURS at t: %g [time in phase: %g]- p_division: %g, random draw: %g \n", id, agent_clock, time_in_phase, p_division, p);
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
      FLAMEGPU->setVariable<float>("radius", CELL_RADIUS / 2); // to prevent diminishing radius over multiple divisions.
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

      const float shift_parent_x = parent_new_x - old_agent_x;
      const float shift_parent_y = parent_new_y - old_agent_y;
      const float shift_parent_z = parent_new_z - old_agent_z;
      for (int i = 0; i < N_ANCHOR_POINTS; i++) {
        FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("x_i", i, agent_x_i[i] + shift_parent_x);
        FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("y_i", i, agent_y_i[i] + shift_parent_y);
        FLAMEGPU->setVariable<float, N_ANCHOR_POINTS>("z_i", i, agent_z_i[i] + shift_parent_z);
      }

      agent_completed_cycles += 1;
      FLAMEGPU->setVariable<int>("completed_cycles", agent_completed_cycles);
      printf("Cell [id: %d] DIVIDES at t: %g [time in phase: %g]- completed cycles: %d \n", id, agent_clock, time_in_phase, agent_completed_cycles);
      printf("Cell [id: %d] PROBS p_step: %g, phase_n_steps:%g, current_phase_step: %g, p_division: %g \n", id, p_step, phase_n_steps, current_phase_step, p_division);
      FLAMEGPU->setVariable<float>("clock", 0.0 + FLAMEGPU->random.uniform<float>(0.0,0.1) * CYCLE_PHASE_G1_DURATION); //add some randomness to the clock
      FLAMEGPU->setVariable<int>("cycle_phase", 1);  
      
      
      // New cell agent
      float rand_dir_x = FLAMEGPU->random.uniform<float>(-1.0,1.0);
      float rand_dir_y = FLAMEGPU->random.uniform<float>(-1.0,1.0);
      float rand_dir_z = FLAMEGPU->random.uniform<float>(-1.0,1.0); 
      float rand_dir_length = vec3Length(rand_dir_x,rand_dir_y,rand_dir_z);
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
      // Daugther cell is created with the same properties as the parent, but with a new unique id, position offset in the opposite direction to the parent, and some damage inherited from the parent.
      FLAMEGPU->agent_out.setVariable<int>("id", daughter_cell_id);
      FLAMEGPU->agent_out.setVariable<int>("max_global_cell_id", daughter_cell_id);
      FLAMEGPU->agent_out.setVariable<float>("x", daughter_x);
      FLAMEGPU->agent_out.setVariable<float>("y", daughter_y);
      FLAMEGPU->agent_out.setVariable<float>("z", daughter_z);
      FLAMEGPU->agent_out.setVariable<float>("vx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("vy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("vz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("orx", rand_dir_x);
      FLAMEGPU->agent_out.setVariable<float>("ory", rand_dir_y);
      FLAMEGPU->agent_out.setVariable<float>("orz", rand_dir_z);
      FLAMEGPU->agent_out.setVariable<float>("k_elast", agent_k_elast);
      FLAMEGPU->agent_out.setVariable<float>("d_dumping", agent_d_dumping);
      FLAMEGPU->agent_out.setVariable<float>("alignment", agent_alignment);
      for (int i = 0; i < N_SPECIES; i++) {
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_consumption", i, agent_k_consumption[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_production", i, agent_k_production[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("k_reaction", i, agent_k_reaction[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("C_sp", i, agent_C_sp[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("M_sp", i, 0.5f * agent_M_sp[i]);
        FLAMEGPU->agent_out.setVariable<float, N_SPECIES>("chemotaxis_sensitivity", i, agent_chemotaxis_sensitivity[i]);
      }
      FLAMEGPU->agent_out.setVariable<float>("speed_ref", agent_speed_ref);
      FLAMEGPU->agent_out.setVariable<float>("radius", CELL_RADIUS / 2); // to prevent diminishing cell size with each division, set radius to the agent base variable and not the current parent's value.
      FLAMEGPU->agent_out.setVariable<float>("nucleus_radius", CELL_NUCLEUS_RADIUS / 2);
      FLAMEGPU->agent_out.setVariable<float>("cc_dvx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_dvy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cc_dvz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_dvx", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_dvy", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("cf_dvz", 0.0f);
      FLAMEGPU->agent_out.setVariable<float>("clock", 0.0 + FLAMEGPU->random.uniform<float>(0.0,0.1) * CYCLE_PHASE_G1_DURATION);
      FLAMEGPU->agent_out.setVariable<int>("cycle_phase", 1);
      FLAMEGPU->agent_out.setVariable<int>("cell_type", agent_cell_type);
      FLAMEGPU->agent_out.setVariable<int>("completed_cycles", 0);
      FLAMEGPU->agent_out.setVariable<int>("dead", 0);
      FLAMEGPU->agent_out.setVariable<int>("dead_by", -1);
      FLAMEGPU->agent_out.setVariable<int>("mother_id", id);
      FLAMEGPU->agent_out.setVariable<int>("just_divided", 0);
      FLAMEGPU->agent_out.setVariable<int>("daughter_id", -1);
      FLAMEGPU->agent_out.setVariable<int>("marked_for_removal", 0);
      FLAMEGPU->agent_out.setVariable<float>("focad_birth_cooldown", fmaxf(0.0f, agent_focad_birth_cooldown));
      FLAMEGPU->agent_out.setVariable<float>("damage", damage_share);

      const float shift_daughter_x = daughter_x - old_agent_x;
      const float shift_daughter_y = daughter_y - old_agent_y;
      const float shift_daughter_z = daughter_z - old_agent_z;
      for (int i = 0; i < N_ANCHOR_POINTS; i++) {
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("x_i", i, agent_x_i[i] + shift_daughter_x);
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("y_i", i, agent_y_i[i] + shift_daughter_y);
        FLAMEGPU->agent_out.setVariable<float, N_ANCHOR_POINTS>("z_i", i, agent_z_i[i] + shift_daughter_z);
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
  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);
  return flamegpu::ALIVE;  
}