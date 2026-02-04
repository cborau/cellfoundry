// defines interactions with ECM agents and computes metabolism of species
FLAMEGPU_AGENT_FUNCTION(cell_ecm_interaction_metabolism, flamegpu::MessageArray3D, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_radius = FLAMEGPU->getVariable<float>("radius");
  float agent_volume = 4.0f / 3.0f * 3.1415926f * powf(agent_radius, 3); // sphere, in um^3

  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float ECM_VOXEL_VOLUME = FLAMEGPU->environment.getProperty<float>("ECM_VOXEL_VOLUME");
  
  // Agent array variables
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  const uint32_t ECM_POPULATION_SIZE = 9261; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");
  
  float k_consumption[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    k_consumption[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_consumption", i);
  }
  
  float k_production[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    k_production[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_production", i);
  }

  float k_reaction[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    k_reaction[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_reaction", i);
  }
  
  float C_sp[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
  }

  float M_sp[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    M_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("M_sp", i);
  }


    // Get number of agents per direction
  const int Nx = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",0);
  const int Ny = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",1);
  const int Nz = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",2);
  // Get position of the boundaries
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);
  
  // transform x,y,z positions to i,j,k grid positions
  int agent_grid_i = roundf(((agent_x - COORD_BOUNDARY_X_NEG) / (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG)) * (Nx - 1));
  int agent_grid_j = roundf(((agent_y - COORD_BOUNDARY_Y_NEG) / (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG)) * (Ny - 1));
  int agent_grid_k = roundf(((agent_z - COORD_BOUNDARY_Z_NEG) / (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG)) * (Nz - 1));

  //Define message variables (agent sending the input message)
  int message_id = 0;
  int message_grid_lin_id = 0;
  float C_sp_sat[N_SPECIES] = {};
  //printf("Cell agent %d at pos (%2.6f , %2.6f, %2.6f) reading ECM agent at grid (%d , %d, %d) \n", agent_id, agent_x, agent_y,  agent_z, agent_grid_i, agent_grid_j, agent_grid_k);

  // Reads the closest ECM agent grid_lin_id to read the corresponding C_SP_MACRO value
  // Then computes metabolism, updating both the cell calling agent and C_SP_MACRO values accordingly
  // The closest ECM agent
  const auto message = FLAMEGPU->message_in.at(agent_grid_i, agent_grid_j, agent_grid_k);
  message_id = message.getVariable<int>("id");
  message_grid_lin_id = message.getVariable<int>("grid_lin_id");
  
  for (int i = 0; i < N_SPECIES; i++) {

    // -------------------------------------------------------------------------
    // ECM ODE (PhysiCell-style), solved for ECM voxel concentration C_ecm:
    //
    //   dC_ecm/dt = ( k_production*(C_sp_sat - C_ecm) - k_consumption*C_ecm ) * alpha
    //
    // with:
    //   alpha = V_cell / V_voxel
    //   C_sp_sat = saturation concentration in the cell
    //
    // Expand:
    //   dC_ecm/dt = alpha * ( k_production*C_sp_sat - (k_production + k_consumption)*C_ecm )
    //
    // Backward Euler:
    //   (C^{n+1} - C^n)/dt = A - B*C^{n+1}
    //   A = alpha*k_production*C_sp_sat
    //   B = alpha*(k_production + k_consumption)
    //
    //   C_ecm^{n+1} = (C_ecm^n + dt*A) / (1 + dt*B)
    //
    // Using PhysiCell nomenclature:
    //   c1 = dt * alpha * k_production * C_sp_sat
    //   c2 = 1  + dt * alpha * (k_production + k_consumption)
    //   C_ecm^{n+1} = (C_ecm^n + c1) / c2
    //
    // Mass conservation via cell "amount" M_sp and voxel concentration C_ecm:
    //   M_voxel = C_ecm * V_voxel
    //
    // Proposed mass change:
    //   deltaM_voxel_prop = (C_ecm_prop - C_ecm_old) * V_voxel
    //
    // Clamp BOTH directions to prevent negative concentrations/amounts:
    //   - Uptake (deltaM_voxel_prop < 0): cannot remove more than M_voxel_old
    //   - Secretion (deltaM_voxel_prop > 0): cannot remove more than M_cell_old
    // -------------------------------------------------------------------------

    // ECM concentration at t^n (read-only message snapshot)
    const float C_ecm_old = message.getVariable<float, N_SPECIES>("C_sp", i);
    C_sp_sat[i] = message.getVariable<float, N_SPECIES>("C_sp_sat", i);
    // Cell amount at t^n (mass-like)
    const float M_cell_old = M_sp[i];

    // Volume coupling
    const float alpha = agent_volume / ECM_VOXEL_VOLUME;

    // PhysiCell-style coefficients
    const float c1 = TIME_STEP * alpha * k_production[i] * C_sp_sat[i];
    const float c2 = 1.0f + TIME_STEP * alpha * (k_production[i] + k_consumption[i]);

    // Backward-Euler proposed ECM concentration
    const float C_ecm_prop = (C_ecm_old + c1) / c2;

    // Convert to proposed voxel mass change
    const float M_voxel_old = C_ecm_old * ECM_VOXEL_VOLUME;
    const float M_voxel_prop = C_ecm_prop * ECM_VOXEL_VOLUME;
    const float deltaM_voxel_prop = M_voxel_prop - M_voxel_old;  // >0 secretion, <0 uptake

    // Clamp proposed transfer to keep both voxel concentration and cell amount non-negative
    float deltaM_voxel = deltaM_voxel_prop;
    
    if (deltaM_voxel_prop < 0.0f) {
      // Uptake: voxel loses mass, cell gains mass
      const float uptake = -deltaM_voxel_prop;
      const float uptake_clamped = fminf(uptake, M_voxel_old);
      deltaM_voxel = -uptake_clamped;
    } else if (deltaM_voxel_prop > 0.0f) {
      // Secretion: voxel gains mass, cell loses mass
      const float secretion = deltaM_voxel_prop;
      const float secretion_clamped = fminf(secretion, M_cell_old);
      deltaM_voxel = secretion_clamped;
    }

    //printf("  Species %d: alpha=%2.6f, C_ecm_old=%2.6f, C_ecm_prop=%2.6f, C_sp_sat=%2.6f, M_voxel_old=%2.6f, M_cell_old=%2.6f, deltaM_voxel_prop=%2.6f, deltaM_voxel=%2.6f \n", i, alpha, C_ecm_old, C_ecm_prop, C_sp_sat[i], M_voxel_old, M_cell_old, deltaM_voxel_prop, deltaM_voxel);

    // Apply clamped mass transfer (conservative)
    const float M_voxel_new = M_voxel_old + deltaM_voxel;
    const float M_cell_new  = M_cell_old  - deltaM_voxel;

    // Convert back to concentrations
    const float C_ecm_new = M_voxel_new / ECM_VOXEL_VOLUME;
    const float C_cell_new = M_cell_new / agent_volume;

    // Write ECM (atomic addition -> multiple cells may write to same voxel in parallel. Concentration cannot be negative due to clamping above)
    C_SP_MACRO[i][message_grid_lin_id].addAtomic(C_ecm_new - C_ecm_old);
    //C_SP_MACRO[i][message_grid_lin_id].exchange(C_ecm_new);
    // Store cell amount + concentration mirror
    M_sp[i] = M_cell_new;
    C_sp[i] = C_cell_new;
    //printf("    After clamp: C_ecm_new=%2.6f, C_cell_new=%2.6f, M_voxel_new=%2.6f, M_cell_new=%2.6f \n", C_ecm_new, C_cell_new, M_voxel_new, M_cell_new);
    //FLAMEGPU->setVariable<float, N_SPECIES>("M_sp", i, M_sp[i]);
    //FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, C_sp[i]);
  }
  //printf(" Before metabolic reaction: C_sp[0]=%2.6f, M_sp[0]=%2.6f, k_reaction[0]=%2.6f \n", C_sp[0], C_sp[0] * agent_volume, k_reaction[0]);
  //printf(" Before metabolic reaction: C_sp[1]=%2.6f, M_sp[1]=%2.6f, k_reaction[1]=%2.6f \n", C_sp[1], C_sp[1] * agent_volume, k_reaction[1]);
  // Compute metabolic reaction within the cell
  C_sp[0] -= TIME_STEP * k_reaction[0] * C_sp[0]; // species 0 is consumed
  C_sp[1] += TIME_STEP * k_reaction[1] * C_sp[0]; // species 1 is produced accordingly

  for (int i = 0; i < N_SPECIES; i++) {
    //printf("After metabolic reaction: C_cell[%d]=%2.6f, M_cell=%2.6f \n", i, C_sp[i], C_sp[i] * agent_volume);
    FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, C_sp[i]);
    FLAMEGPU->setVariable<float, N_SPECIES>("M_sp", i, C_sp[i] * agent_volume);
  }

  return flamegpu::ALIVE;
}