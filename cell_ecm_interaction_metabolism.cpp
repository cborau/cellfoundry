// defines interactions with ECM agents and computes metabolism of species
FLAMEGPU_AGENT_FUNCTION(cell_ecm_interaction_metabolism, flamegpu::MessageArray3D, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");

  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  
  // Agent array variables
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  const uint32_t ECM_POPULATION_SIZE = 1000; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");
  
  float k_consumption[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    k_consumption[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_consumption", i);
  }
  // Agent array variables
  float k_production[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    k_production[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_production", i);
  }
  // Agent array variables
  float C_sp[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
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
  float message_C_sp[N_SPECIES] = {};

  // Reads the closest ECM agent grid_lin_id to read the corresponding C_SP_MACRO value
  // Then computes metabolism, updating both the cell calling agent and C_SP_MACRO values accordingly
  // The closest ECM agent
  const auto message = FLAMEGPU->message_in.at(agent_grid_i, agent_grid_j, agent_grid_k);
  message_id = message.getVariable<int>("id");
  message_grid_lin_id = message.getVariable<int>("grid_lin_id");
  for (int i = 0; i < N_SPECIES; i++) {
    message_C_sp[i] = (float)C_SP_MACRO[i][message_grid_lin_id]; // read concentration of species from the MACRO variable
    // compute metabolism
    float delta_C = (-k_consumption[i] + k_production[i]) * TIME_STEP; // TODO: CHECK EQUATION
    // Update ECM MACRO variable -> THIS RAISES AN ERROR DUE TO RACING CONDITIONS
    //C_SP_MACRO[i][message_grid_lin_id] += delta_C;
    C_sp[i] += delta_C; // update cell species concentration
    FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, C_sp[i]);
  }



  return flamegpu::ALIVE;
}