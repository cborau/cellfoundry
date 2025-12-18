// defines interactions with ECM agents and computes metabolism of species
FLAMEGPU_AGENT_FUNCTION(cell_ecm_interaction_metabolism, flamegpu::MessageArray3D, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");

  // Agent array variables
  const uint8_t k_consumption_ARRAY_SIZE = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float k_consumption[k_consumption_ARRAY_SIZE] = {};
  for (int i = 0; i < k_consumption_ARRAY_SIZE; i++) {
    k_consumption[i] = FLAMEGPU->getVariable<float, k_consumption_ARRAY_SIZE>("k_consumption", i);
  }
  // Agent array variables
  const uint8_t k_production_ARRAY_SIZE = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float k_production[k_production_ARRAY_SIZE] = {};
  for (int i = 0; i < k_production_ARRAY_SIZE; i++) {
    k_production[i] = FLAMEGPU->getVariable<float, k_production_ARRAY_SIZE>("k_production", i);
  }
  // Agent array variables
  const uint8_t C_sp_ARRAY_SIZE = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float C_sp[C_sp_ARRAY_SIZE] = {};
  for (int i = 0; i < C_sp_ARRAY_SIZE; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, C_sp_ARRAY_SIZE>("C_sp", i);
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
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;
  uint8_t message_grid_i = 0;
  uint8_t message_grid_j = 0;
  uint8_t message_grid_k = 0;
  const uint8_t message_C_sp_ARRAY_SIZE = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float message_C_sp[message_C_sp_ARRAY_SIZE] = {};

  // TODO: reads the closest ECM agent grid_lin_id and computes metabolism, updating both the calling agnent and Macro C_sp values accordingly
  // The closest ECM agent
  const auto message = FLAMEGPU->message_in.at(agent_grid_i, agent_grid_j, agent_grid_k);
  message_id = message.getVariable<int>("id");
  message_x = message.getVariable<float>("x");
  message_y = message.getVariable<float>("y");
  message_z = message.getVariable<float>("z");
  message_grid_i = message.getVariable<uint8_t>("grid_i");
  message_grid_j = message.getVariable<uint8_t>("grid_j");
  message_grid_k = message.getVariable<uint8_t>("grid_k");
  for (int i = 0; i < message_C_sp_ARRAY_SIZE; i++) {
    message_C_sp[i] = message.getVariable<float, message_C_sp_ARRAY_SIZE>("C_sp", i);
  }


  //Set agent variables
  FLAMEGPU->setVariable<int>("id", agent_id);
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);

  for (int i = 0; i < k_consumption_ARRAY_SIZE; i++) {
    FLAMEGPU->setVariable<float, k_consumption_ARRAY_SIZE>("k_consumption", i, k_consumption[i]);
  }

  for (int i = 0; i < k_production_ARRAY_SIZE; i++) {
    FLAMEGPU->setVariable<float, k_production_ARRAY_SIZE>("k_production", i, k_production[i]);
  }

  for (int i = 0; i < C_sp_ARRAY_SIZE; i++) {
    FLAMEGPU->setVariable<float, C_sp_ARRAY_SIZE>("C_sp", i, C_sp[i]);
  }


  return flamegpu::ALIVE;
}