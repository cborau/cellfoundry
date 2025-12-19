// interacts with neighbour ECM to compute the diffusion of species
FLAMEGPU_AGENT_FUNCTION(ecm_ecm_interaction, flamegpu::MessageArray3D, flamegpu::MessageNone) {
  // TODO: implement diffusion between ECM agents
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  uint8_t agent_grid_i = FLAMEGPU->getVariable<uint8_t>("grid_i");
  uint8_t agent_grid_j = FLAMEGPU->getVariable<uint8_t>("grid_j");
  uint8_t agent_grid_k = FLAMEGPU->getVariable<uint8_t>("grid_k");
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");


  // Agent array variables
  int INCLUDE_DIFFUSION = FLAMEGPU->environment.getProperty<int>("INCLUDE_DIFFUSION");
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float C_sp[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
  }


  // Mechanical constants of the ecm 
  float k_elast = 0.0; //Equivalent elastic constant of two springs in series (agent and message)
  const float BUCKLING_COEFF_D0 = FLAMEGPU->environment.getProperty<float>("BUCKLING_COEFF_D0");
  const float STRAIN_STIFFENING_COEFF_DS = FLAMEGPU->environment.getProperty<float>("STRAIN_STIFFENING_COEFF_DS");
  const float CRITICAL_STRAIN = FLAMEGPU->environment.getProperty<float>("CRITICAL_STRAIN");
  float agent_k_elast = FLAMEGPU->getVariable<float>("k_elast");
  float agent_d_dumping = FLAMEGPU->getVariable<float>("d_dumping");

  //Define message variables (agent sending the input message)
  // WARNING: this function is not currently wired to any message source
  // TODO: initialise message variables as needed

  //Loop through all agents sending input messages
  for (const auto &message : FLAMEGPU->message_in(/* TODO: provide grid coordinates */)) {
    // WARNING: this function is not currently wired to any message source
    // TODO: process incoming message data
  }

  //Set agent variables
  FLAMEGPU->setVariable<int>("id", agent_id);
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);
  FLAMEGPU->setVariable<uint8_t>("grid_i", agent_grid_i);
  FLAMEGPU->setVariable<uint8_t>("grid_j", agent_grid_j);
  FLAMEGPU->setVariable<uint8_t>("grid_k", agent_grid_k);

  for (int i = 0; i < N_SPECIES; i++) {
    FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, C_sp[i]);
  }
  FLAMEGPU->setVariable<float>("k_elast", agent_k_elast);
  FLAMEGPU->setVariable<uint8_t>("d_dumping", agent_d_dumping);
  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);


  return flamegpu::ALIVE;
}