/**
 * fnode_boundary_interaction
 *
 * Purpose:
 *   Compute boundary reaction forces on FNODE agents near domain boundaries,
 *   including optional elastic and damping contributions per face.
 *
 * Inputs:
 *   - FNODE position/velocity
 *   - Boundary coordinates, stiffness/damping, and movement settings
 *
 * Outputs:
 *   - boundary_fx, boundary_fy, boundary_fz stored on each FNODE
 */
FLAMEGPU_AGENT_FUNCTION(fnode_boundary_interaction, flamegpu::MessageNone, flamegpu::MessageNone) {
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");

  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  
  // Agent velocity
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");
   
  //Interaction with boundaries
  float boundary_fx = 0.0;
  float boundary_fy = 0.0;
  float boundary_fz = 0.0;
  float separation_x_pos = 0.0;
  float separation_x_neg = 0.0;
  float separation_y_pos = 0.0;
  float separation_y_neg = 0.0;
  float separation_z_pos = 0.0;
  float separation_z_neg = 0.0;
  const float ECM_BOUNDARY_INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("FIBRE_NODE_BOUNDARY_INTERACTION_RADIUS");
  const float ECM_BOUNDARY_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE");
  float EPSILON = FLAMEGPU->environment.getProperty<float>("EPSILON");

  // Get position of the boundaries
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);

  const float INIT_COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("INIT_COORDS_BOUNDARIES", 0);
  const float INIT_COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("INIT_COORDS_BOUNDARIES", 1);
  const float INIT_COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("INIT_COORDS_BOUNDARIES", 2);
  const float INIT_COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("INIT_COORDS_BOUNDARIES", 3);
  const float INIT_COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("INIT_COORDS_BOUNDARIES", 4);
  const float INIT_COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("INIT_COORDS_BOUNDARIES", 5);
  
  // Get displacement rate of the boundaries
  const float DISP_RATE_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",0);
  const float DISP_RATE_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",1);
  const float DISP_RATE_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",2);
  const float DISP_RATE_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",3);
  const float DISP_RATE_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",4);
  const float DISP_RATE_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",5);

  // Get boundarY conditions
  const int ALLOW_BOUNDARY_ELASTIC_MOVEMENT_X_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_BOUNDARY_ELASTIC_MOVEMENT", 0);
  const int ALLOW_BOUNDARY_ELASTIC_MOVEMENT_X_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_BOUNDARY_ELASTIC_MOVEMENT", 1);
  const int ALLOW_BOUNDARY_ELASTIC_MOVEMENT_Y_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_BOUNDARY_ELASTIC_MOVEMENT", 2);
  const int ALLOW_BOUNDARY_ELASTIC_MOVEMENT_Y_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_BOUNDARY_ELASTIC_MOVEMENT", 3);
  const int ALLOW_BOUNDARY_ELASTIC_MOVEMENT_Z_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_BOUNDARY_ELASTIC_MOVEMENT", 4);
  const int ALLOW_BOUNDARY_ELASTIC_MOVEMENT_Z_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_BOUNDARY_ELASTIC_MOVEMENT", 5);

  const float BOUNDARY_STIFFNESS_X_POS = FLAMEGPU->environment.getProperty<float>("BOUNDARY_STIFFNESS", 0);
  const float BOUNDARY_STIFFNESS_X_NEG = FLAMEGPU->environment.getProperty<float>("BOUNDARY_STIFFNESS", 1);
  const float BOUNDARY_STIFFNESS_Y_POS = FLAMEGPU->environment.getProperty<float>("BOUNDARY_STIFFNESS", 2);
  const float BOUNDARY_STIFFNESS_Y_NEG = FLAMEGPU->environment.getProperty<float>("BOUNDARY_STIFFNESS", 3);
  const float BOUNDARY_STIFFNESS_Z_POS = FLAMEGPU->environment.getProperty<float>("BOUNDARY_STIFFNESS", 4);
  const float BOUNDARY_STIFFNESS_Z_NEG = FLAMEGPU->environment.getProperty<float>("BOUNDARY_STIFFNESS", 5);

  const float BOUNDARY_DUMPING_X_POS = FLAMEGPU->environment.getProperty<float>("BOUNDARY_DUMPING", 0);
  const float BOUNDARY_DUMPING_X_NEG = FLAMEGPU->environment.getProperty<float>("BOUNDARY_DUMPING", 1);
  const float BOUNDARY_DUMPING_Y_POS = FLAMEGPU->environment.getProperty<float>("BOUNDARY_DUMPING", 2);
  const float BOUNDARY_DUMPING_Y_NEG = FLAMEGPU->environment.getProperty<float>("BOUNDARY_DUMPING", 3);
  const float BOUNDARY_DUMPING_Z_POS = FLAMEGPU->environment.getProperty<float>("BOUNDARY_DUMPING", 4);
  const float BOUNDARY_DUMPING_Z_NEG = FLAMEGPU->environment.getProperty<float>("BOUNDARY_DUMPING", 5);

  // Check for ecm-boundary separations
  //this takes into account the distance with respect to the actual boundary position, while forces are calculated with respect to boundary initial position
  separation_x_pos = (agent_x - COORD_BOUNDARY_X_POS); 
  separation_x_neg = (agent_x - COORD_BOUNDARY_X_NEG);
  separation_y_pos = (agent_y - COORD_BOUNDARY_Y_POS);
  separation_y_neg = (agent_y - COORD_BOUNDARY_Y_NEG);
  separation_z_pos = (agent_z - COORD_BOUNDARY_Z_POS);
  separation_z_neg = (agent_z - COORD_BOUNDARY_Z_NEG);

    

  if (fabsf(separation_x_pos) < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_x_pos) > EPSILON && ALLOW_BOUNDARY_ELASTIC_MOVEMENT_X_POS > 0){
      boundary_fx += -1 * (agent_x - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE - INIT_COORD_BOUNDARY_X_POS) * (BOUNDARY_STIFFNESS_X_POS) - BOUNDARY_DUMPING_X_POS * (agent_vx - DISP_RATE_BOUNDARY_X_POS);
  }
  if (fabsf(separation_x_neg) < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_x_neg) > EPSILON && ALLOW_BOUNDARY_ELASTIC_MOVEMENT_X_NEG > 0){
      boundary_fx +=  -1* (agent_x - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE - INIT_COORD_BOUNDARY_X_NEG) * (BOUNDARY_STIFFNESS_X_NEG) - BOUNDARY_DUMPING_X_NEG * (agent_vx - DISP_RATE_BOUNDARY_X_NEG);
  }
  if (fabsf(separation_y_pos) < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_y_pos) > EPSILON && ALLOW_BOUNDARY_ELASTIC_MOVEMENT_Y_POS > 0) {
      boundary_fy += -1 * (agent_y - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE - INIT_COORD_BOUNDARY_Y_POS) * (BOUNDARY_STIFFNESS_Y_POS) - BOUNDARY_DUMPING_Y_POS * (agent_vy - DISP_RATE_BOUNDARY_Y_POS);
  }
  if (fabsf(separation_y_neg) < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_y_neg) > EPSILON && ALLOW_BOUNDARY_ELASTIC_MOVEMENT_Y_NEG > 0) {
      boundary_fy +=  -1* (agent_y - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE - INIT_COORD_BOUNDARY_Y_NEG) * (BOUNDARY_STIFFNESS_Y_NEG) - BOUNDARY_DUMPING_Y_NEG * (agent_vy - DISP_RATE_BOUNDARY_Y_NEG);
  }
  if (fabsf(separation_z_pos) < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_z_pos) > EPSILON && ALLOW_BOUNDARY_ELASTIC_MOVEMENT_Z_POS > 0) {
      boundary_fz += -1 * (agent_z - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE - INIT_COORD_BOUNDARY_Z_POS) * (BOUNDARY_STIFFNESS_Z_POS) - BOUNDARY_DUMPING_Z_POS * (agent_vz - DISP_RATE_BOUNDARY_Z_POS);
  }
  if (fabsf(separation_z_neg) < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_z_neg) > EPSILON && ALLOW_BOUNDARY_ELASTIC_MOVEMENT_Z_NEG > 0) {
      boundary_fz +=  -1* (agent_z - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE - INIT_COORD_BOUNDARY_Z_NEG) * (BOUNDARY_STIFFNESS_Z_NEG) - BOUNDARY_DUMPING_Z_NEG * (agent_vz - DISP_RATE_BOUNDARY_Z_NEG);
  }

  FLAMEGPU->setVariable<float>("boundary_fx", boundary_fx);
  FLAMEGPU->setVariable<float>("boundary_fy", boundary_fy);
  FLAMEGPU->setVariable<float>("boundary_fz", boundary_fz);

  return flamegpu::ALIVE;
}