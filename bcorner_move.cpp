FLAMEGPU_AGENT_FUNCTION(bcorner_move, flamegpu::MessageNone, flamegpu::MessageNone) {
  //Agent position vector
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  int agent_id = FLAMEGPU->getVariable<int>("id");
  
  //Boundary  positions 
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);

  //printf("BCORNER %d position update -> (%2.6f, %2.6f,%2.6f, %2.6f, %2.6f, %2.6f)\n", agent_id, COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG);
    
  switch((int)agent_id) {
    case 1  :
    // +x,+y,+z
      agent_x = COORD_BOUNDARY_X_POS;
      agent_y = COORD_BOUNDARY_Y_POS;
      agent_z = COORD_BOUNDARY_Z_POS;
      break; 
    case 2  :
    // -x,+y,+z
      agent_x = COORD_BOUNDARY_X_NEG;
      agent_y = COORD_BOUNDARY_Y_POS;
      agent_z = COORD_BOUNDARY_Z_POS;
      break; 
    case 3  :
    // -x,-y,+z
      agent_x = COORD_BOUNDARY_X_NEG;
      agent_y = COORD_BOUNDARY_Y_NEG;
      agent_z = COORD_BOUNDARY_Z_POS;
      break; 
    case 4  :
    // +x,-y,+z
      agent_x = COORD_BOUNDARY_X_POS;
      agent_y = COORD_BOUNDARY_Y_NEG;
      agent_z = COORD_BOUNDARY_Z_POS;
      break; 
    case 5  :
    // +x,+y,-z
      agent_x = COORD_BOUNDARY_X_POS;
      agent_y = COORD_BOUNDARY_Y_POS;
      agent_z = COORD_BOUNDARY_Z_NEG;
      break; 
    case 6  :
    // -x,+y,-z
      agent_x = COORD_BOUNDARY_X_NEG;
      agent_y = COORD_BOUNDARY_Y_POS;
      agent_z = COORD_BOUNDARY_Z_NEG;
      break; 
    case 7  :
    // -x,-y,-z
      agent_x = COORD_BOUNDARY_X_NEG;
      agent_y = COORD_BOUNDARY_Y_NEG;
      agent_z = COORD_BOUNDARY_Z_NEG;
      break; 
    case 8  :
    // +x,-y,-z
      agent_x = COORD_BOUNDARY_X_POS;
      agent_y = COORD_BOUNDARY_Y_NEG;
      agent_z = COORD_BOUNDARY_Z_NEG;
      break;    
  }

  FLAMEGPU->setVariable<float>("x",agent_x);
  FLAMEGPU->setVariable<float>("y",agent_y);
  FLAMEGPU->setVariable<float>("z",agent_z);

  return flamegpu::ALIVE;
}