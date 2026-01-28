FLAMEGPU_DEVICE_FUNCTION void vec3CrossProd(float &x, float &y, float &z, float x1, float y1, float z1, float x2, float y2, float z2) {
  x = (y1 * z2 - z1 * y2);
  y = (z1 * x2 - x1 * z2);
  z = (x1 * y2 - y1 * x2);
}
FLAMEGPU_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
  x /= divisor;
  y /= divisor;
  z /= divisor;
}
FLAMEGPU_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}
FLAMEGPU_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
  float length = vec3Length(x, y, z);
  vec3Div(x, y, z, length);
}
FLAMEGPU_DEVICE_FUNCTION float getAngleBetweenVec(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2) {
  float dot_dir = x1 * x2 + y1 * y2 + z1 * z2;
  float cross_x_dir = 0.0;
  float cross_y_dir = 0.0;
  float cross_z_dir = 0.0;
  float angle = 0.0;
  float EPSILON = 0.0000000001;
  vec3CrossProd(cross_x_dir, cross_y_dir, cross_z_dir, x1, y1, z1, x2, y2, z2);
  float det_dir = vec3Length(cross_x_dir, cross_y_dir, cross_z_dir);
  if (fabsf(dot_dir) > EPSILON) {
    angle = atan2f(det_dir, dot_dir);
  }
  else {
    angle = 0.0;
  }
  
  return angle; //in radians
}
// This function computes the interaction between cells
FLAMEGPU_AGENT_FUNCTION(fnode_fnode_spatial_interaction, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");
  
  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  
  // Agent force
  float agent_fx = 0.0;
  float agent_fy = 0.0;
  float agent_fz = 0.0;
 
  const float MAX_SEARCH_RADIUS_FNODES = FLAMEGPU->environment.getProperty<float>("MAX_SEARCH_RADIUS_FNODES");
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  float EPSILON = FLAMEGPU->environment.getProperty<float>("EPSILON");
   
  int message_id = 0;
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;
 
  // direction: the vector joining interacting agents
  float dir_x = 0.0; 
  float dir_y = 0.0; 
  float dir_z = 0.0; 
  float distance = 0.0;
  
  // director cosines (with respect to global axis) of the direction vector
  float cos_x = 0.0;
  float cos_y = 0.0;
  float cos_z = 0.0;

  // total force between agents
  float total_f = 0.0;
  
  
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) { // find fnode agents within radius
    message_id = message.getVariable<int>("id");
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_z = message.getVariable<float>("z");
    
    //printf("agent %d -> message xyz (%d) = %2.6f, %2.6f, %2.6f \n", id, message_id, message_x, message_y, message_z);
        
    dir_x = agent_x - message_x; 
    dir_y = agent_y - message_y; 
    dir_z = agent_z - message_z; 
    distance = vec3Length(dir_x, dir_y, dir_z); 

    if ((distance < MAX_SEARCH_RADIUS_FNODES) && (distance > 0.0)) {

      cos_x = (1.0 * dir_x + 0.0 * dir_y + 0.0 * dir_z) / distance;
      cos_y = (0.0 * dir_x + 1.0 * dir_y + 0.0 * dir_z) / distance;
      cos_z = (0.0 * dir_x + 0.0 * dir_y + 1.0 * dir_z) / distance;
      
      
      float repulsion_force_mag = 0.1;
      // if total_f > 0, agents are attracted, if <0 agents are repelled.
      total_f = -1.0 * repulsion_force_mag * (MAX_SEARCH_RADIUS_FNODES - distance);
            
      agent_fx += -1 * total_f * cos_x; // minus comes from the direction definition (agent-message)
      agent_fy += -1 * total_f * cos_y;
      agent_fz += -1 * total_f * cos_z;
      
      //printf("Agent %d repelled -> fx = %2.6f, fy = %2.6f, fz = %2.6f,\n", id, agent_fx, agent_fy, agent_fz);
  
    }   
   
  }
  
  FLAMEGPU->setVariable<float>("fx", agent_fx);
  FLAMEGPU->setVariable<float>("fy", agent_fy);
  FLAMEGPU->setVariable<float>("fz", agent_fz);


  return flamegpu::ALIVE;
}