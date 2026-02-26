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
FLAMEGPU_DEVICE_FUNCTION void getMaxForceDir(float &dx, float &dy, float &dz,float x, float y, float z){

  if(x >= y && x >= z){
    dx = 1.0;
    dy = 0.0;
    dz = 0.0;
  } 
  else if(y >= z && y >= x){
    dx = 0.0;
    dy = 1.0;
    dz = 0.0;
  } 
  else{   
    dx = 0.0;
    dy = 0.0;
    dz = 1.0;
  }
}
/**
 * fnode_fnode_bucket_interaction
 *
 * Purpose:
 *   Compute spring-damper forces along explicit FNODE connectivity links and
 *   accumulate network mechanical metrics (extension/compression/elastic energy).
 *
 * Inputs:
 *   - Bucket messages keyed by linked node ids
 *   - Connectivity arrays and per-link equilibrium distances
 *
 * Outputs:
 *   - Updated FNODE force components and mechanical summary variables
 */
FLAMEGPU_AGENT_FUNCTION(fnode_fnode_bucket_interaction, flamegpu::MessageBucket, flamegpu::MessageNone) {
  // Current simulation step
  const unsigned int CURRENT_STEP = FLAMEGPU->getStepCounter();

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
  
  // Agent neighbours
  const uint8_t MAX_CONNECTIVITY = 8; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function. TODO: declare it somehow at compile time
  
  // Elastic constant of the fibre 
  float k_elast = 0.0; //Equivalent elastic constant of two springs in series (agent and message)
  const float FIBRE_SEGMENT_K_ELAST = FLAMEGPU->environment.getProperty<float>("FIBRE_SEGMENT_K_ELAST");
  const float agent_degradation = FLAMEGPU->getVariable<float>("degradation");
  const float agent_k_elast = fmaxf(0.0f, FIBRE_SEGMENT_K_ELAST * (1.0f - agent_degradation));
   
  // Dumping constant of the fibre 
  const float d_dumping = FLAMEGPU->getVariable<float>("d_dumping");

  float equilibrium_distance[MAX_CONNECTIVITY] = {};
  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
    equilibrium_distance[i] = FLAMEGPU->getVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i);
  }


    
  float agent_fx = 0.0f;
  float agent_fy = 0.0f;
  float agent_fz = 0.0f; 
  float agent_fx_abs = 0.0; // if there are opposing forces (F) in the same direction, agent_fx = 0, but agent_fx_abs = 2*F
  float agent_fy_abs = 0.0;
  float agent_fz_abs = 0.0; 
  float agent_f_extension = 0.0;
  float agent_f_compression = 0.0;
  float agent_elastic_energy = 0.0;
  
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;
  int message_id = 0;
  float message_vx = 0.0;
  float message_vy = 0.0;
  float message_vz = 0.0;
  float message_degradation = 0.0;
  float message_k_elast = 0.0;

  // Initialize other variables
  float EPSILON = FLAMEGPU->environment.getProperty<float>("EPSILON");
  // direction: the vector joining interacting agents
  float dir_x = 0.0; 
  float dir_y = 0.0; 
  float dir_z = 0.0; 
  float distance = 0.0; 
  // director cosines (with respect to global axis) of the direction vector
  float cos_x = 0.0;
  float cos_y = 0.0;
  float cos_z = 0.0;
  // angle (in radians) between agent velocity vector and direction vector
  float angle_agent_v_dir = 0.0;
  float angle_message_v_dir = 0.0;
  // relative speed between agents
  float relative_speed = 0.0;
  // total force between agents
  float total_f = 0.0;

  int DEBUG_PRINTING = FLAMEGPU->environment.getProperty<int>("DEBUG_PRINTING");
  
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  
  float linked_nodes[MAX_CONNECTIVITY] = {}; 
  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
    linked_nodes[i] = FLAMEGPU->getVariable<float, MAX_CONNECTIVITY>("linked_nodes", i);
    if (linked_nodes[i] < 0) // -1 values when no connection
      continue;
    for (const auto& message : FLAMEGPU->message_in(linked_nodes[i])) {
      message_id = message.getVariable<int>("id");
      message_x = message.getVariable<float>("x");
      message_y = message.getVariable<float>("y");
      message_z = message.getVariable<float>("z");
      message_vx = message.getVariable<float>("vx");
      message_vy = message.getVariable<float>("vy");
      message_vz = message.getVariable<float>("vz");
      message_degradation = message.getVariable<float>("degradation");
      message_k_elast = fmaxf(0.0f, FIBRE_SEGMENT_K_ELAST * (1.0f - message_degradation));
      
      dir_x = agent_x - message_x; 
      dir_y = agent_y - message_y; 
      dir_z = agent_z - message_z; 
      distance = vec3Length(dir_x, dir_y, dir_z); 
     

      // compute equivalent stiffness 
      if ((agent_k_elast + message_k_elast) > 1e-12f) {
        k_elast = (agent_k_elast * message_k_elast) / ((agent_k_elast) + (message_k_elast));
      } else {
        k_elast = 0.0f;
      }
      
      cos_x = (1.0 * dir_x + 0.0 * dir_y + 0.0 * dir_z) / distance;
      cos_y = (0.0 * dir_x + 1.0 * dir_y + 0.0 * dir_z) / distance;
      cos_z = (0.0 * dir_x + 0.0 * dir_y + 1.0 * dir_z) / distance;
      
      // angles between agent & message velocity vector and the direction joining them      
      angle_agent_v_dir = getAngleBetweenVec(agent_vx,agent_vy,agent_vz,dir_x,dir_y,dir_z);
      angle_message_v_dir = getAngleBetweenVec(message_vx,message_vy,message_vz,dir_x,dir_y,dir_z);

      // relative speed <0 means nodes are getting closer
      relative_speed = vec3Length(agent_vx, agent_vy, agent_vz) * cosf(angle_agent_v_dir) - vec3Length(message_vx, message_vy, message_vz) * cosf(angle_message_v_dir);
	    
      // if CURRENT_STEP == 0, network is relaxed, so we can store the equilibrium distance between nodes. In subsequent steps, we can compute the relative distance (distance - equilibrium_distance) to determine if the fibre is under tension or compression.
      if (CURRENT_STEP == 0) { 
        equilibrium_distance[i] = distance;
        FLAMEGPU->setVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i, equilibrium_distance[i]);
      }
      else {
        float relative_dist = (distance - equilibrium_distance[i]);

        //printf("DEBUG: ECM interaction id1: %d - id2: %d distance -> (%2.6f), relative_dist (%2.6f)\n", id, message_id, distance, relative_dist);
        if (relative_dist < 0) {
          //total_f = 0.0; // TEST: if fibre is under compression, it will buckle exerting no elastic resistance
          total_f = relative_dist * (k_elast) + d_dumping * relative_speed;
        }
        else {
        // if total_f > 0, agents are attracted, if <0 agents are repelled
          total_f = relative_dist * (k_elast) + d_dumping * relative_speed;  
        }
        

        if (total_f < 0) {
          agent_f_compression += total_f;
        } 
        else {
          agent_f_extension += total_f;
          // store the absolute extensions in each direction
          agent_fx_abs += fabsf(total_f * cos_x);
          agent_fy_abs += fabsf(total_f * cos_y);
          agent_fz_abs += fabsf(total_f * cos_z);
        }

        if (k_elast > 1e-12f) {
          agent_elastic_energy += 0.5f * (total_f * total_f) / k_elast;
        }

        agent_fx += -1 * total_f * cos_x; // minus comes from the direction definition (agent-message)
        agent_fy += -1 * total_f * cos_y;
        agent_fz += -1 * total_f * cos_z;
      }

      if (DEBUG_PRINTING == 1 && (id == 9 || id == 10 || id == 11 || id == 12)) {
         printf("ECM interaction [id1: %d - id2: %d] agent_pos (%2.6f, %2.6f, %2.6f), message_pos (%2.6f, %2.6f, %2.6f)\n", id, message_id, agent_x, agent_y, agent_z, message_x, message_y, message_z);
         printf("ECM interaction id1: %d - id2: %d distance -> (%2.6f)\n", id, message_id, distance);
         printf("ECM interaction id1: %d - id2: %d total_f -> %2.6f (%2.6f , %2.6f, %2.6f)\n", id, message_id, total_f, -1 * total_f * cos_x, -1 * total_f * cos_y, -1 * total_f * cos_z);
      }
    }
  }
  
      
  FLAMEGPU->setVariable<float>("fx", agent_fx);
  FLAMEGPU->setVariable<float>("fy", agent_fy);
  FLAMEGPU->setVariable<float>("fz", agent_fz);
  FLAMEGPU->setVariable<float>("f_extension", agent_f_extension);
  FLAMEGPU->setVariable<float>("f_compression", agent_f_compression);
  FLAMEGPU->setVariable<float>("elastic_energy", agent_elastic_energy);


  return flamegpu::ALIVE;
}