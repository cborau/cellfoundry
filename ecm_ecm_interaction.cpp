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

FLAMEGPU_AGENT_FUNCTION(ecm_ecm_interaction, flamegpu::MessageArray3D, flamegpu::MessageNone) {
  
  // Agent array variables
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  const uint32_t ECM_POPULATION_SIZE = 9261; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");
    
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");
  int grid_lin_id = FLAMEGPU->getVariable<int>("grid_lin_id");

  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  // Agent grid position
  uint8_t agent_grid_i = FLAMEGPU->getVariable<uint8_t>("grid_i");
  uint8_t agent_grid_j = FLAMEGPU->getVariable<uint8_t>("grid_j");
  uint8_t agent_grid_k = FLAMEGPU->getVariable<uint8_t>("grid_k");
  // Agent velocity
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");
  // Agent concentration of species
  int INCLUDE_DIFFUSION = FLAMEGPU->environment.getProperty<int>("INCLUDE_DIFFUSION");
  int UNSTABLE_DIFFUSION = FLAMEGPU->environment.getProperty<int>("UNSTABLE_DIFFUSION");
  float C_sp[N_SPECIES] = {}; 
  for (int i = 0; i < N_SPECIES; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
  }
  
  // Elastic constant of the ecm, only used to drag ECM agents if boundaries are moving
  float agent_k_elast = FLAMEGPU->getVariable<float>("k_elast");
  float k_elast = 0.0; //Equivalent elastic constant of two springs in series (agent and message)
  // Dumping constant of the ecm 
  const float d_dumping = FLAMEGPU->getVariable<float>("d_dumping");
  const float ECM_ECM_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("ECM_ECM_EQUILIBRIUM_DISTANCE");
  // Equilibrium distance must be adapted depending on the message grid position respect to the agent. Messages in the Neuman neighbourhood will use the original
  // whereas messages in the diagonals will use increased values.
  float grid_equilibrium_distance = 0.0; 
  float agent_fx = 0.0;
  float agent_fy = 0.0;
  float agent_fz = 0.0; 

  // Message variables (neighbouring agents)
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;
  float message_C_sp[N_SPECIES] = {}; //initialize values to 0.0
  int message_id = 0;
  float message_vx = 0.0;
  float message_vy = 0.0;
  float message_vz = 0.0;
  uint8_t message_grid_i = 0;
  uint8_t message_grid_j = 0;
  uint8_t message_grid_k = 0;
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

  int conn = 0;
  int i_diff = 0;
  int j_diff = 0;
  int k_diff = 0;
  int ct = 0;
  int DEBUG_PRINTING = FLAMEGPU->environment.getProperty<int>("DEBUG_PRINTING");
  int DEBUG_DIFFUSION = FLAMEGPU->environment.getProperty<int>("DEBUG_DIFFUSION");
  
  // Distance data of Neuman neighbourhood. Needed to solve diffusion equation
  float n_up_dist = 0.0; 
  float n_down_dist = 0.0;
  float n_right_dist = 0.0; 
  float n_left_dist = 0.0; 
  float n_front_dist = 0.0; 
  float n_back_dist = 0.0; 

  // Concentration and distance data of Neuman neighbourhood. Needed to solve diffusion equation for multiple species
  float n_up_C_sp[N_SPECIES] = {};  // concentration of agent on top of current one
  float n_down_C_sp[N_SPECIES] = {}; 
  float n_right_C_sp[N_SPECIES] = {};  
  float n_left_C_sp[N_SPECIES] = {};  
  float n_front_C_sp[N_SPECIES] = {};  
  float n_back_C_sp[N_SPECIES] = {};  
  
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  int message_count = 0;
  // Iterate location messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_grid_i, agent_grid_j, agent_grid_k)) {
    message_id = message.getVariable<int>("id");
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_z = message.getVariable<float>("z");
    message_grid_i = message.getVariable<uint8_t>("grid_i");
    message_grid_j = message.getVariable<uint8_t>("grid_j");
    message_grid_k = message.getVariable<uint8_t>("grid_k");
    for (int i = 0; i < N_SPECIES; i++) {
      message_C_sp[i] = message.getVariable<float, N_SPECIES>("C_sp", i);
    }

    i_diff = abs(agent_grid_i - message_grid_i);
    j_diff = abs(agent_grid_j - message_grid_j);
    k_diff = abs(agent_grid_k - message_grid_k);
    conn = i_diff + j_diff + k_diff;
    message_count++;
    /*
    if (id == 9 || id == 10 || id == 13 || id == 14 || id == 25 || id == 26 || id == 29 || id == 30) {
        printf("agent id %d, agent grid [%d %d %d] -> (message %d): message grid [%d %d %d], conn = %d \n", id, agent_grid_i, agent_grid_j, agent_grid_k, message_id, message_grid_i, message_grid_j, message_grid_k, conn);
    }
    */
    // If conn < 2 only the Neuman neighbourhood is checked. conn < 4 checks the 26 surrounding agents
    /*
    printf("agent id %d, agent grid [%u %u %u], message count %d -> (message %d): message grid [%u %u %u], message C_sp1 %.6f, message C_sp2 %.6f, conn = %d\n",
      id,
      (unsigned)agent_grid_i, (unsigned)agent_grid_j, (unsigned)agent_grid_k,
      message_count,
      message_id,
      (unsigned)message_grid_i, (unsigned)message_grid_j, (unsigned)message_grid_k,
      message_C_sp[0], message_C_sp[1],
      conn);
    */
    // BEWARE!!: grid domain wraps itself, meaning that agents at the grid boundaries, read messages from opposite boundaries. A grid distance condition must be added to avoid that. 
    if ((id != message_id) && (conn < 4) && (i_diff < 2) && (j_diff < 2) && (k_diff < 2)){
      if (conn < 2) {
        grid_equilibrium_distance = ECM_ECM_EQUILIBRIUM_DISTANCE; //Neuman neighbourhood
      } 
	    else if (conn < 3) {
        grid_equilibrium_distance = sqrtf(2.0) * ECM_ECM_EQUILIBRIUM_DISTANCE; //in-plane diagonals
      } 
	    else {
        grid_equilibrium_distance = sqrtf(3.0) * ECM_ECM_EQUILIBRIUM_DISTANCE; //corners of the 3x3x3 adjacent agents
      }

      ct++;
      dir_x = agent_x - message_x; 
      dir_y = agent_y - message_y; 
      dir_z = agent_z - message_z; 
      distance = vec3Length(dir_x, dir_y, dir_z);    

      if (conn < 2) {      
        if (message_grid_i < agent_grid_i)        
          n_left_dist = distance;
        if (message_grid_i > agent_grid_i) 
          n_right_dist = distance;
        if (message_grid_j < agent_grid_j) 
          n_back_dist = distance;
        if (message_grid_j > agent_grid_j) 
          n_front_dist = distance;
        if (message_grid_k < agent_grid_k) 
          n_down_dist = distance;
        if (message_grid_k > agent_grid_k) 
          n_up_dist = distance;

        // For multiple species diffusion
        for (int i = 0; i < N_SPECIES; i++) {
          if (message_grid_i < agent_grid_i) 
            n_left_C_sp[i] = message_C_sp[i];
          if (message_grid_i > agent_grid_i) 
            n_right_C_sp[i] = message_C_sp[i];
          if (message_grid_j < agent_grid_j) 
            n_back_C_sp[i] = message_C_sp[i];
          if (message_grid_j > agent_grid_j) 
            n_front_C_sp[i] = message_C_sp[i];
          if (message_grid_k < agent_grid_k) 
            n_down_C_sp[i] = message_C_sp[i];
          if (message_grid_k > agent_grid_k) 
            n_up_C_sp[i] = message_C_sp[i];
        }
      }
      
      message_vx = message.getVariable<float>("vx");
      message_vy = message.getVariable<float>("vy");
      message_vz = message.getVariable<float>("vz");
      message_k_elast = message.getVariable<float>("k_elast");
    
      // compute equivalent stiffness of springs in series
      k_elast = (agent_k_elast * message_k_elast) / (agent_k_elast + message_k_elast);
      
      cos_x = (1.0 * dir_x + 0.0 * dir_y + 0.0 * dir_z) / distance;
      cos_y = (0.0 * dir_x + 1.0 * dir_y + 0.0 * dir_z) / distance;
      cos_z = (0.0 * dir_x + 0.0 * dir_y + 1.0 * dir_z) / distance;
      
      // angles between agent & message velocity vector and the direction joining them      
      angle_agent_v_dir = getAngleBetweenVec(agent_vx,agent_vy,agent_vz,dir_x,dir_y,dir_z);
      angle_message_v_dir = getAngleBetweenVec(message_vx,message_vy,message_vz,dir_x,dir_y,dir_z);

      // relative speed <0 means particles are getting closer
      relative_speed = vec3Length(agent_vx, agent_vy, agent_vz) * cosf(angle_agent_v_dir) - vec3Length(message_vx, message_vy, message_vz) * cosf(angle_message_v_dir);
      // if total_f > 0, agents are attracted, if <0 agents are repelled
      total_f = (distance - grid_equilibrium_distance) * (k_elast) + d_dumping * relative_speed;

      agent_fx += -1 * total_f * cos_x; // minus comes from the direction definition (agent-message)
      agent_fy += -1 * total_f * cos_y;
      agent_fz += -1 * total_f * cos_z;	
      
      //if (DEBUG_PRINTING == 1 && (id == 9 || id == 10 || id == 13 || id == 22)) {
      /*if (DEBUG_PRINTING == 1) {
        printf("ECM interaction [id1: %d - id2: %d] agent_pos (%2.6f, %2.6f, %2.6f), message_pos (%2.6f, %2.6f, %2.6f)\n", id, message_id, agent_x, agent_y, agent_z, message_x, message_y, message_z);
        printf("ECM interaction id1: %d - id2: %d distance -> (%2.6f)\n", id, message_id, distance);
        printf("ECM interaction id1: %d - id2: %d total_f -> %2.6f (%2.6f , %2.6f, %2.6f)\n", id, message_id, total_f, -1 * total_f * cos_x, -1 * total_f * cos_y, -1 * total_f * cos_z);
      }
      */
    }
  }

  //printf("Array3D for agent %d read %d messages! grid [%d %d %d], pos (%2.6f , %2.6f, %2.6f) \n", id, ct, agent_grid_i, agent_grid_j, agent_grid_k, agent_x, agent_y, agent_z);
  FLAMEGPU->setVariable<float>("fx", agent_fx);
  FLAMEGPU->setVariable<float>("fy", agent_fy);
  FLAMEGPU->setVariable<float>("fz", agent_fz);

  //Apply diffusion equation
  if (INCLUDE_DIFFUSION == 1){
    float R = 0.0; // reactive term. Unused here, as cell agents consume species in a different function
    //Calculate distances to neighbours
    float dx = ((n_left_dist > 0.0) & (n_right_dist > 0.0)) ? (n_left_dist + n_right_dist) / 2.0 : fmaxf(n_left_dist,n_right_dist);
    float dy = ((n_front_dist > 0.0) & (n_back_dist > 0.0)) ? (n_front_dist + n_back_dist) / 2.0 : fmaxf(n_front_dist,n_back_dist);
    float dz = ((n_up_dist > 0.0) & (n_down_dist > 0.0)) ? (n_up_dist + n_down_dist) / 2.0 : fmaxf(n_up_dist,n_down_dist);
    
    //Apply diffusion equation for multiple species
    float agent_C_sp_prev[N_SPECIES] = {}; 
    for (int i = 0; i < N_SPECIES; i++) {
      float DIFFUSION_COEFF = FLAMEGPU->environment.getProperty<float>("DIFFUSION_COEFF_MULTI",i);
      float Fx = DIFFUSION_COEFF * TIME_STEP / powf(dx, 2.0);
      float Fy = DIFFUSION_COEFF * TIME_STEP / powf(dy, 2.0);
      float Fz = DIFFUSION_COEFF * TIME_STEP / powf(dz, 2.0);
      agent_C_sp_prev[i] = C_sp[i];

      if (UNSTABLE_DIFFUSION) {

        // Stability fix for diffusion when Forward Euler in 3D becomes unstable (too large dt, too small dx/dy/dz, or large D).
        // Instead of:
        //   C^{n+1} = C^n + Fx*(C_L - 2C^n + C_R) + Fy*(C_F - 2C^n + C_B) + Fz*(C_U - 2C^n + C_D) + R*dt
        //
        // use a "center-implicit" / semi-implicit (damped) update that keeps neighbour values at time n
        // but treats the central -2C term implicitly:
        //
        //   (1 + 2(Fx+Fy+Fz)) * C^{n+1}
        //     = C^n + Fx*(C_L + C_R) + Fy*(C_F + C_B) + Fz*(C_U + C_D) + R*dt
        //
        //   => C^{n+1} = [ C^n + Fx*(C_L + C_R) + Fy*(C_F + C_B) + Fz*(C_U + C_D) + R*dt ]
        //               / [ 1 + 2(Fx+Fy+Fz) ]
        //
        // where Fx = D*dt/dx^2, Fy = D*dt/dy^2, Fz = D*dt/dz^2 (dx,dy,dz may vary per voxel).
        //
        // This behaves like one Jacobi iteration of the fully implicit diffusion solve:
        //   (I - dt*D*Laplace) C^{n+1} = C^n
        // It is unconditionally stable (no blow-up), but more "damped" for large Fx,Fy,Fz (approaches local averaging).
        const float S = Fx + Fy + Fz;  // >= 0
        const float numer =
            agent_C_sp_prev[i]
          + Fx * (n_left_C_sp[i] + n_right_C_sp[i])
          + Fy * (n_front_C_sp[i] + n_back_C_sp[i])
          + Fz * (n_up_C_sp[i] + n_down_C_sp[i])
          + R * TIME_STEP;

        C_sp[i] = numer / (1.0f + 2.0f * S);
      } 
      else {
        // Standard explicit Forward Euler diffusion update (conditionally stable):
        //   C^{n+1} = C^n + Fx*(C_L - 2C^n + C_R) + Fy*(C_F - 2C^n + C_B) + Fz*(C_U - 2C^n + C_D) + R*dt
        // Requires dt small enough to satisfy the 3D diffusion stability limit.
        C_sp[i] = agent_C_sp_prev[i] 
        + Fx * (n_left_C_sp[i] - (2 * agent_C_sp_prev[i]) + n_right_C_sp[i]) 
        + Fy * (n_front_C_sp[i] - (2 * agent_C_sp_prev[i]) + n_back_C_sp[i]) 
        + Fz * (n_up_C_sp[i] - (2 * agent_C_sp_prev[i]) + n_down_C_sp[i]) 
        + R * TIME_STEP;
      }

      // Prevent negative concentrations 
      if (C_sp[i] < 0.0f) C_sp[i] = 0.0f;

      FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, C_sp[i]);

      //Update macro property
      C_SP_MACRO[i][grid_lin_id].exchange(C_sp[i]);

      //printf("agent id: %d, agent_C_sp AFTER: %.6f, species %d \n", id, C_sp[i], i+1);
      if (DEBUG_DIFFUSION == 1 && i == 0) {  
        //printf("DIFFUSION for agent %d, species %d, [dx,dy,dz] = [%2.6f , %2.6f, %2.6f], [Fx,Fy,Fz] = [%2.6f , %2.6f, %2.6f] \n", id, i+1, dx, dy, dz, Fx, Fy, Fz);
        printf("agent %d: MULTI left conc = %2.6f, right conc = %2.6f \n", id, n_left_C_sp[i], n_right_C_sp[i]);
        printf("agent %d: MULTI front conc = %2.6f, back conc = %2.6f \n", id, n_front_C_sp[i], n_back_C_sp[i]);
        printf("agent %d: MULTI up conc = %2.6f, down conc = %2.6f \n", id, n_up_C_sp[i], n_down_C_sp[i]);
        printf("agent %d: MULTI conc prev = %2.6f, current conc = %2.6f \n", id, agent_C_sp_prev[i], C_sp[i]); 
      }   
    }     
  }
  return flamegpu::ALIVE;
}