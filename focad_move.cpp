FLAMEGPU_AGENT_FUNCTION(focad_move, flamegpu::MessageBucket, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_x_i = FLAMEGPU->getVariable<float>("x_i"); // position of the anchor point
  float agent_y_i = FLAMEGPU->getVariable<float>("y_i");
  float agent_z_i = FLAMEGPU->getVariable<float>("z_i");
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");

  uint8_t agent_active = FLAMEGPU->getVariable<uint8_t>("active");
  int agent_fnode_id = FLAMEGPU->getVariable<int>("fnode_id");
  int agent_attached = FLAMEGPU->getVariable<int>("attached");

  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float MAX_SEARCH_RADIUS_FOCAD = FLAMEGPU->environment.getProperty<float>("MAX_SEARCH_RADIUS_FOCAD");
  const float MAX_FOCAD_ARM_LENGTH = FLAMEGPU->environment.getProperty<float>("MAX_FOCAD_ARM_LENGTH");
  const float CELL_NUCLEUS_RADIUS = FLAMEGPU->environment.getProperty<float>("CELL_NUCLEUS_RADIUS");

  if (agent_active == 0 || agent_attached == 0) {
    // move randomly away from the anchor point if not attached, or if active but not attached (this can happen when a focal adhesion detaches but is still active for a few steps)
    float dx = agent_x - agent_x_i;
    float dy = agent_y - agent_y_i;
    float dz = agent_z - agent_z_i;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);
    if (distance > MAX_FOCAD_ARM_LENGTH) { 
      // If the agent is too far from the anchor point, move it back towards the anchor point
      float direction_x = dx / distance;
      float direction_y = dy / distance;
      float direction_z = dz / distance;
      float excess = distance - MAX_FOCAD_ARM_LENGTH;  // move back towards the anchor point at a speed that would bring it back within the MAX_FOCAD_ARM_LENGTH in one time step
      agent_vx = -direction_x * excess / TIME_STEP;
      agent_vy = -direction_y * excess / TIME_STEP;
      agent_vz = -direction_z * excess / TIME_STEP;
    }
      else if (distance < CELL_NUCLEUS_RADIUS) {
        // If the agent is inside the nucleus radius, move it outward to the nucleus surface
        float direction_x = 0.0f;
        float direction_y = 0.0f;
        float direction_z = 0.0f;
        if (distance > 1e-6f) {
          direction_x = dx / distance;
          direction_y = dy / distance;
          direction_z = dz / distance;
        } else {
          float rand_dir_x = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
          float rand_dir_y = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
          float rand_dir_z = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
          float rand_len = sqrtf(rand_dir_x * rand_dir_x + rand_dir_y * rand_dir_y + rand_dir_z * rand_dir_z);
          if (rand_len < 1e-6f) {
            rand_len = 1.0f;
          }
          direction_x = rand_dir_x / rand_len;
          direction_y = rand_dir_y / rand_len;
          direction_z = rand_dir_z / rand_len;
        }
        float excess = CELL_NUCLEUS_RADIUS - distance;
        agent_vx = direction_x * excess / TIME_STEP;
        agent_vy = direction_y * excess / TIME_STEP;
        agent_vz = direction_z * excess / TIME_STEP;
      }
      else {
        // If the agent is within the max arm length, move it randomly with a capped step length
        float rand_dir_x = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
        float rand_dir_y = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
        float rand_dir_z = FLAMEGPU->random.uniform<float>(-1.0, 1.0);
        float rand_len = sqrtf(rand_dir_x * rand_dir_x + rand_dir_y * rand_dir_y + rand_dir_z * rand_dir_z);
        if (rand_len < 1e-6f) {
          rand_len = 1.0f;
        }
        float max_speed = MAX_SEARCH_RADIUS_FOCAD / TIME_STEP;
        float rand_speed_factor = FLAMEGPU->random.uniform<float>(0.0, 1.0);
        agent_vx = (rand_dir_x / rand_len) * max_speed * rand_speed_factor;
        agent_vy = (rand_dir_y / rand_len) * max_speed * rand_speed_factor;
        agent_vz = (rand_dir_z / rand_len) * max_speed * rand_speed_factor;
      }
  }
  else {
    if (agent_fnode_id == -1) {
      // This should not happen.
      printf("WARNING: FOCAD agent %d is active and attached but has no FNODE ID.\n", agent_id);
    }
    else {
      // move with the FNODE agent
      for (const auto& message : FLAMEGPU->message_in(agent_fnode_id)) {
        float message_x = message.getVariable<float>("x");
        float message_y = message.getVariable<float>("y");
        float message_z = message.getVariable<float>("z");
        agent_x = message_x;
        agent_y = message_y;
        agent_z = message_z;
        float message_vx = message.getVariable<float>("vx");
        float message_vy = message.getVariable<float>("vy");
        float message_vz = message.getVariable<float>("vz");
        agent_vx = message_vx;
        agent_vy = message_vy;
        agent_vz = message_vz;

        FLAMEGPU->setVariable<float>("x", agent_x);
        FLAMEGPU->setVariable<float>("y", agent_y);
        FLAMEGPU->setVariable<float>("z", agent_z);
        FLAMEGPU->setVariable<float>("vx", agent_vx);
        FLAMEGPU->setVariable<float>("vy", agent_vy);
        FLAMEGPU->setVariable<float>("vz", agent_vz);

        return flamegpu::ALIVE;
      }      
    }
  }


  agent_x += agent_vx * TIME_STEP;
  agent_y += agent_vy * TIME_STEP;
  agent_z += agent_vz * TIME_STEP;

  // Make sure the agent does not move outside the environment boundaries. If it does, set its position to the boundary 
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);
  if (agent_x > COORD_BOUNDARY_X_POS) {
    agent_x = COORD_BOUNDARY_X_POS;
    agent_vx = 0.0;
  }
  else if (agent_x < COORD_BOUNDARY_X_NEG) {
    agent_x = COORD_BOUNDARY_X_NEG;
    agent_vx = 0.0;
  }

  if (agent_y > COORD_BOUNDARY_Y_POS) {
    agent_y = COORD_BOUNDARY_Y_POS;
    agent_vy = 0.0;
  }
  else if (agent_y < COORD_BOUNDARY_Y_NEG) {
    agent_y = COORD_BOUNDARY_Y_NEG;
    agent_vy = 0.0;
  }

  if (agent_z > COORD_BOUNDARY_Z_POS) {
    agent_z = COORD_BOUNDARY_Z_POS;
    agent_vz = 0.0;
  }
  else if (agent_z < COORD_BOUNDARY_Z_NEG) {
    agent_z = COORD_BOUNDARY_Z_NEG;
    agent_vz = 0.0;
  }

  //Set agent variables
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);
  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);
  


  return flamegpu::ALIVE;
}