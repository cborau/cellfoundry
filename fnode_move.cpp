FLAMEGPU_HOST_DEVICE_FUNCTION void boundPosition(int id, float &x, float &y, float &z, 
        uint8_t &cxpos, uint8_t &cxneg, uint8_t &cypos, uint8_t &cyneg, uint8_t &czpos, uint8_t &czneg, 
        const float bxpos, const float bxneg, const float bypos, const float byneg, const float bzpos, const float bzneg,
        const int clamp_on_xpos, const int clamp_on_xneg, const int clamp_on_ypos, const int clamp_on_yneg, const int clamp_on_zpos, const int clamp_on_zneg, const float FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE) {
    
  //if (id == 9 || id = 10) {
  //    printf("Boundposition ANTES agent %d position %2.4f, %2.4f, %2.4f ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n", id, x, y, z, bxpos, bxneg, bypos, byneg, bzpos, bzneg, cxpos, cxneg, cypos, cyneg, czpos, czneg);
  //}
  float EPSILON = 0.00000001;

  if (cxpos == 1) {
      x = bxpos - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE; // redundant. Could say "do nothing"
  } else {
    if (x > bxpos || fabsf(x - bxpos) < FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE + EPSILON) {          
      if (clamp_on_xpos == 1) {
        cxpos = 1;
        x = bxpos - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
      }
    }
  }
     
  if (cxneg == 1) {
    x = bxneg + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
  } else {
    if (x < bxneg || fabsf(x - bxneg) < FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE + EPSILON) {          
      if (clamp_on_xneg == 1) {
        cxneg = 1;
        x = bxneg + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
      }
    }
  }

  if (cypos == 1) {
    y = bypos - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
  } else {
    if (y > bypos || fabsf(y - bypos) < FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE + EPSILON) {
      if (clamp_on_ypos == 1) {
        cypos = 1;
        y = bypos - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
      }
    }
  }
  
  if (cyneg == 1) {
    y = byneg + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
  } else {
    if (y < byneg || fabsf(y - byneg) < FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE + EPSILON) {     
      if (clamp_on_yneg == 1) {
        cyneg = 1;
        y = byneg + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
      }
    }
  }


  if (czpos == 1) {
    z = bzpos - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
  } else {
    if (z > bzpos || fabsf(z - bzpos) < FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE + EPSILON) {
      if (clamp_on_zpos == 1) {
        czpos = 1;
        z = bzpos - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
      }
    }
  }
   
  if (czneg == 1) {
    z = bzneg + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
  }
  else {
    if (z < bzneg || fabsf(z - bzneg) < FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE + EPSILON) {
      if (clamp_on_zneg == 1) {
        czneg = 1;
        z = bzneg + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
      }
    }
  }

  //if (id == 9 || id = 10) {
  //   printf("Boundposition DESPUES agent %d position %2.4f, %2.4f, %2.4f ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n", id, x, y, z, bxpos, bxneg, bypos, byneg, bzpos, bzneg, cxpos, cxneg, cypos, cyneg, czpos, czneg);
  //}
}
FLAMEGPU_AGENT_FUNCTION(fnode_move, flamegpu::MessageNone, flamegpu::MessageNone) {
  
  int id = FLAMEGPU->getVariable<int>("id");
  //Agent position vector
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");

  int DEBUG_PRINTING = FLAMEGPU->environment.getProperty<int>("DEBUG_PRINTING");

  // Agent velocity
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");
  
  // Agent clamps
  uint8_t clamped_bx_pos = FLAMEGPU->getVariable<uint8_t>("clamped_bx_pos");
  uint8_t clamped_bx_neg = FLAMEGPU->getVariable<uint8_t>("clamped_bx_neg");
  uint8_t clamped_by_pos = FLAMEGPU->getVariable<uint8_t>("clamped_by_pos");
  uint8_t clamped_by_neg = FLAMEGPU->getVariable<uint8_t>("clamped_by_neg");
  uint8_t clamped_bz_pos = FLAMEGPU->getVariable<uint8_t>("clamped_bz_pos");
  uint8_t clamped_bz_neg = FLAMEGPU->getVariable<uint8_t>("clamped_bz_neg");

  // Agent force transmitted to clamped or elastic boundaries 
  float f_bx_pos = 0.0;
  float f_bx_neg = 0.0;
  float f_by_pos = 0.0;
  float f_by_neg = 0.0;
  float f_bz_pos = 0.0;
  float f_bz_neg = 0.0;

  float f_bx_pos_y = 0.0;
  float f_bx_pos_z = 0.0;
  float f_bx_neg_y = 0.0;
  float f_bx_neg_z = 0.0;
  float f_by_pos_x = 0.0;
  float f_by_pos_z = 0.0;
  float f_by_neg_x = 0.0;
  float f_by_neg_z = 0.0;
  float f_bz_pos_x = 0.0;
  float f_bz_pos_y = 0.0;
  float f_bz_neg_x = 0.0;
  float f_bz_neg_y = 0.0;
 
  //Forces acting on the agent
  float agent_fx = FLAMEGPU->getVariable<float>("fx");
  float agent_fy = FLAMEGPU->getVariable<float>("fy");
  float agent_fz = FLAMEGPU->getVariable<float>("fz");
  float agent_boundary_fx = FLAMEGPU->getVariable<float>("boundary_fx");
  float agent_boundary_fy = FLAMEGPU->getVariable<float>("boundary_fy");
  float agent_boundary_fz = FLAMEGPU->getVariable<float>("boundary_fz");
  
 
  //Add the force coming from the boundaries
  agent_fx += agent_boundary_fx;
  agent_fy += agent_boundary_fy;
  agent_fz += agent_boundary_fz;

  
  if (DEBUG_PRINTING == 1 && (id == 9 || id == 10 || id == 14 || id == 25 || id == 26 || id == 29 || id == 30)){
    printf("ECM move ID: %d clamps before -> (%d, %d, %d, %d, %d, %d)\n", id, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);
    printf("ECM move ID: %d pos -> (%2.6f, %2.6f, %2.6f)\n", id, agent_x, agent_y, agent_z);
    printf("ECM move ID: %d vel -> (%2.6f, %2.6f, %2.6f)\n", id, agent_vx, agent_vy, agent_vz);
    printf("ECM move ID: %d f -> (%2.6f, %2.6f, %2.6f)\n", id, agent_fx, agent_fy, agent_fz);
    printf("ECM move ID: %d bf -> (%2.6f, %2.6f, %2.6f)\n", id, agent_boundary_fx, agent_boundary_fy, agent_boundary_fz);
    printf("ECM move ID: %d f after -> (%2.6f, %2.6f, %2.6f)\n", id, agent_fx, agent_fy, agent_fz);
  }

  //Get the new position and velocity: 
  // SUM(F) = ETA * v;
  // v(t) = SUM(F) / ETA; 
  // x(t) = x(t-1) + v(t) * dt
  const float  TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float ECM_ETA = FLAMEGPU->environment.getProperty<float>("ECM_ETA");
  //Bound the position within the environment   
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);
  const float DISP_RATE_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",0);
  const float DISP_RATE_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",1);
  const float DISP_RATE_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",2);
  const float DISP_RATE_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",3);
  const float DISP_RATE_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",4);
  const float DISP_RATE_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",5);
  const int CLAMP_AGENT_TOUCHING_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<int>("CLAMP_AGENT_TOUCHING_BOUNDARY", 0);
  const int CLAMP_AGENT_TOUCHING_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<int>("CLAMP_AGENT_TOUCHING_BOUNDARY", 1);
  const int CLAMP_AGENT_TOUCHING_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<int>("CLAMP_AGENT_TOUCHING_BOUNDARY", 2);
  const int CLAMP_AGENT_TOUCHING_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<int>("CLAMP_AGENT_TOUCHING_BOUNDARY", 3);
  const int CLAMP_AGENT_TOUCHING_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<int>("CLAMP_AGENT_TOUCHING_BOUNDARY", 4);
  const int CLAMP_AGENT_TOUCHING_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<int>("CLAMP_AGENT_TOUCHING_BOUNDARY", 5);
  const int ALLOW_AGENT_SLIDING_X_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 0);
  const int ALLOW_AGENT_SLIDING_X_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 1);
  const int ALLOW_AGENT_SLIDING_Y_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 2);
  const int ALLOW_AGENT_SLIDING_Y_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 3);
  const int ALLOW_AGENT_SLIDING_Z_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 4);
  const int ALLOW_AGENT_SLIDING_Z_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 5);
  const float FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE");
  const float FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE");

  const float DISP_RATE_BOUNDARY_PARALLEL_X_POS_Y = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 0);
  const float DISP_RATE_BOUNDARY_PARALLEL_X_POS_Z = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 1);
  const float DISP_RATE_BOUNDARY_PARALLEL_X_NEG_Y = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 2);
  const float DISP_RATE_BOUNDARY_PARALLEL_X_NEG_Z = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 3);
  const float DISP_RATE_BOUNDARY_PARALLEL_Y_POS_X = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 4);
  const float DISP_RATE_BOUNDARY_PARALLEL_Y_POS_Z = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 5);
  const float DISP_RATE_BOUNDARY_PARALLEL_Y_NEG_X = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 6);
  const float DISP_RATE_BOUNDARY_PARALLEL_Y_NEG_Z = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 7);
  const float DISP_RATE_BOUNDARY_PARALLEL_Z_POS_X = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 8);
  const float DISP_RATE_BOUNDARY_PARALLEL_Z_POS_Y = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 9);
  const float DISP_RATE_BOUNDARY_PARALLEL_Z_NEG_X = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 10);
  const float DISP_RATE_BOUNDARY_PARALLEL_Z_NEG_Y = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES_PARALLEL", 11);

  float prev_agent_x = agent_x;
  float prev_agent_y = agent_y;
  float prev_agent_z = agent_z;
  float inc_pos_max = 0.0;
   
  if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) {
    agent_vx = (agent_fx) / ECM_ETA;
    agent_x += agent_vx * TIME_STEP;
    inc_pos_max = fmaxf(inc_pos_max, fabsf(agent_vx * TIME_STEP));
  }

  if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) {
    agent_vy = (agent_fy) / ECM_ETA;
    agent_y += agent_vy * TIME_STEP;
    inc_pos_max = fmaxf(inc_pos_max, fabsf(agent_vy * TIME_STEP));
  }
  
  if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
    agent_vz = (agent_fz) / ECM_ETA;
    agent_z += agent_vz * TIME_STEP;
    inc_pos_max = fmaxf(inc_pos_max, fabsf(agent_vz * TIME_STEP));
  }

  if (inc_pos_max > FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE) {
    printf("WARNING: ECM agent %d moved more than FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = %2.6f in a single time step (moved %2.6f). Consider reducing TIME_STEP or tweaking k_elast, d_dumping.\n", id, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE, inc_pos_max);
    //TODO: implement a fix (e.g., scale back the movement to the maximum allowed)
  }
  
  
  if (clamped_bx_pos == 1){
    agent_x = COORD_BOUNDARY_X_POS - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vx = DISP_RATE_BOUNDARY_X_POS;
    f_bx_pos = agent_fx;
    if (ALLOW_AGENT_SLIDING_X_POS == 0) {
      f_bx_pos_y = agent_fy;
      f_bx_pos_z = agent_fz;
      if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) { // this must be checked to avoid overwriting when agent is clamped to multiple boundaries
        agent_y = prev_agent_y;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_X_POS_Y) > 0.0) {
          agent_vy = DISP_RATE_BOUNDARY_PARALLEL_X_POS_Y;
          agent_y += agent_vy * TIME_STEP;                
        } else {
          agent_vy = 0.0;
        }
      }
      if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {            
        agent_z = prev_agent_z;
         if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_X_POS_Z) > 0.0) {
           agent_vz = DISP_RATE_BOUNDARY_PARALLEL_X_POS_Z;
           agent_z += agent_vz * TIME_STEP;
         } else {
           agent_vz = 0.0;
         }
      }
    }
  }
  if (clamped_bx_neg == 1){
    agent_x = COORD_BOUNDARY_X_NEG + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vx = DISP_RATE_BOUNDARY_X_NEG;
    f_bx_neg = agent_fx;
    if (ALLOW_AGENT_SLIDING_X_NEG == 0) {
      f_bx_neg_y = agent_fy;
      f_bx_neg_z = agent_fz;
      if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) { 
        agent_y = prev_agent_y;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_X_NEG_Y) > 0.0) {
          agent_vy = DISP_RATE_BOUNDARY_PARALLEL_X_NEG_Y;
          agent_y += agent_vy * TIME_STEP;
        }
        else {
          agent_vy = 0.0;
        }
      }
      if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
        agent_z = prev_agent_z;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_X_NEG_Z) > 0.0) {
          agent_vz = DISP_RATE_BOUNDARY_PARALLEL_X_NEG_Z;
          agent_z += agent_vz * TIME_STEP;
        }
        else {
          agent_vz = 0.0;
        }
      }
    }
  }
  if (clamped_by_pos == 1){
    agent_y = COORD_BOUNDARY_Y_POS - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vy = DISP_RATE_BOUNDARY_Y_POS;
    f_by_pos = agent_fy;
    if (ALLOW_AGENT_SLIDING_Y_POS == 0) {
        f_by_pos_x = agent_fx;
        f_by_pos_z = agent_fz;
        if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) { 
          agent_x = prev_agent_x;
          if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_Y_POS_X) > 0.0) {
            agent_vx = DISP_RATE_BOUNDARY_PARALLEL_Y_POS_X;
            agent_x += agent_vx * TIME_STEP;
          }
          else {
            agent_vx = 0.0;
          }
        }
        if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
          agent_z = prev_agent_z;
          if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_Y_POS_Z) > 0.0) {
            agent_vz = DISP_RATE_BOUNDARY_PARALLEL_Y_POS_Z;
            agent_z += agent_vz * TIME_STEP;
          }
          else {
            agent_vz = 0.0;
          }
        }
    }
  }
  if (clamped_by_neg == 1){
    agent_y = COORD_BOUNDARY_Y_NEG + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vy = DISP_RATE_BOUNDARY_Y_NEG;
    f_by_neg = agent_fy;
    if (ALLOW_AGENT_SLIDING_Y_NEG == 0) {
      f_by_neg_x = agent_fx;
      f_by_neg_z = agent_fz;
      if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) {
        agent_x = prev_agent_x;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_Y_NEG_X) > 0.0) {
          agent_vx = DISP_RATE_BOUNDARY_PARALLEL_Y_NEG_X;
          agent_x += agent_vx * TIME_STEP;
        }
        else {
          agent_vx = 0.0;
        }
      }
      if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
        agent_z = prev_agent_z;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_Y_NEG_Z) > 0.0) {
          agent_vz = DISP_RATE_BOUNDARY_PARALLEL_Y_NEG_Z;
          agent_z += agent_vz * TIME_STEP;
        }
        else {
           agent_vz = 0.0;
        }
      }
    }
  }
  if (clamped_bz_pos == 1){
    agent_z = COORD_BOUNDARY_Z_POS - FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vz = DISP_RATE_BOUNDARY_Z_POS;
    f_bz_pos = agent_fz;
    if (ALLOW_AGENT_SLIDING_Z_POS == 0) {
      f_bz_pos_x = agent_fx;
      f_bz_pos_y = agent_fy;
      if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) {
        agent_x = prev_agent_x;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_Z_POS_X) > 0.0) {
          agent_vx = DISP_RATE_BOUNDARY_PARALLEL_Z_POS_X;
          agent_x += agent_vx * TIME_STEP;
        }
        else {
          agent_vx = 0.0;
        }
      }
      if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) {
        agent_y = prev_agent_y;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_Z_POS_Y) > 0.0) {
          agent_vy = DISP_RATE_BOUNDARY_PARALLEL_Z_POS_Y;
          agent_y += agent_vy * TIME_STEP;
        }
        else {
          agent_vy = 0.0;
        }
      }
    }
  }
  if (clamped_bz_neg == 1){
    agent_z = COORD_BOUNDARY_Z_NEG + FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vz = DISP_RATE_BOUNDARY_Z_NEG;
    f_bz_neg = agent_fz;
    if (ALLOW_AGENT_SLIDING_Z_NEG == 0) {
      f_bz_neg_x = agent_fx;
      f_bz_neg_y = agent_fy;
      if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) {
        agent_x = prev_agent_x;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_Z_NEG_X) > 0.0) {
          agent_vx = DISP_RATE_BOUNDARY_PARALLEL_Z_NEG_X;
          agent_x += agent_vx * TIME_STEP;
        }
        else {
         agent_vx = 0.0;
        }
      }
      if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) {
        agent_y = prev_agent_y;
        if (fabsf(DISP_RATE_BOUNDARY_PARALLEL_Z_NEG_Y) > 0.0) {
          agent_vy = DISP_RATE_BOUNDARY_PARALLEL_Z_NEG_Y;
          agent_y += agent_vy * TIME_STEP;
        }
        else {
          agent_vy = 0.0;
        }
      }
    }
  }
  // Add forces from elastic boundaries (therefore, not clamped)
  if (fabsf(agent_x - COORD_BOUNDARY_X_POS) > fabsf(agent_x - COORD_BOUNDARY_X_NEG)) { //if closer to xpos
    f_bx_pos += agent_boundary_fx;// agent_boundary_fx will be 0 except for agents closer to boundaries
  }
  else {
    f_bx_neg += agent_boundary_fx;
  }

  if (fabsf(agent_y - COORD_BOUNDARY_Y_POS) > fabsf(agent_y - COORD_BOUNDARY_Y_NEG)) { //if closer to ypos
    f_by_pos += agent_boundary_fy; // agent_boundary_fy will be 0 except for agents closer to boundaries
  }
  else {
    f_by_neg += agent_boundary_fy;
  }

  if (fabsf(agent_z - COORD_BOUNDARY_Z_POS) > fabsf(agent_z - COORD_BOUNDARY_Z_NEG)) { //if closer to zpos
    f_bz_pos += agent_boundary_fz; // agent_boundary_fz will be 0 except for agents closer to boundaries
  }
  else {
    f_bz_neg += agent_boundary_fz;
  }
  
  //if (id == 9 || id == 10 || id == 13 || id == 14 || id == 25 || id == 26 || id == 29 || id == 30) {
  if (DEBUG_PRINTING == 1 && (id == 9 || id == 10 || id == 11 || id == 12)) {
      printf("agent %d position EN MEDIO (%2.4f, %2.4f, %2.4f) ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n", id, agent_x, agent_y, agent_z, COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);
  }
   
  boundPosition(id,agent_x, agent_y, agent_z, 
                clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg, 
                COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG,
                CLAMP_AGENT_TOUCHING_BOUNDARY_X_POS, CLAMP_AGENT_TOUCHING_BOUNDARY_X_NEG, CLAMP_AGENT_TOUCHING_BOUNDARY_Y_POS, CLAMP_AGENT_TOUCHING_BOUNDARY_Y_NEG, CLAMP_AGENT_TOUCHING_BOUNDARY_Z_POS, CLAMP_AGENT_TOUCHING_BOUNDARY_Z_NEG,
                FIBRE_NODE_BOUNDARY_EQUILIBRIUM_DISTANCE);

  
  //if (id == 9 || id == 10 || id == 13 || id == 14 || id == 25 || id == 26 || id == 29 || id == 30) {
  if (DEBUG_PRINTING == 1 && (id == 9 || id == 10 || id == 11 || id == 12)) {
    printf("agent %d position DESPUES (%2.4f, %2.4f, %2.4f) ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n", id, agent_x, agent_y, agent_z, COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);
  }
  
  //printf("ECM move ID: %d clamps after -> (%d, %d, %d, %d, %d, %d)\n", id, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);

  //Update the agents position and velocity
  FLAMEGPU->setVariable<float>("x",agent_x);
  FLAMEGPU->setVariable<float>("y",agent_y);
  FLAMEGPU->setVariable<float>("z",agent_z);
  FLAMEGPU->setVariable<float>("vx",agent_vx);
  FLAMEGPU->setVariable<float>("vy",agent_vy);
  FLAMEGPU->setVariable<float>("vz",agent_vz);
  FLAMEGPU->setVariable<uint8_t>("clamped_bx_pos", clamped_bx_pos);
  FLAMEGPU->setVariable<uint8_t>("clamped_bx_neg", clamped_bx_neg);
  FLAMEGPU->setVariable<uint8_t>("clamped_by_pos", clamped_by_pos);
  FLAMEGPU->setVariable<uint8_t>("clamped_by_neg", clamped_by_neg);
  FLAMEGPU->setVariable<uint8_t>("clamped_bz_pos", clamped_bz_pos);
  FLAMEGPU->setVariable<uint8_t>("clamped_bz_neg", clamped_bz_neg);
  FLAMEGPU->setVariable<float>("f_bx_pos", f_bx_pos);
  FLAMEGPU->setVariable<float>("f_bx_neg", f_bx_neg);
  FLAMEGPU->setVariable<float>("f_by_pos", f_by_pos);
  FLAMEGPU->setVariable<float>("f_by_neg", f_by_neg);
  FLAMEGPU->setVariable<float>("f_bz_pos", f_bz_pos);
  FLAMEGPU->setVariable<float>("f_bz_neg", f_bz_neg);
  FLAMEGPU->setVariable<float>("f_bx_pos_y", f_bx_pos_y);
  FLAMEGPU->setVariable<float>("f_bx_pos_z", f_bx_pos_z);
  FLAMEGPU->setVariable<float>("f_bx_neg_y", f_bx_neg_y);
  FLAMEGPU->setVariable<float>("f_bx_neg_z", f_bx_neg_z);
  FLAMEGPU->setVariable<float>("f_by_pos_x", f_by_pos_x);
  FLAMEGPU->setVariable<float>("f_by_pos_z", f_by_pos_z);
  FLAMEGPU->setVariable<float>("f_by_neg_x", f_by_neg_x);
  FLAMEGPU->setVariable<float>("f_by_neg_z", f_by_neg_z);
  FLAMEGPU->setVariable<float>("f_bz_pos_x", f_bz_pos_x);
  FLAMEGPU->setVariable<float>("f_bz_pos_y", f_bz_pos_y);
  FLAMEGPU->setVariable<float>("f_bz_neg_x", f_bz_neg_x);
  FLAMEGPU->setVariable<float>("f_bz_neg_y", f_bz_neg_y);

  return flamegpu::ALIVE;
}