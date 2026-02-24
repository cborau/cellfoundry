FLAMEGPU_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
  x /= divisor;
  y /= divisor;
  z /= divisor;
}
FLAMEGPU_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}
FLAMEGPU_AGENT_FUNCTION(cell_cycle, flamegpu::MessageNone, flamegpu::MessageNone) {
    
  int id = FLAMEGPU->getVariable<int>("id");
  
  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  // Agent orientation
  float agent_orx = FLAMEGPU->getVariable<float>("orx");
  float agent_ory = FLAMEGPU->getVariable<float>("ory");
  float agent_orz = FLAMEGPU->getVariable<float>("orz");
  
  float agent_k_elast = FLAMEGPU->getVariable<float>("k_elast");
  float agent_d_dumping = FLAMEGPU->getVariable<float>("d_dumping");
  float agent_radius = FLAMEGPU->getVariable<float>("radius");
  int agent_completed_cycles = FLAMEGPU->getVariable<int>("completed_cycles");
  
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float CELL_RADIUS = FLAMEGPU->environment.getProperty<float>("CELL_RADIUS");
  const float CELL_CYCLE_DURATION = FLAMEGPU->environment.getProperty<float>("CELL_CYCLE_DURATION");
  const float CYCLE_PHASE_G1_START = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_G1_START");
  const float CYCLE_PHASE_S_START = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_S_START");
  const float CYCLE_PHASE_G2_START = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_G2_START");
  const float CYCLE_PHASE_M_START = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_M_START");
  const float CYCLE_PHASE_G1_DURATION = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_G1_DURATION");
  const float CYCLE_PHASE_M_DURATION = FLAMEGPU->environment.getProperty<float>("CYCLE_PHASE_M_DURATION");
  
  float agent_clock = FLAMEGPU->getVariable<float>("clock");
  agent_clock += TIME_STEP;
  FLAMEGPU->setVariable<float>("clock", agent_clock);
  

  if ((agent_clock >= CYCLE_PHASE_G1_START) && (agent_clock < CYCLE_PHASE_S_START)) {
    FLAMEGPU->setVariable<int>("cycle_phase", 1);
  }
  if ((agent_clock >= CYCLE_PHASE_S_START) && (agent_clock < CYCLE_PHASE_G2_START)) {
    FLAMEGPU->setVariable<int>("cycle_phase", 2);
  }
  if ((agent_clock >= CYCLE_PHASE_G2_START) && (agent_clock < CYCLE_PHASE_M_START)) {
    FLAMEGPU->setVariable<int>("cycle_phase", 3);
  }
  
  // Increasing probability of division with time in M phase
  if (agent_clock >= CYCLE_PHASE_M_START) {
    float time_in_phase = agent_clock - CYCLE_PHASE_M_START;
    float phase_n_steps = CYCLE_PHASE_M_DURATION / TIME_STEP; 
    float p_step = 1 / phase_n_steps;
    float current_phase_step = time_in_phase / TIME_STEP;
    float p_division = p_step / ((phase_n_steps - current_phase_step + 1) / phase_n_steps); //actual probability in current step.
    float p = FLAMEGPU->random.uniform<float>(0.0,1.0);
    FLAMEGPU->setVariable<int>("cycle_phase", 4);
    if (agent_clock > CELL_CYCLE_DURATION) { // this should never happen as the cell should divide first
      agent_clock -= CELL_CYCLE_DURATION;      
      FLAMEGPU->setVariable<float>("clock", agent_clock);  
    }
    
    if (p < p_division) {
      
      // Division occurs      
      FLAMEGPU->setVariable<float>("x", agent_x + (agent_orx * agent_radius / 2));
      FLAMEGPU->setVariable<float>("y", agent_y + (agent_ory * agent_radius / 2));
      FLAMEGPU->setVariable<float>("z", agent_z + (agent_orz * agent_radius / 2));
      FLAMEGPU->setVariable<float>("radius", agent_radius / 2); 
      agent_completed_cycles += 1;
      FLAMEGPU->setVariable<int>("completed_cycles", agent_completed_cycles);
      printf("Cell [id: %d] DIVIDES at t: %g [time in phase: %g]- completed cycles: %d \n", id, agent_clock, time_in_phase, agent_completed_cycles);
      printf("Cell [id: %d] PROBS p_step: %g, phase_n_steps:%g, current_phase_step: %g, p_division: %g \n", id, p_step, phase_n_steps, current_phase_step, p_division);
      FLAMEGPU->setVariable<float>("clock", 0.0 + FLAMEGPU->random.uniform<float>(0.0,0.1) * CYCLE_PHASE_G1_DURATION); //add some randomness to the clock
      FLAMEGPU->setVariable<int>("cycle_phase", 1);  
      
      
      // New cell agent
      float rand_dir_x = FLAMEGPU->random.uniform<float>(-1.0,1.0);
      float rand_dir_y = FLAMEGPU->random.uniform<float>(-1.0,1.0);
      float rand_dir_z = FLAMEGPU->random.uniform<float>(-1.0,1.0); 
      float rand_dir_length = vec3Length(rand_dir_x,rand_dir_y,rand_dir_z);  
      vec3Div(rand_dir_x, rand_dir_y, rand_dir_z, rand_dir_length); 
      FLAMEGPU->agent_out.setVariable<float>("x", agent_x - (agent_orx * agent_radius / 2));
      FLAMEGPU->agent_out.setVariable<float>("y", agent_y - (agent_ory * agent_radius / 2));
      FLAMEGPU->agent_out.setVariable<float>("z", agent_z - (agent_orz * agent_radius / 2));
      FLAMEGPU->agent_out.setVariable<float>("orx", rand_dir_x);
      FLAMEGPU->agent_out.setVariable<float>("ory", rand_dir_y);
      FLAMEGPU->agent_out.setVariable<float>("orz", rand_dir_z);
      FLAMEGPU->agent_out.setVariable<float>("k_elast", agent_k_elast);
      FLAMEGPU->agent_out.setVariable<float>("d_dumping", agent_d_dumping); 
      FLAMEGPU->agent_out.setVariable<float>("radius", agent_radius / 2);
      FLAMEGPU->agent_out.setVariable<float>("clock", 0.0 + FLAMEGPU->random.uniform<float>(0.0,0.1) * CYCLE_PHASE_G1_DURATION);
      FLAMEGPU->agent_out.setVariable<int>("cycle_phase", 1);
      FLAMEGPU->agent_out.setVariable<int>("completed_cycles", 0);
    }       
  } else {
    agent_radius += ((CELL_RADIUS / 2) / CYCLE_PHASE_M_START) * TIME_STEP;  
    FLAMEGPU->setVariable<float>("radius", agent_radius); 
  }
  return flamegpu::ALIVE;  
}