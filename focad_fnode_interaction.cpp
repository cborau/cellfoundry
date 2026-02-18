// Mirror function to fnode_focad_interaction (calling order should be:
// 1) focad_fnode_interaction and then 2) fnode_focad_interaction).
// If an FNODE is close enough to the calling FOCAD agent, they get attached.
// Then, computes and stores the adhesion force that will be later transmitted
// to the corresponding fnode (in a different function).
FLAMEGPU_AGENT_FUNCTION(focad_fnode_interaction, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  // -------------------------
  // Read FOCAD variables
  // -------------------------
  const int agent_focad_id = FLAMEGPU->getVariable<int>("id");
  const int agent_cell_id  = FLAMEGPU->getVariable<int>("cell_id");

  // FOCAD position (used as spatial query center). After attachment (it it happens) this is set to FNODE pos.
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");

  // Nucleus anchor point for this adhesion (xi on nucleus surface)
  const float agent_x_i = FLAMEGPU->getVariable<float>("x_i");
  const float agent_y_i = FLAMEGPU->getVariable<float>("y_i");
  const float agent_z_i = FLAMEGPU->getVariable<float>("z_i");

  // Mechanics parameters/state
  float  agent_rest_length_0 = FLAMEGPU->getVariable<float>("rest_length_0");  // L0 at creation
  float  agent_rest_length = FLAMEGPU->getVariable<float>("rest_length");    // L(t)
  const float agent_k_fa = FLAMEGPU->getVariable<float>("k_fa");
  const float agent_f_max = FLAMEGPU->getVariable<float>("f_max");         // WARNING: 0 means "no cap" 

  const uint8_t agent_active = FLAMEGPU->getVariable<uint8_t>("active");     // actomyosin engaged
  const float   agent_v_c = FLAMEGPU->getVariable<float>("v_c");          // um/s rest-length shortening
  uint8_t agent_attached  = FLAMEGPU->getVariable<uint8_t>("attached");
  int agent_fnode_id = FLAMEGPU->getVariable<int>("fnode_id");

  float agent_age = FLAMEGPU->getVariable<float>("age");

  // Outputs (force stored on FOCAD to be applied to FNODE later)
  float agent_fx = 0.0f;
  float agent_fy = 0.0f;
  float agent_fz = 0.0f;

  // -------------------------
  // Read environment
  // -------------------------
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float MAX_SEARCH_RADIUS_FOCAD = FLAMEGPU->environment.getProperty<float>("MAX_SEARCH_RADIUS_FOCAD");

  // Prevent L->0 forever
  const float FOCAD_MIN_REST_LENGTH = FLAMEGPU->environment.getProperty<float>("FOCAD_MIN_REST_LENGTH");

  // -------------------------
  // 1) Attachment: if not attached, find closest FNODE in search radius
  // -------------------------
  float message_x = 0.0f;
  float message_y = 0.0f;
  float message_z = 0.0f;   // FNODE position (to be determined)
  if (agent_attached == 0) {
    float best_r2 = MAX_SEARCH_RADIUS_FOCAD * MAX_SEARCH_RADIUS_FOCAD;
    int   best_id = -1;
    float best_x = 0.0f;
    float best_y = 0.0f;
    float best_z = 0.0f;

    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
      const float nx = message.getVariable<float>("x");
      const float ny = message.getVariable<float>("y");
      const float nz = message.getVariable<float>("z");
      const int   nid = message.getVariable<int>("id");

      const float dx = nx - agent_x;
      const float dy = ny - agent_y;
      const float dz = nz - agent_z;
      const float r2 = dx*dx + dy*dy + dz*dz;

      if (r2 < best_r2) {
        best_r2 = r2;
        best_id = nid;
        best_x = nx;
        best_y = ny;
        best_z = nz;
      }
    }

    if (best_id >= 0) {
      // Attach to closest node
      agent_attached = 1;
      agent_fnode_id = best_id;
      message_x = best_x;
      message_y = best_y; 
      message_z = best_z;
      agent_x = message_x;
      agent_y = message_y;
      agent_z = message_z;

      // Initialize rest lengths so the adhesion starts unstrained even if yi is far from nucleus
      const float dx0 = message_x - agent_x_i;
      const float dy0 = message_y - agent_y_i;
      const float dz0 = message_z - agent_z_i;
      const float ell0 = sqrtf(dx0*dx0 + dy0*dy0 + dz0*dz0);

      agent_rest_length_0 = ell0;
      agent_rest_length   = ell0;
      agent_age = 0.0f;  // reset age on attachment
      printf("focad_fnode -- FOCAD %d (cell %d) attached to FNODE %d at distance %.4f um with initial rest length %.4f um\n", agent_focad_id, agent_cell_id, agent_fnode_id, sqrtf(best_r2), ell0);
    } else {
      // Not attached and no node found, keep force = 0 and exit early
      printf("focad_fnode -- FOCAD %d (cell %d) not attached, no FNODE found within search radius.\n", agent_focad_id, agent_cell_id);
      FLAMEGPU->setVariable<uint8_t>("attached", agent_attached);
      FLAMEGPU->setVariable<float>("fx", 0.0f);
      FLAMEGPU->setVariable<float>("fy", 0.0f);
      FLAMEGPU->setVariable<float>("fz", 0.0f);
      FLAMEGPU->setVariable<float>("age", agent_age);
      // keep x,y,z as-is
      return flamegpu::ALIVE;
    }
  } else {
    // Already attached: FOCAD position == FNODE position.
    printf("focad_fnode -- FOCAD %d (cell %d) already attached to FNODE %d at position (%.4f, %.4f, %.4f)\n", agent_focad_id, agent_cell_id, agent_fnode_id, agent_x, agent_y, agent_z);
    message_x = agent_x; 
    message_y = agent_y; 
    message_z = agent_z;
  }

  // -------------------------
  // 2) Contractility: shorten rest length if active
  // -------------------------
  if (agent_attached && agent_active) {
    float rl = agent_rest_length - agent_v_c * TIME_STEP;    
    agent_rest_length = fmaxf(FOCAD_MIN_REST_LENGTH, agent_rest_length - agent_v_c * TIME_STEP);
    if (rl < FOCAD_MIN_REST_LENGTH) {
      printf("focad_fnode -- FOCAD %d (cell %d) rest length reached minimum value of %.4f um and cannot shorten further.\n", agent_focad_id, agent_cell_id, FOCAD_MIN_REST_LENGTH);
    }
  }

  // -------------------------
  // 3) Compute adhesion traction (tension-only spring with optional cap)
  //    xi on nucleus surface, message_i at FNODE position
  // -------------------------

  const float dx = message_x - agent_x_i;
  const float dy = message_y - agent_y_i;
  const float dz = message_z - agent_z_i;
  const float ell = sqrtf(dx*dx + dy*dy + dz*dz);  

  // extension = max(0, ell - L)
  const float ext = fmaxf(0.0f, ell - agent_rest_length);

  printf("focad_fnode -- FOCAD %d, message_pos (%.4f, %.4f, %.4f) um, agent_rest_length = %.4f um, ell0 = %.4f um, ell = %.4f um, diff = %.4f um, extension = %.4f um\n", agent_focad_id, message_x, message_y, message_z, agent_rest_length, agent_rest_length_0, ell, ell - agent_rest_length, ext);

  // |F| = k_fa * extension
  float Fmag = agent_k_fa * ext;

  // Optional cap to avoid runaway
  if (agent_f_max > 0.0f) {
    Fmag = fminf(Fmag, agent_f_max);
  }

  // Direction from xi -> message_i (avoid divide by 0)
  float inv_ell = 0.0f;
  if (ell > 1e-12f) inv_ell = 1.0f / ell;

  const float ux = dx * inv_ell;
  const float uy = dy * inv_ell;
  const float uz = dz * inv_ell;

  // Important sign convention:
  // - Tension wants to reduce ell, pulling message_i toward xi.
  // - Force ON THE FNODE (at message_i) points from message_i toward xi, i.e. -(xi->message_i) direction.
  agent_fx = -Fmag * ux;
  agent_fy = -Fmag * uy;
  agent_fz = -Fmag * uz;

  // -------------------------
  // 4) Update bookkeeping
  // -------------------------
  if (agent_attached) {
    agent_age += TIME_STEP;
  }

  // -------------------------
  // Write back variables
  // -------------------------
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);

  FLAMEGPU->setVariable<float>("rest_length_0", agent_rest_length_0);
  FLAMEGPU->setVariable<float>("rest_length", agent_rest_length);

  FLAMEGPU->setVariable<uint8_t>("attached", agent_attached);
  FLAMEGPU->setVariable<int>("fnode_id", agent_fnode_id);

  FLAMEGPU->setVariable<float>("fx", agent_fx);
  FLAMEGPU->setVariable<float>("fy", agent_fy);
  FLAMEGPU->setVariable<float>("fz", agent_fz);
  printf("focad_fnode -- FOCAD %d (cell %d) force on FNODE %d: (%.3f, %.3f, %.3f) nN\n", agent_focad_id, agent_cell_id, agent_fnode_id, agent_fx, agent_fy, agent_fz);

  FLAMEGPU->setVariable<float>("age", agent_age);

  return flamegpu::ALIVE;
}
