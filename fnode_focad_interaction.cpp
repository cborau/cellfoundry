// Mirror function to focad_fnode_interaction.
// Reads the force from the focal adhesion and adds it to the current FNODE.
// The calling FNODE agent checks the closest FOCAD agent, and if that FOCAD:
//   - has fnode_id == this FNODE id
//   - is attached == 1
//   - is active   == 1
// then the FNODE adds the adhesion force (fx,fy,fz) stored in the FOCAD.
//
// Notes:
// - This assumes focad_fnode_interaction has already stored (fx,fy,fz) in the FOCAD,
//   where (fx,fy,fz) is the force that should be applied to the FNODE.
// - The force direction is already "pull FNODE towards xi".
FLAMEGPU_AGENT_FUNCTION(fnode_focad_interaction, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  // -------------------------
  // Get FNODE agent variables (agent calling the function)
  // -------------------------
  const int   agent_id = FLAMEGPU->getVariable<int>("id");
  const float agent_x  = FLAMEGPU->getVariable<float>("x");
  const float agent_y  = FLAMEGPU->getVariable<float>("y");
  const float agent_z  = FLAMEGPU->getVariable<float>("z");
  float agent_fx = FLAMEGPU->getVariable<float>("fx");
  float agent_fy = FLAMEGPU->getVariable<float>("fy");
  float agent_fz = FLAMEGPU->getVariable<float>("fz");

  const float MAX_SEARCH_RADIUS_FOCAD = FLAMEGPU->environment.getProperty<float>("MAX_SEARCH_RADIUS_FOCAD");
  const float max_r2 = MAX_SEARCH_RADIUS_FOCAD * MAX_SEARCH_RADIUS_FOCAD;

  // -------------------------
  // Find closest relevant FOCAD
  // -------------------------
  float best_r2 = max_r2;

  float best_fx = 0.0f;
  float best_fy = 0.0f;
  float best_fz = 0.0f;

  int found = 0;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    // Basic spatial info (for "closest" criterion)
    const float message_x = message.getVariable<float>("x");
    const float message_y = message.getVariable<float>("y");
    const float message_z = message.getVariable<float>("z");

    const float dx = message_x - agent_x;
    const float dy = message_y - agent_y;
    const float dz = message_z - agent_z;
    const float r2 = dx*dx + dy*dy + dz*dz;

    // Filter: only adhesions that belong to this node and are active/attached
    const int     message_fnode_id = message.getVariable<int>("fnode_id");
    const int message_attached = message.getVariable<int>("attached");
    const uint8_t message_active   = message.getVariable<uint8_t>("active");

    if (message_fnode_id != agent_id) continue;
    if (!message_attached) continue;
    if (!message_active) continue;

    // Choose closest relevant FOCAD
    if (r2 < best_r2) {
      best_r2 = r2;
      best_fx = message.getVariable<float>("fx");
      best_fy = message.getVariable<float>("fy");
      best_fz = message.getVariable<float>("fz");
      found = 1;
    }
  }

  if (found == 1) {
    agent_fx += best_fx;
    agent_fy += best_fy;
    agent_fz += best_fz;
    // printf("fnode_focad_interaction -- FNODE %d found relevant FOCAD with force (%.4f, %.4f, %.4f)\n", agent_id, best_fx, best_fy, best_fz);
  }

  FLAMEGPU->setVariable<float>("fx", agent_fx);
  FLAMEGPU->setVariable<float>("fy", agent_fy);
  FLAMEGPU->setVariable<float>("fz", agent_fz);

  return flamegpu::ALIVE;
}