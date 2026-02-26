/**
 * focad_post_cycle_update
 *
 * Purpose:
 *   Update FOCAD anchor association after cell division, switching anchor points between parent and daughter cells.
 *   Ensures spatial and orientation variables are updated for correct cell association.
 *
 * Inputs:
 *   - FOCAD agent variables: cell_id, anchor_id, spatial coordinates
 *   - MessageBucket: parent cell state
 *   - Environment: N_ANCHOR_POINTS
 *
 * Outputs:
 *   - Updated anchor_id, spatial variables, cell association
 *
 * Notes:
 *   - Uses a two-pass loop to select the correct target message, avoiding duplicated anchor update code.
 *   - Pass 0: For parent cell, require just_divided message (ensures correct anchor after division).
 *   - Pass 1: Fallback, accept any available message if none found in pass 0.
 */
FLAMEGPU_AGENT_FUNCTION(focad_post_cycle_update, flamegpu::MessageBucket, flamegpu::MessageNone) {
  const int agent_cell_id = FLAMEGPU->getVariable<int>("cell_id");
  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: must match main python model
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  int parent_found = 0;
  int parent_dead = 0;
  int parent_marked_for_removal = 0;
  int parent_just_divided = 0;
  int parent_daughter_id = -1;

  for (const auto &message : FLAMEGPU->message_in(agent_cell_id)) {
    if (parent_found == 0) {
      parent_found = 1;
      parent_dead = message.getVariable<int>("dead");
      parent_marked_for_removal = message.getVariable<int>("marked_for_removal");
      parent_just_divided = message.getVariable<int>("just_divided");
      parent_daughter_id = message.getVariable<int>("daughter_id");
    }

    const int message_just_divided = message.getVariable<int>("just_divided");
    const int message_daughter_id = message.getVariable<int>("daughter_id");
    if (message_just_divided == 1 && message_daughter_id > 0) {
      parent_dead = message.getVariable<int>("dead");
      parent_marked_for_removal = message.getVariable<int>("marked_for_removal");
      parent_just_divided = message_just_divided;
      parent_daughter_id = message_daughter_id;
      break;
    }
  }

  if (parent_found == 0) {
      return flamegpu::DEAD;
  }

  if (parent_dead == 1) {
    return flamegpu::DEAD;
  }

  if (parent_just_divided == 1 && parent_daughter_id > 0) {
    const float switch_draw = FLAMEGPU->random.uniform<float>(0.0f, 1.0f);
    const int switch_to_daughter = (switch_draw < 0.5f) ? 1 : 0;
    const int target_cell_id = (switch_to_daughter == 1) ? parent_daughter_id : agent_cell_id;
    printf("FOCAD [cell_id: %d] just_divided parent [id: %d, daughter_id: %d], switch_draw: %.3f, switch_to_daughter: %d \n", agent_cell_id, agent_cell_id, parent_daughter_id, switch_draw, switch_to_daughter);

    if (switch_to_daughter == 1) { // This FOCAD will now be associated with the daughter cell, so we update the cell_id and reset anchor_id to detach from the parent.
      FLAMEGPU->setVariable<int>("cell_id", parent_daughter_id);
      FLAMEGPU->setVariable<int>("anchor_id", -1);
      FLAMEGPU->setVariable<int>("attached", 0);
      FLAMEGPU->setVariable<uint8_t>("active", 1);
      FLAMEGPU->setVariable<int>("fnode_id", -1);
      FLAMEGPU->setVariable<float>("fx", 0.0f);
      FLAMEGPU->setVariable<float>("fy", 0.0f);
      FLAMEGPU->setVariable<float>("fz", 0.0f);
    }
    
    int found_target_message = 0;
    // Parent cell broadcasts its state twice (cell_bucket_location_data is called in two different layers) 
    // just_divided=1 is required here to ensure we get the correct post-division state of the parent cell, then accepting any message for the daughter.
    const int pass_count = (target_cell_id == agent_cell_id) ? 2 : 1;
    for (int pass = 0; pass < pass_count && found_target_message == 0; ++pass) {
      const int require_just_divided = (pass == 0 && target_cell_id == agent_cell_id) ? 1 : 0;
      for (const auto &target_message : FLAMEGPU->message_in(target_cell_id)) {
        if (require_just_divided == 1 && target_message.getVariable<int>("just_divided") != 1) {
          continue;
        }

        const float target_x = target_message.getVariable<float>("x");
        const float target_y = target_message.getVariable<float>("y");
        const float target_z = target_message.getVariable<float>("z");
        FLAMEGPU->setVariable<float>("x_c", target_x);
        FLAMEGPU->setVariable<float>("y_c", target_y);
        FLAMEGPU->setVariable<float>("z_c", target_z);
        FLAMEGPU->setVariable<float>("orx", target_message.getVariable<float>("orx"));
        FLAMEGPU->setVariable<float>("ory", target_message.getVariable<float>("ory"));
        FLAMEGPU->setVariable<float>("orz", target_message.getVariable<float>("orz"));

        const float target_nucleus_radius = target_message.getVariable<float>("nucleus_radius");
        const float target_eps_xx = target_message.getVariable<float>("eps_xx");
        const float target_eps_yy = target_message.getVariable<float>("eps_yy");
        const float target_eps_zz = target_message.getVariable<float>("eps_zz");
        const float target_eps_xy = target_message.getVariable<float>("eps_xy");
        const float target_eps_xz = target_message.getVariable<float>("eps_xz");
        const float target_eps_yz = target_message.getVariable<float>("eps_yz");

        float best_r2 = 1e30f;
        int best_anchor_id = -1;
        float best_xi = FLAMEGPU->getVariable<float>("x_i");
        float best_yi = FLAMEGPU->getVariable<float>("y_i");
        float best_zi = FLAMEGPU->getVariable<float>("z_i");

        for (unsigned int ai = 0; ai < N_ANCHOR_POINTS; ++ai) {
          const float ux = target_message.getVariable<float, N_ANCHOR_POINTS>("u_ref_x_i", ai);
          const float uy = target_message.getVariable<float, N_ANCHOR_POINTS>("u_ref_y_i", ai);
          const float uz = target_message.getVariable<float, N_ANCHOR_POINTS>("u_ref_z_i", ai);

          const float dux = ux + target_eps_xx * ux + target_eps_xy * uy + target_eps_xz * uz;
          const float duy = uy + target_eps_xy * ux + target_eps_yy * uy + target_eps_yz * uz;
          const float duz = uz + target_eps_xz * ux + target_eps_yz * uy + target_eps_zz * uz;

          const float ax = target_x + target_nucleus_radius * dux;
          const float ay = target_y + target_nucleus_radius * duy;
          const float az = target_z + target_nucleus_radius * duz;
          const float dx = ax - agent_x;
          const float dy = ay - agent_y;
          const float dz = az - agent_z;
          const float r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < best_r2) {
            best_r2 = r2;
            best_anchor_id = static_cast<int>(ai);
            best_xi = ax;
            best_yi = ay;
            best_zi = az;
          }
        }

        if (best_anchor_id >= 0) {
          FLAMEGPU->setVariable<int>("anchor_id", best_anchor_id);
          FLAMEGPU->setVariable<float>("x_i", best_xi);
          FLAMEGPU->setVariable<float>("y_i", best_yi);
          FLAMEGPU->setVariable<float>("z_i", best_zi);
          printf("FOCAD [cell_id: %d] SWITCHED to %s cell [id: %d] \n", agent_cell_id, (switch_to_daughter == 1) ? "DAUGHTER" : "PARENT", target_cell_id);
        }
        found_target_message = 1;
        break;
      }
    }
  }

  return flamegpu::ALIVE;
}
