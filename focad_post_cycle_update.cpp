FLAMEGPU_AGENT_FUNCTION(focad_post_cycle_update, flamegpu::MessageBucket, flamegpu::MessageNone) {
  const int agent_cell_id = FLAMEGPU->getVariable<int>("cell_id");
  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: must match main python model
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  bool found_parent = false;
  int parent_dead = 0;
  int parent_marked_for_removal = 0;
  int parent_just_divided = 0;
  int parent_daughter_id = -1;

  for (const auto &message : FLAMEGPU->message_in(agent_cell_id)) {
    found_parent = true;
    parent_dead = message.getVariable<int>("dead");
    parent_marked_for_removal = message.getVariable<int>("marked_for_removal");
    parent_just_divided = message.getVariable<int>("just_divided");
    parent_daughter_id = message.getVariable<int>("daughter_id");
    break;
  }

  if (!found_parent) {
      return flamegpu::DEAD;
  }

  if (parent_dead == 1) {
    return flamegpu::DEAD;
  }

  const float MAX_FOCAD_ARM_LENGTH = FLAMEGPU->environment.getProperty<float>("MAX_FOCAD_ARM_LENGTH");

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
    
    for (const auto &target_message : FLAMEGPU->message_in(target_cell_id)) {
      FLAMEGPU->setVariable<float>("x_c", target_message.getVariable<float>("x"));
      FLAMEGPU->setVariable<float>("y_c", target_message.getVariable<float>("y"));
      FLAMEGPU->setVariable<float>("z_c", target_message.getVariable<float>("z"));
      FLAMEGPU->setVariable<float>("orx", target_message.getVariable<float>("orx"));
      FLAMEGPU->setVariable<float>("ory", target_message.getVariable<float>("ory"));
      FLAMEGPU->setVariable<float>("orz", target_message.getVariable<float>("orz"));

      float best_r2 = 20 * (MAX_FOCAD_ARM_LENGTH * MAX_FOCAD_ARM_LENGTH); // 2 times just to give it some margin
      int best_anchor_id = -1;
      float best_xi = FLAMEGPU->getVariable<float>("x_i");
      float best_yi = FLAMEGPU->getVariable<float>("y_i");
      float best_zi = FLAMEGPU->getVariable<float>("z_i");

      for (unsigned int ai = 0; ai < N_ANCHOR_POINTS; ++ai) {
        const float ax = target_message.getVariable<float, N_ANCHOR_POINTS>("x_i", ai);
        const float ay = target_message.getVariable<float, N_ANCHOR_POINTS>("y_i", ai);
        const float az = target_message.getVariable<float, N_ANCHOR_POINTS>("z_i", ai);
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
      break;
    }
  }

  return flamegpu::ALIVE;
}
