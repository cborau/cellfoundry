/**
 * fnode_apply_remodel_updates
 *
 * Purpose:
 *   Apply remodeling topology updates and optionally remove terminally degraded nodes.
 *
 * Inputs:
 *   - FNODE connectivity arrays
 *   - Spatial FNODE messages (id, x,y,z, closest_fnode_id)
 *
 * Outputs:
 *   - Updated `linked_nodes` / `equilibrium_distance`
 *   - Updated `connectivity_count`
 *   - Returns DEAD when `marked_for_removal` is set
 */
FLAMEGPU_AGENT_FUNCTION(fnode_apply_remodel_updates, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  const uint32_t INCLUDE_NETWORK_REMODELING = FLAMEGPU->environment.getProperty<uint32_t>("INCLUDE_NETWORK_REMODELING");
  if (INCLUDE_NETWORK_REMODELING == 0) {
    return flamegpu::ALIVE;
  }

  const uint8_t MAX_CONNECTIVITY = 8;      // must match model.py
  const int LOCAL_CACHE_SIZE = 128;

  const int id = FLAMEGPU->getVariable<int>("id");
  const int self_marked_for_removal = FLAMEGPU->getVariable<int>("marked_for_removal");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  float linked_nodes[MAX_CONNECTIVITY] = {};
  float equilibrium_distance[MAX_CONNECTIVITY] = {};
  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
    linked_nodes[i] = FLAMEGPU->getVariable<float, MAX_CONNECTIVITY>("linked_nodes", i);
    equilibrium_distance[i] = FLAMEGPU->getVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i);
  }

  // Build local cache of candidate newborn links
  int newborn_ids[LOCAL_CACHE_SIZE] = {};
  float newborn_dist[LOCAL_CACHE_SIZE] = {};
  int newborn_count = 0;

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const int mid = message.getVariable<int>("id");

    const int closest_fnode_id = message.getVariable<int>("closest_fnode_id");
    if (closest_fnode_id == id && mid != id && newborn_count < LOCAL_CACHE_SIZE) {
      float dx = message.getVariable<float>("x") - agent_x;
      float dy = message.getVariable<float>("y") - agent_y;
      float dz = message.getVariable<float>("z") - agent_z;
      newborn_ids[newborn_count] = mid;
      newborn_dist[newborn_count] = fmaxf(1e-6f, sqrtf(dx * dx + dy * dy + dz * dz));
      newborn_count++;
    }
  }

  // Add reciprocal links for newborn nodes that selected this node as closest_fnode_id
  // Skip birth-link updates if caller is being removed this step.
  if (self_marked_for_removal == 0) {
    for (int q = 0; q < newborn_count; q++) {
      const int new_id = newborn_ids[q];
      if (new_id < 0) {
        continue;
      }

      int already_present = 0;
      for (int i = 0; i < MAX_CONNECTIVITY; i++) {
        const int ln = static_cast<int>(linked_nodes[i]);
        if (ln == new_id) {
          already_present = 1;
          break;
        }
      }
      if (already_present) {
        continue;
      }

      int first_free_slot = -1;
      for (int i = 0; i < MAX_CONNECTIVITY; i++) {
        const int ln = static_cast<int>(linked_nodes[i]);
        if (ln < 0) {
          first_free_slot = i;
          break;  
        }
      }
      if (first_free_slot >= 0) {
        linked_nodes[first_free_slot] = static_cast<float>(new_id);
        equilibrium_distance[first_free_slot] = newborn_dist[q];
        break;  // consume at most one free slot per pass
      }
    }
  }

  uint8_t connectivity_count = 0;
  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
    if (linked_nodes[i] >= 0.0f) {
      connectivity_count += 1;
    }
  }

  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
    FLAMEGPU->setVariable<float, MAX_CONNECTIVITY>("linked_nodes", i, linked_nodes[i]);
    FLAMEGPU->setVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i, equilibrium_distance[i]);
  }
  FLAMEGPU->setVariable<uint8_t>("connectivity_count", connectivity_count);
  FLAMEGPU->setVariable<int>("closest_fnode_id", -1);

  if (self_marked_for_removal == 1) {
    return flamegpu::DEAD;
  }

  return flamegpu::ALIVE;
}
