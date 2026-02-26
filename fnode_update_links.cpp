/**
 * fnode_update_links
 *
 * Purpose:
 *   Update FNODE link list using bucket messages keyed by linked node id.
 *   If a linked node has no bucket message (e.g., removed), clear that link.
 *
 * Inputs:
 *   - FNODE bucket messages keyed by id
 *   - FNODE linked_nodes / equilibrium_distance
 *
 * Outputs:
 *   - Updated linked_nodes
 *   - Updated connectivity_count
 */
FLAMEGPU_AGENT_FUNCTION(fnode_update_links, flamegpu::MessageBucket, flamegpu::MessageNone) {
  const uint8_t MAX_CONNECTIVITY = 8; // must match model.py

  float linked_nodes[MAX_CONNECTIVITY] = {};
  float equilibrium_distance[MAX_CONNECTIVITY] = {};
  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
    linked_nodes[i] = FLAMEGPU->getVariable<float, MAX_CONNECTIVITY>("linked_nodes", i);
    equilibrium_distance[i] = FLAMEGPU->getVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i);
  }

  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
    const int linked_id = static_cast<int>(linked_nodes[i]);
    if (linked_id < 0) {
      continue;
    }

    int found_message = 0;
    int neighbor_marked_for_removal = 0;
    for (const auto &message : FLAMEGPU->message_in(linked_id)) {
      found_message = 1;
      neighbor_marked_for_removal = message.getVariable<int>("marked_for_removal");
      break;
    }

    if (found_message == 0 || neighbor_marked_for_removal == 1) {
      linked_nodes[i] = -1.0f;
    }
  }

  uint8_t connectivity_count = 0;
  for (int i = 0; i < MAX_CONNECTIVITY; i++) {
    FLAMEGPU->setVariable<float, MAX_CONNECTIVITY>("linked_nodes", i, linked_nodes[i]);
    FLAMEGPU->setVariable<float, MAX_CONNECTIVITY>("equilibrium_distance", i, equilibrium_distance[i]);
    if (linked_nodes[i] >= 0.0f) {
      connectivity_count += 1;
    }
  }
  FLAMEGPU->setVariable<uint8_t>("connectivity_count", connectivity_count);

  return flamegpu::ALIVE;
}
