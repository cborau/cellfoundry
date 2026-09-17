FLAMEGPU_AGENT_FUNCTION(cell_read_markers, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const int id = FLAMEGPU->getVariable<int>("id");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in) {
        if (message.getVariable<int>("owner_cell_id") == id) ++count;
    }
    FLAMEGPU->setVariable<int>("own_marker_count", count);  // Also reset when messages are empty.
    return flamegpu::ALIVE;
}
