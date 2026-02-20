/**
 * bcorner_output_location_data
 *
 * Purpose:
 *   Publish BCORNER identifiers and coordinates to spatial messages.
 *
 * Inputs:
 *   - Agent variables: id, x, y, z
 *
 * Outputs:
 *   - MessageSpatial3D payload for downstream consumers
 */
FLAMEGPU_AGENT_FUNCTION(bcorner_output_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  return flamegpu::ALIVE;
}