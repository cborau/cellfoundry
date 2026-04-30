/**
 * lumen_spatial_location_data
 *
 * Purpose:
 *   Broadcast LUMEN agent position and radius over a spatial message list so that
 *   other agents (LUMEN, CELL, ECM) can query nearby LUMEN droplets.
 *
 * Inputs:
 *   - LUMEN variables: id, x, y, z, radius
 *
 * Outputs:
 *   - MessageSpatial3D record for nearby agent queries
 */
FLAMEGPU_AGENT_FUNCTION(lumen_spatial_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("radius", FLAMEGPU->getVariable<float>("radius"));
  return flamegpu::ALIVE;
}
