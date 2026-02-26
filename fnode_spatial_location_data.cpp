/**
 * fnode_spatial_location_data
 *
 * Purpose:
 *   Broadcast FNODE position for spatial proximity queries.
 *
 * Inputs:
 *   - FNODE variables: id, x, y, z
 *
 * Outputs:
 *   - MessageSpatial3D payload used by FNODE/FOCAD interaction kernels
 */
FLAMEGPU_AGENT_FUNCTION(fnode_spatial_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<uint8_t>("connectivity_count", FLAMEGPU->getVariable<uint8_t>("connectivity_count"));
  FLAMEGPU->message_out.setVariable<int>("closest_fnode_id", FLAMEGPU->getVariable<int>("closest_fnode_id"));
  FLAMEGPU->message_out.setVariable<int>("marked_for_removal", FLAMEGPU->getVariable<int>("marked_for_removal"));

  return flamegpu::ALIVE;
}