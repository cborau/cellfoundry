/**
 * focad_spatial_location_data
 *
 * Purpose:
 *   Broadcast active adhesion position/force state for local spatial interaction queries.
 *
 * Inputs:
 *   - FOCAD variables: id, x,y,z, fx,fy,fz, fnode_id, attached, active
 *
 * Outputs:
 *   - MessageSpatial3D payload consumed by FNODE-side force transfer
 */
FLAMEGPU_AGENT_FUNCTION(focad_spatial_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
  FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
  FLAMEGPU->message_out.setVariable<float>("fz", FLAMEGPU->getVariable<float>("fz"));
  FLAMEGPU->message_out.setVariable<int>("fnode_id", FLAMEGPU->getVariable<int>("fnode_id"));
  FLAMEGPU->message_out.setVariable<int>("attached", FLAMEGPU->getVariable<int>("attached"));
  FLAMEGPU->message_out.setVariable<uint8_t>("active", FLAMEGPU->getVariable<uint8_t>("active"));


  return flamegpu::ALIVE;
}