/**
 * fnode_bucket_location_data_postmove
 *
 * Purpose:
 *   Broadcast FNODE post-move positions into a dedicated bucket message list.
 *   This runs after fnode_move in L8 so that focad_move can read the current-step
 *   FNODE position instead of the stale L1 pre-move data.
 *
 * Notes:
 *   This is a lightweight version of fnode_bucket_location_data that only
 *   carries the variables needed by focad_move (position and velocity).
 */
FLAMEGPU_AGENT_FUNCTION(fnode_bucket_location_data_postmove, flamegpu::MessageNone, flamegpu::MessageBucket) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
  FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
  FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));
  FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<int>("id"));

  return flamegpu::ALIVE;
}
