/**
 * cell_bucket_location_data
 *
 * Purpose:
 *   Export CELL state required by bucket-based readers (e.g., focal adhesion updates).
 *
 * Inputs:
 *   - CELL variables: id, position, orientation, anchor arrays
 *
 * Outputs:
 *   - MessageBucket keyed by CELL id containing anchor geometry and pose
 */
FLAMEGPU_AGENT_FUNCTION(cell_bucket_location_data, flamegpu::MessageNone, flamegpu::MessageBucket) {

  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("orx", FLAMEGPU->getVariable<float>("orx"));
  FLAMEGPU->message_out.setVariable<float>("ory", FLAMEGPU->getVariable<float>("ory"));
  FLAMEGPU->message_out.setVariable<float>("orz", FLAMEGPU->getVariable<float>("orz"));
  FLAMEGPU->message_out.setVariable<int>("dead", FLAMEGPU->getVariable<int>("dead"));

  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  for (int i = 0; i < N_ANCHOR_POINTS; i++) {
    float val1 = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("x_i", i);
    FLAMEGPU->message_out.setVariable<float, N_ANCHOR_POINTS>("x_i", i, val1);
    float val2 = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("y_i", i);
    FLAMEGPU->message_out.setVariable<float, N_ANCHOR_POINTS>("y_i", i, val2);
    float val3 = FLAMEGPU->getVariable<float, N_ANCHOR_POINTS>("z_i", i);
    FLAMEGPU->message_out.setVariable<float, N_ANCHOR_POINTS>("z_i", i, val3);
  }


  FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<int>("id"));

  return flamegpu::ALIVE;
}