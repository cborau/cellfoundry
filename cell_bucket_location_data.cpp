FLAMEGPU_AGENT_FUNCTION(cell_bucket_location_data, flamegpu::MessageNone, flamegpu::MessageBucket) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));

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