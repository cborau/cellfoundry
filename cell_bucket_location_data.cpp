FLAMEGPU_AGENT_FUNCTION(cell_bucket_location_data, flamegpu::MessageNone, flamegpu::MessageBucket) {
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));


  FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<int>("id"));

  return flamegpu::ALIVE;
}