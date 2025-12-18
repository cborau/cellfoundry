FLAMEGPU_AGENT_FUNCTION(bcorner_output_location_data, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
  // bconer_output_location_data agent function for BCORNER agents, which outputs publicly visible properties to a message list
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  return flamegpu::ALIVE;
}