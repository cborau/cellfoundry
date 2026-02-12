// computes CELL agent movement
FLAMEGPU_AGENT_FUNCTION(cell_move, flamegpu::MessageNone, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");

  //Update agent position based on velocity
  // TODO
  // CHEMOTAXIS: add contribution to velocity based on chemical gradient // CONTACT FORCES: add contribution to velocity based on forces from other agents (e.g. focal adhesion forces, cell-cell adhesion forces, etc.) // RANDOM MOTILITY: add contribution to velocity based on random motility (e.g. Brownian motion)

  //Set agent variables
  FLAMEGPU->setVariable<int>("id", agent_id);
  FLAMEGPU->setVariable<float>("x", agent_x);
  FLAMEGPU->setVariable<float>("y", agent_y);
  FLAMEGPU->setVariable<float>("z", agent_z);
  FLAMEGPU->setVariable<float>("vx", agent_vx);
  FLAMEGPU->setVariable<float>("vy", agent_vy);
  FLAMEGPU->setVariable<float>("vz", agent_vz);

  return flamegpu::ALIVE;
}