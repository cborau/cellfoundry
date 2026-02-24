FLAMEGPU_AGENT_FUNCTION(cell_MaxID_update, flamegpu::MessageNone, flamegpu::MessageNone) {
  const int macro_max_global_cell_id = FLAMEGPU->environment.getMacroProperty<int, 1>("MACRO_MAX_GLOBAL_CELL_ID");
  FLAMEGPU->setVariable<int>("max_global_cell_id", macro_max_global_cell_id);
  return flamegpu::ALIVE;
}
