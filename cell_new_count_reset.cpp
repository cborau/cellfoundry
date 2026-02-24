FLAMEGPU_AGENT_FUNCTION(cell_new_count_reset, flamegpu::MessageNone, flamegpu::MessageNone) {
  const int macro_max_global_cell_id = FLAMEGPU->environment.getMacroProperty<int, 1>("MACRO_MAX_GLOBAL_CELL_ID");
  FLAMEGPU->setVariable<int>("max_global_cell_id", macro_max_global_cell_id);
  FLAMEGPU->environment.getMacroProperty<int, 1>("MACRO_N_NEW_CELLS").exchange(0);
  return flamegpu::ALIVE;
}
