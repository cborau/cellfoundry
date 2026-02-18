// FOCAD agent updates its anchor point on the nucleus surface (xi) using bucket messages from CELL.
// - The calling agent is FOCAD
// - Receives MessageBucket from CELL agents
// - Reads the bucket with index = agent_cell_id
// - Updates FOCAD nucleus center (x_c,y_c,z_c) from the CELL
// - Loops through all CELL anchors (a fixed-size list) and selects the closest anchor to the FOCAD position (x,y,z)
// - Updates FOCAD (x_i,y_i,z_i) with that closest anchor position
//
// Assumptions / required message fields from CELL bucket message:
// - "id" (CELL id, to sanity check)
// - "x_c","y_c","z_c" (cell nucleus center)
// - Anchor arrays, size = N_ANCHOR_POINTS (compile-time constant):
//     "x_i","y_i","z_i" as variable arrays in the message
//
FLAMEGPU_AGENT_FUNCTION(focad_anchor_update, flamegpu::MessageBucket, flamegpu::MessageNone) {
  
  const uint8_t N_ANCHOR_POINTS = 100; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  // -------------------------
  // Get FOCAD agent variables (agent calling the function)
  // -------------------------
  const int agent_cell_id = FLAMEGPU->getVariable<int>("cell_id");
  const float MAX_FOCAD_ARM_LENGTH = FLAMEGPU->environment.getProperty<float>("MAX_FOCAD_ARM_LENGTH");

  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  // Current stored cell center / anchor positions
  float agent_x_c = FLAMEGPU->getVariable<float>("x_c");
  float agent_y_c = FLAMEGPU->getVariable<float>("y_c");
  float agent_z_c = FLAMEGPU->getVariable<float>("z_c");

  float agent_x_i = FLAMEGPU->getVariable<float>("x_i");
  float agent_y_i = FLAMEGPU->getVariable<float>("y_i");
  float agent_z_i = FLAMEGPU->getVariable<float>("z_i");

  int agent_anchor_id = FLAMEGPU->getVariable<int>("anchor_id"); 

  // -------------------------
  // Read CELL bucket: index = cell_id
  // -------------------------
  // Expect exactly 1 message in that bucket (the corresponding CELL).
  uint8_t found_cell = 0;

  // We will pick the closest anchor to the current FOCAD position (x,y,z)
  float best_r2 = 2 * (MAX_FOCAD_ARM_LENGTH * MAX_FOCAD_ARM_LENGTH); // 2 times just to give it some margin
  float best_xi = agent_x_i;
  float best_yi = agent_y_i;
  float best_zi = agent_z_i;
  int best_anchor_id = agent_anchor_id;

  for (const auto &message : FLAMEGPU->message_in(agent_cell_id)) {
    // Optional sanity check: you can ensure message id matches agent_cell_id
    // const int message_id = message.getVariable<int>("id");

    // Update stored nucleus center
    agent_x_c = message.getVariable<float>("x");
    agent_y_c = message.getVariable<float>("y");
    agent_z_c = message.getVariable<float>("z");

    if (agent_anchor_id >= 0) {
      // Anchor already assigned, skip the search
      best_xi = message.getVariable<float, N_ANCHOR_POINTS>("x_i", agent_anchor_id);
      best_yi = message.getVariable<float, N_ANCHOR_POINTS>("y_i", agent_anchor_id);
      best_zi = message.getVariable<float, N_ANCHOR_POINTS>("z_i", agent_anchor_id);
      found_cell = 1;
    } else {
      // Loop anchors in message arrays and find closest to FOCAD (x,y,z)
      for (unsigned int ai = 0; ai < N_ANCHOR_POINTS; ++ai) {
        const float ax = message.getVariable<float, N_ANCHOR_POINTS>("x_i", ai);
        const float ay = message.getVariable<float, N_ANCHOR_POINTS>("y_i", ai);
        const float az = message.getVariable<float, N_ANCHOR_POINTS>("z_i", ai);

        const float dx = ax - agent_x;
        const float dy = ay - agent_y;
        const float dz = az - agent_z;
        const float r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < best_r2) {
          best_r2 = r2;
          best_xi = ax;
          best_yi = ay;
          best_zi = az;
          best_anchor_id = ai;
        }
      }
      found_cell = 1;
      // There should only be one CELL message in the bucket;
      break;
    }
  }

  // If we found the cell message, update the chosen anchor
  if (found_cell != 0) {
    agent_x_i = best_xi;
    agent_y_i = best_yi;
    agent_z_i = best_zi;
    agent_anchor_id = best_anchor_id;
  }
  // else: no message found in bucket (should not happen). Keep previous x_c/xi values.

  // -------------------------
  // Write back updated variables
  // -------------------------
  FLAMEGPU->setVariable<float>("x_c", agent_x_c);
  FLAMEGPU->setVariable<float>("y_c", agent_y_c);
  FLAMEGPU->setVariable<float>("z_c", agent_z_c);

  FLAMEGPU->setVariable<float>("x_i", agent_x_i);
  FLAMEGPU->setVariable<float>("y_i", agent_y_i);
  FLAMEGPU->setVariable<float>("z_i", agent_z_i);

  //FLAMEGPU->setVariable<int>("anchor_id", agent_anchor_id);
  FLAMEGPU->setVariable<int>("anchor_id", -1); // Reset anchor_id to -1 to allow dynamic re-assignment at each step, or keep it fixed after first assignment.

  return flamegpu::ALIVE;
}
