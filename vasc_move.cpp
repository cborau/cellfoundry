/**
 * vasc_move
 *
 * Purpose:
 *   Advect each alive VASC node with the local ECM grid velocity so the
 *   vascular network deforms coherently when the simulation boundaries move.
 *
 * Inputs:
 *   - Agent variables: dead, x, y, z
 *   - MessageArray3D: ecm_grid_location_message (vx, vy, vz per grid voxel)
 *   - Environment properties: TIME_STEP, ECM_AGENTS_PER_DIR[3], COORDS_BOUNDARIES[6]
 *
 * Outputs:
 *   - Updated agent x, y, z (new position after advection)
 *   - Updated agent vx, vy, vz (local ECM velocity, cached for output)
 *
 * Notes:
 *   Only registered in the model when MOVING_BOUNDARIES is True.
 */
FLAMEGPU_AGENT_FUNCTION(vasc_move, flamegpu::MessageArray3D, flamegpu::MessageNone) {

    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    float agent_z = FLAMEGPU->getVariable<float>("z");

    const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");

    // ECM grid dimensions
    const int Nx = (int)FLAMEGPU->environment.getProperty<unsigned int>("ECM_AGENTS_PER_DIR", 0);
    const int Ny = (int)FLAMEGPU->environment.getProperty<unsigned int>("ECM_AGENTS_PER_DIR", 1);
    const int Nz = (int)FLAMEGPU->environment.getProperty<unsigned int>("ECM_AGENTS_PER_DIR", 2);

    // Domain boundaries
    const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 0);
    const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 1);
    const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 2);
    const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 3);
    const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 4);
    const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 5);

    // Map continuous position to nearest ECM grid index (same formula as cell_ecm_interaction_metabolism.cpp)
    int grid_i = (int)roundf(((agent_x - COORD_BOUNDARY_X_NEG) / (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG)) * (float)(Nx - 1));
    int grid_j = (int)roundf(((agent_y - COORD_BOUNDARY_Y_NEG) / (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG)) * (float)(Ny - 1));
    int grid_k = (int)roundf(((agent_z - COORD_BOUNDARY_Z_NEG) / (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG)) * (float)(Nz - 1));

    // Clamp to valid grid range
    grid_i = max(0, min(Nx - 1, grid_i));
    grid_j = max(0, min(Ny - 1, grid_j));
    grid_k = max(0, min(Nz - 1, grid_k));

    // Read ECM velocity from the nearest grid voxel
    const auto& msg = FLAMEGPU->message_in.at(grid_i, grid_j, grid_k);
    float ecm_vx = msg.getVariable<float>("vx");
    float ecm_vy = msg.getVariable<float>("vy");
    float ecm_vz = msg.getVariable<float>("vz");

    // Advect the VASC node with the local ECM velocity
    FLAMEGPU->setVariable<float>("x", agent_x + ecm_vx * TIME_STEP);
    FLAMEGPU->setVariable<float>("y", agent_y + ecm_vy * TIME_STEP);
    FLAMEGPU->setVariable<float>("z", agent_z + ecm_vz * TIME_STEP);
    // Cache velocity for post-processing / output
    FLAMEGPU->setVariable<float>("vx", ecm_vx);
    FLAMEGPU->setVariable<float>("vy", ecm_vy);
    FLAMEGPU->setVariable<float>("vz", ecm_vz);

    return flamegpu::ALIVE;
}
