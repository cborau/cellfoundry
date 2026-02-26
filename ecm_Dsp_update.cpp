/**
 * vec3CrossProd
 *
 * Purpose:
 *   Compute the cross product of two 3D vectors and store result in (x, y, z).
 *
 * Inputs:
 *   - x1, y1, z1: first vector
 *   - x2, y2, z2: second vector
 *
 * Outputs:
 *   - x, y, z: cross product result (modified)
 */
FLAMEGPU_DEVICE_FUNCTION void vec3CrossProd(float &x, float &y, float &z, float x1, float y1, float z1, float x2, float y2, float z2) {
  x = (y1 * z2 - z1 * y2);
  y = (z1 * x2 - x1 * z2);
  z = (x1 * y2 - y1 * x2);
}

/**
 * vec3Div
 *
 * Purpose:
 *   Divide a 3D vector (x, y, z) by a scalar divisor in-place.
 *
 * Inputs:
 *   - x, y, z: vector components (modified)
 *   - divisor: scalar value
 *
 * Outputs:
 *   - x, y, z: scaled vector components
 */
FLAMEGPU_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
  x /= divisor;
  y /= divisor;
  z /= divisor;
}

/**
 * vec3Length
 *
 * Purpose:
 *   Compute the Euclidean length of a 3D vector (x, y, z).
 *
 * Inputs:
 *   - x, y, z: vector components
 *
 * Outputs:
 *   - Returns the magnitude of the vector
 */
FLAMEGPU_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}

/**
 * vec3Normalize
 *
 * Purpose:
 *   Normalize a 3D vector in-place using its length.
 *
 * Inputs:
 *   - x, y, z: vector components (modified)
 *
 * Outputs:
 *   - x, y, z: normalized vector components
 */
FLAMEGPU_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
  float length = vec3Length(x, y, z);
  vec3Div(x, y, z, length);
}

/**
 * ecm_Dsp_update
 *
 * Purpose:
 *   Compute local FNODE crowding around each ECM voxel and downscale diffusion
 *   coefficients to represent heterogeneous transport in dense regions.
 *
 * Inputs:
 *   - Spatial FNODE messages around each ECM position
 *   - Environment controls: equilibrium distance, average voxel density
 *
 * Outputs:
 *   - Updated D_sp array per ECM agent
 */
FLAMEGPU_AGENT_FUNCTION(ecm_Dsp_update, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");
  const float ECM_ECM_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("ECM_ECM_EQUILIBRIUM_DISTANCE");
  const unsigned int AVG_NETWORK_VOXEL_DENSITY = FLAMEGPU->environment.getProperty<unsigned int>("AVG_NETWORK_VOXEL_DENSITY");
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float D_sp[N_SPECIES] = {}; 
  for (int i = 0; i < N_SPECIES; i++) {
    D_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("D_sp", i);
  }
  // ECM agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
    
  // Fnode agent position 
  int message_id = 0;
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;

  // direction: the vector joining interacting agents
  float dir_x = 0.0; 
  float dir_y = 0.0; 
  float dir_z = 0.0; 
  float distance = 0.0;
  
  unsigned int n_fibre = 0;
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) { // find fnode agents within broadcast radius
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_z = message.getVariable<float>("z");
    dir_x = agent_x - message_x; 
    dir_y = agent_y - message_y; 
    dir_z = agent_z - message_z; 
    distance = vec3Length(dir_x, dir_y, dir_z); 
    if (distance < ECM_ECM_EQUILIBRIUM_DISTANCE) {
        n_fibre++;
      }   
  }

  // Normalise by average density (avoid div by zero)
  const float avg = (AVG_NETWORK_VOXEL_DENSITY > 0) ? (float)AVG_NETWORK_VOXEL_DENSITY : 1.0f;
  const float rho = (float)n_fibre / avg;
  
  // Tunables (use diff_reduction_test.py to find optimal values for a given problem)
  const float alpha = 1.0f;     // strength of reduction
  const float m_min = 0.05f;    // floor multiplier so diffusion never fully stops

  // Saturating mapping
  float m = 1.0f / (1.0f + alpha * fmaxf(rho - 1.0f, 0.0f)); // no penalty for densities below 1, then saturating reduction for higher densities. 
  m = fmaxf(m_min, fminf(1.0f, m));

  // Apply to each species
  for (int i = 0; i < N_SPECIES; i++) {
    D_sp[i] *= m;
    FLAMEGPU->setVariable<float, N_SPECIES>("D_sp", i, D_sp[i]);
  }

  return flamegpu::ALIVE;
}