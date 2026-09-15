/**
 * multiscale_ecm_diffusion_step
 *
 * Purpose:
 *   Advance the selected species by one fixed explicit diffusion timestep,
 *   followed by first-order degradation and prescribed concentration constraints.
 *
 * Inputs:
 *   - MessageArray3D: previous substep's ECM positions, C_sp and D_sp
 *   - Agent position, grid indices, C_sp, D_sp and cached concentration constraints
 *   - DIFFUSION_ACTIVE_SPECIES, TIME_STEP_DIFFUSION and ECM_AGENTS_PER_DIR
 *   - HETEROGENEOUS_DIFFUSION, DIFFUSION_COEFF_MULTI, ECM_DEGRADATION_RATE_MULTI
 *   - DIFFUSION_CFL_SAFETY
 *
 * Outputs:
 *   - Updated agent C_sp for the active species only
 *   - diffusion_error: 0 = valid, 1 = invalid data, 2 = unsafe fixed timestep
 *
 * Notes:
 *   The same ordinary C++ function is registered in the parent and submodels.
 *   Each model's environment selects its species; no source-code placeholders
 *   or per-group C++ function names are needed. Missing grid neighbors have
 *   zero flux. Symmetric pair coefficients conserve equal-reference-volume
 *   mass in the absence of degradation and imposed concentrations.
 */
FLAMEGPU_AGENT_FUNCTION(multiscale_ecm_diffusion_step, flamegpu::MessageArray3D, flamegpu::MessageNone) {
  // Agent array variables
  const uint8_t N_SPECIES = 3; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  float C_sp[N_SPECIES] = {};
  float C_sp_next[N_SPECIES] = {};
  float D_sp[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
    C_sp_next[i] = C_sp[i];
    D_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("D_sp", i);
  }

  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  const int agent_grid[3] = {
    FLAMEGPU->getVariable<uint8_t>("grid_i"),
    FLAMEGPU->getVariable<uint8_t>("grid_j"),
    FLAMEGPU->getVariable<uint8_t>("grid_k")
  };
  unsigned int ECM_AGENTS_PER_DIR[3] = {};
  for (int axis = 0; axis < 3; axis++) {
    ECM_AGENTS_PER_DIR[axis] = FLAMEGPU->environment.getProperty<unsigned int>("ECM_AGENTS_PER_DIR", axis);
  }
  const unsigned int HETEROGENEOUS_DIFFUSION = FLAMEGPU->environment.getProperty<unsigned int>("HETEROGENEOUS_DIFFUSION");
  const float DIFFUSION_CFL_SAFETY = FLAMEGPU->environment.getProperty<float>("DIFFUSION_CFL_SAFETY");

  // Named error codes replace the previous bit-pattern construction of infinity.
  // The host checks this flag before another substep or any main-step output.
  const unsigned int DIFFUSION_INVALID_DATA = 1;
  const unsigned int DIFFUSION_UNSAFE_TIMESTEP = 2;
  FLAMEGPU->setVariable<unsigned int>("diffusion_error", 0);

  for (int i = 0; i < N_SPECIES; i++) {
    const unsigned int DIFFUSION_ACTIVE_SPECIES = FLAMEGPU->environment.getProperty<unsigned int>("DIFFUSION_ACTIVE_SPECIES", i);
    if (DIFFUSION_ACTIVE_SPECIES == 0) {
      continue;
    }
    const float TIME_STEP_DIFFUSION = FLAMEGPU->environment.getProperty<float>("TIME_STEP_DIFFUSION", i);
    float DIFFUSION_COEFF = FLAMEGPU->environment.getProperty<float>("DIFFUSION_COEFF_MULTI", i);
    if (HETEROGENEOUS_DIFFUSION == 1) {
      DIFFUSION_COEFF = D_sp[i];
    }
    if (!isfinite(DIFFUSION_COEFF) || DIFFUSION_COEFF < 0.0f || !isfinite(C_sp[i])) {
      FLAMEGPU->setVariable<unsigned int>("diffusion_error", DIFFUSION_INVALID_DATA);
      return flamegpu::ALIVE;
    }

    float diffusion_rate = 0.0f; // Sum of D_face / distance^2, used for stability.
    float concentration_rate = 0.0f; // Sum of neighbor transfers, in concentration/s.
    // Visit left/right, back/front and down/up. No diagonal or wrapped neighbors.
    for (int axis = 0; axis < 3; axis++) {
      for (int direction = -1; direction <= 1; direction += 2) {
        int neighbor_grid[3] = {agent_grid[0], agent_grid[1], agent_grid[2]};
        neighbor_grid[axis] += direction;
        if (neighbor_grid[axis] < 0 || neighbor_grid[axis] >= (int)ECM_AGENTS_PER_DIR[axis]) {
          // Omitting this edge is equivalent to the original zero-flux ghost
          // concentration being equal to this agent's own concentration.
          continue;
        }
        const auto message = FLAMEGPU->message_in.at(neighbor_grid[0], neighbor_grid[1], neighbor_grid[2]);
        const float dir_x = agent_x - message.getVariable<float>("x");
        const float dir_y = agent_y - message.getVariable<float>("y");
        const float dir_z = agent_z - message.getVariable<float>("z");
        const float distance_squared = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;
        const float message_C_sp = message.getVariable<float, N_SPECIES>("C_sp", i);
        float message_D_sp = DIFFUSION_COEFF;
        if (HETEROGENEOUS_DIFFUSION == 1) {
          message_D_sp = message.getVariable<float, N_SPECIES>("D_sp", i);
        }
        // Coincident agents cannot define a finite diffusion exchange rate.
        if (!isfinite(distance_squared) || distance_squared <= 0.0f ||
            !isfinite(message_D_sp) || message_D_sp < 0.0f || !isfinite(message_C_sp)) {
          FLAMEGPU->setVariable<unsigned int>("diffusion_error", DIFFUSION_INVALID_DATA);
          return flamegpu::ALIVE;
        }

        // Harmonic mean: 2*D_agent*D_neighbor/(D_agent+D_neighbor).
        // The equivalent ratio form avoids multiplying two large coefficients.
        // If either coefficient is zero, this pair is impermeable.
        float DIFFUSION_COEFF_FACE = 0.0f;
        if (DIFFUSION_COEFF > 0.0f && message_D_sp > 0.0f) {
          const float smaller_coefficient = fminf(DIFFUSION_COEFF, message_D_sp);
          const float larger_coefficient = fmaxf(DIFFUSION_COEFF, message_D_sp);
          DIFFUSION_COEFF_FACE = smaller_coefficient * (2.0f / (1.0f + smaller_coefficient / larger_coefficient));
        }
        const float neighbor_rate = DIFFUSION_COEFF_FACE / distance_squared;
        diffusion_rate += neighbor_rate;
        concentration_rate += neighbor_rate * (message_C_sp - C_sp[i]);
      }
    }

    // The timestep was fixed at startup using the minimum expected spacing.
    // Detect a broken bound instead of silently adapting the clock or clipping C.
    // The small tolerance only allows float32 roundoff at the configured margin.
    const float diffusion_cfl = TIME_STEP_DIFFUSION * diffusion_rate;
    if (!isfinite(diffusion_cfl) || diffusion_cfl > DIFFUSION_CFL_SAFETY * (1.0f + 1.0e-5f)) {
      FLAMEGPU->setVariable<unsigned int>("diffusion_error", DIFFUSION_UNSAFE_TIMESTEP);
      return flamegpu::ALIVE;
    }

    const float ECM_DEGRAD = FLAMEGPU->environment.getProperty<float>("ECM_DEGRADATION_RATE_MULTI", i);
    const float concentration_after_diffusion = C_sp[i] + TIME_STEP_DIFFUSION * concentration_rate;
    // Exact decay-only integration; the combined diffusion scheme is first order.
    C_sp_next[i] = concentration_after_diffusion * expf(-ECM_DEGRAD * TIME_STEP_DIFFUSION);
    if (!isfinite(C_sp_next[i])) {
      FLAMEGPU->setVariable<unsigned int>("diffusion_error", DIFFUSION_INVALID_DATA);
      return flamegpu::ALIVE;
    }
    const float vascular_floor = FLAMEGPU->getVariable<float, N_SPECIES>("diffusion_vascular_floor", i);
    if (vascular_floor >= 0.0f) {
      C_sp_next[i] = fmaxf(C_sp_next[i], vascular_floor);
    }
    const float boundary_concentration = FLAMEGPU->getVariable<float, N_SPECIES>("diffusion_boundary", i);
    if (boundary_concentration >= 0.0f) {
      C_sp_next[i] = boundary_concentration;
    }
  }

  // Do not partially update this agent if any active species failed validation.
  for (int i = 0; i < N_SPECIES; i++) {
    if (FLAMEGPU->environment.getProperty<unsigned int>("DIFFUSION_ACTIVE_SPECIES", i) == 1) {
      FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, C_sp_next[i]);
    }
  }
  return flamegpu::ALIVE;
}
