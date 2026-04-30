/**
 * vec3Length_dsp_lumen (local helper, avoids name collision)
 */
FLAMEGPU_DEVICE_FUNCTION float vec3Length_dsp_lumen(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}

/**
 * ecm_Dsp_lumen_update
 *
 * Purpose:
 *   Override the D_sp diffusion coefficient of ECM voxels that contain at least
 *   one LUMEN agent. LUMEN represents a liquid void with free diffusion, so if
 *   any LUMEN droplet is found within the voxel's equilibrium distance, D_sp is
 *   overwritten with LUMEN_DIFFUSION_COEFF_MULTI (the lumen-specific diffusion
 *   coefficient) for all species. This runs after ecm_Dsp_update (the
 *   fibre-crowding reduction step), replacing its value where lumen is present.
 *
 *   Only called when INCLUDE_LUMEN is True.
 *
 * Inputs:
 *   - lumen_spatial_location_message (broadcast radius >= ECM_ECM_EQUILIBRIUM_DISTANCE)
 *   - Environment: ECM_ECM_EQUILIBRIUM_DISTANCE, LUMEN_DIFFUSION_COEFF_MULTI
 *
 * Outputs:
 *   - Updated D_sp array per ECM agent (overwritten to lumen value when lumen present)
 */
FLAMEGPU_AGENT_FUNCTION(ecm_Dsp_lumen_update, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  const float ECM_ECM_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("ECM_ECM_EQUILIBRIUM_DISTANCE");
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.

  // Agent position
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");

  bool lumen_present = false;
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const float mx = message.getVariable<float>("x");
    const float my = message.getVariable<float>("y");
    const float mz = message.getVariable<float>("z");
    const float dist = vec3Length_dsp_lumen(agent_x - mx, agent_y - my, agent_z - mz);
    if (dist < ECM_ECM_EQUILIBRIUM_DISTANCE) {
      lumen_present = true;
      break;
    }
  }

  if (lumen_present) {
    for (int i = 0; i < N_SPECIES; i++) {
      const float D_lumen = FLAMEGPU->environment.getProperty<float, N_SPECIES>("LUMEN_DIFFUSION_COEFF_MULTI", i);
      FLAMEGPU->setVariable<float, N_SPECIES>("D_sp", i, D_lumen);
    }
  }

  return flamegpu::ALIVE;
}
