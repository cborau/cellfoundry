/**
 * cell_rg_differentiation  [RG variant - new function]
 *
 * Purpose:
 *   Drive iPSC -> NPC -> RG differentiation via a logistic ODE gated by
 *   local morphogen concentration (ECM species 2) and autocrine/paracrine
 *   signalling from already-committed RG neighbours.  When rg_commitment
 *   crosses a threshold, the agent's cell_type is updated and its k_production
 *   for species 2 is synchronised with the new type.
 *
 * Layer: L3b_RG_Differentiation - after L3 (metabolism; fresh C_sp values)
 *        and before L4 (ECM Csp update; so updated k_production is used in
 *        the same step's diffusion solve).
 *
 * Message input: cell_spatial_location_message (same spatial message broadcast
 *                in L1; read-only - neighbours are from the previous step, but
 *                this one-step lag is acceptable for slow differentiation ODEs).
 *
 * New env properties (registered in variants/radial_glia/__init__.py):
 *   RG_COMMIT_RATE             [1/s]       - morphogen-independent basal rate
 *   RG_COMMIT_AUTOCRINE_RATE   [1/(s.a.u.)]- per-unit morphogen drive
 *   RG_COMMIT_INHIBIT_RATE     [1/s]       - Notch-Delta lateral inhibition rate
 *   RG_COMMIT_THRESHOLD_NPC    [-]         - rg_commitment threshold: iPSC->NPC
 *   RG_COMMIT_THRESHOLD_RG     [-]         - rg_commitment threshold: NPC->RG
 *   RG_EPITHELIAL_RATE         [1/s]       - epithelialization kinetics
 *   CELL_PRODUCTION_MULTIPLIER [N_CELL_TYPES] - inherited from base model
 *   INIT_CELL_PRODUCTION_RATES [N_SPECIES] - base k_production per species
 */
FLAMEGPU_AGENT_FUNCTION(cell_rg_differentiation, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
  if (FLAMEGPU->getVariable<int>("dead") == 1) {
    return flamegpu::ALIVE;
  }

  // -------------------------------------------------------------------------
  // Own state
  // -------------------------------------------------------------------------
  const int agent_id = FLAMEGPU->getVariable<int>("id");
  const float agent_x = FLAMEGPU->getVariable<float>("x");
  const float agent_y = FLAMEGPU->getVariable<float>("y");
  const float agent_z = FLAMEGPU->getVariable<float>("z");
  int agent_cell_type = FLAMEGPU->getVariable<int>("cell_type");
  float rg_commit = FLAMEGPU->getVariable<float>("rg_commit_level");
  float epi = FLAMEGPU->getVariable<float>("epithelialization_level");
  float rosette_maturity = 0.0f;

  const float apx = FLAMEGPU->getVariable<float>("apx");
  const float apy = FLAMEGPU->getVariable<float>("apy");
  const float apz = FLAMEGPU->getVariable<float>("apz");

  // -------------------------------------------------------------------------
  // Environment
  // -------------------------------------------------------------------------
  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");

  const uint8_t N_SPECIES = 3;   // WARNING: must match main python
  const uint8_t N_CELL_TYPES = 3;
  const uint32_t ECM_POPULATION_SIZE = 9261; // WARNING: must match Nx*Ny*Nz

  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");

  const int Nx = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 0);
  const int Ny = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 1);
  const int Nz = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR", 2);

  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES", 5);

  const float RG_COMMIT_RATE           = FLAMEGPU->environment.getProperty<float>("RG_COMMIT_RATE");
  const float RG_COMMIT_AUTOCRINE_RATE = FLAMEGPU->environment.getProperty<float>("RG_COMMIT_AUTOCRINE_RATE");
  const float RG_COMMIT_THRESHOLD_NPC  = FLAMEGPU->environment.getProperty<float>("RG_COMMIT_THRESHOLD_NPC");
  const float RG_COMMIT_THRESHOLD_RG   = FLAMEGPU->environment.getProperty<float>("RG_COMMIT_THRESHOLD_RG");
  const float RG_EPITHELIAL_RATE       = FLAMEGPU->environment.getProperty<float>("RG_EPITHELIAL_RATE");

  // -------------------------------------------------------------------------
  // Map agent position to ECM voxel
  // -------------------------------------------------------------------------
  int gi = (int)roundf(((agent_x - COORD_BOUNDARY_X_NEG) / (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG)) * (Nx - 1));
  int gj = (int)roundf(((agent_y - COORD_BOUNDARY_Y_NEG) / (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG)) * (Ny - 1));
  int gk = (int)roundf(((agent_z - COORD_BOUNDARY_Z_NEG) / (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG)) * (Nz - 1));
  gi = (gi < 0) ? 0 : (gi >= Nx ? Nx - 1 : gi);
  gj = (gj < 0) ? 0 : (gj >= Ny ? Ny - 1 : gj);
  gk = (gk < 0) ? 0 : (gk >= Nz ? Nz - 1 : gk);
  const uint32_t c_idx = (uint32_t)(gi * (Ny * Nz) + gj * Nz + gk);

  // Local morphogen (species 2) concentration - sampled from the ECM macro grid at this cell's
  // nearest voxel.  This is the EXTRACELLULAR concentration (uM), NOT the cell's intracellular
  // sp2 amount stored in the cell variable C_sp[2].  The intracellular quantity is managed by
  // cell_ecm_interaction_metabolism and represents morphogen that has been taken up or retained
  // inside the cell.  Only morphogen_local (the ECM gradient) drives differentiation; C_sp[2] is
  // a different compartment and is not used here.
  const float local_sp2 = fmaxf((float)C_SP_MACRO[2][c_idx], 0.0f);

  // -------------------------------------------------------------------------
  // Neighbour census via spatial message loop
  // -------------------------------------------------------------------------
  int total_neighbours = 0;
  int rg_neighbours = 0;
  float rg_align_sum   = 0.0f;
  float inhibitory_sum = 0.0f;  // Notch-Delta: sum of commit_nb * |apz_nb| for RG neighbours

  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
    const int msg_id = message.getVariable<int>("id");
    if (msg_id == agent_id) continue;
    if (message.getVariable<int>("dead") == 1) continue;

    total_neighbours++;
    const int nb_type = message.getVariable<int>("cell_type");
    if (nb_type == 2) {
      rg_neighbours++;
      // Rosette maturity: accumulate |nb_apz| (upward orientation of neighbour's apical pole).
      // Using |apz| of neighbours (and the cell itself, after the loop) is the chosen proxy
      // geometric signature of apicobasal polarisation in a columnar rosette.
      const float nb_apz    = message.getVariable<float>("apz");
      const float nb_commit = message.getVariable<float>("rg_commit_level");
      rg_align_sum   += fabsf(nb_apz);
      // Notch-Delta inhibitory signal: Delta ligand expression proportional to rg_commit * |apz|.
      // Using |apz| as junction-maturity proxy - only cells that have established
      // apicobasal polarity send strong inhibitory signal (~8h delay after RG commitment
      // before the inhibitory fence goes up, allowing the initial rosette to nucleate).
      inhibitory_sum += nb_commit * fabsf(nb_apz);
    }
  }

  const float local_rg_fraction = (total_neighbours > 0)
      ? ((float)rg_neighbours / (float)(total_neighbours + 1))
      : 0.0f;

  // Notch-Delta inhibitory signal normalised by neighbourhood size (same denominator
  // as local_rg_fraction so it scales consistently with cell density).
  const float local_delta_signal = inhibitory_sum / (float)(total_neighbours + 1);

  // Rosette maturity: own |apz| * mean(neighbour |apz|).
  if (rg_neighbours > 0) {
    rosette_maturity = fabsf(apz) * rg_align_sum / (float)rg_neighbours;
  }

  // -------------------------------------------------------------------------
  // Commitment ODE (logistic, forward Euler + Gaussian noise)
  //   d_commit/dt = (rate_basal + rate_autocrine * sp2 + rate_paracrine * rg_frac)
  //                 * (1 - rg_commit) + noise
  // Gaussian noise (RG_COMMIT_NOISE * sqrt(dt), Ito convention) spreads
  // differentiation timing across the population so cells do not all cross
  // NPC/RG thresholds simultaneously, enabling spatial heterogeneity and
  // cluster/rosette nucleation.
  // -------------------------------------------------------------------------
  const float RG_COMMIT_NOISE       = FLAMEGPU->environment.getProperty<float>("RG_COMMIT_NOISE");
  const float RG_COMMIT_DECAY_RATE  = FLAMEGPU->environment.getProperty<float>("RG_COMMIT_DECAY_RATE");
  const float RG_COMMIT_INHIBIT_RATE = FLAMEGPU->environment.getProperty<float>("RG_COMMIT_INHIBIT_RATE");
  // Basal rate applies only to iPSC (cell_type == 0).  Once a cell is NPC or
  // RG, further progression requires sp2 signalling.  Without this gate the
  // positive basal drive accumulates every step and ALL cells reach commit = 1
  // within ~67 h regardless of sp2, producing all-RG outcomes.
  // Positive paracrine term removed: replaced by Notch-Delta lateral inhibition below.
  const float drive = (agent_cell_type == 0 ? RG_COMMIT_RATE : 0.0f)
                    + RG_COMMIT_AUTOCRINE_RATE * local_sp2;
  const float noise = RG_COMMIT_NOISE * FLAMEGPU->random.normal<float>() * sqrtf(TIME_STEP);
  rg_commit += drive * (1.0f - rg_commit) * TIME_STEP + noise;
  // First-order decay: cells not receiving morphogen signal lose commitment.
  rg_commit -= RG_COMMIT_DECAY_RATE * rg_commit * TIME_STEP;
  // Notch-Delta lateral inhibition: committed RG cells send Delta signal proportional
  // to their (rg_commit * |apz|) (Delta expression * junction maturity).  This adds an
  // effective decay term that suppresses neighbours' commitment, producing spatial
  // pattern (isolated RG islands surrounded by NPC).  The |apz| factor delays the
  // inhibitory fence by ~8h after RG commitment, allowing the rosette ring to nucleate.
  // Equilibrium: x_eq = drive / (drive + k_decay + k_inhibit * delta_signal)
  // With 1 mature RG neighbor (delta~0.09): effective_decay doubles -> x_eq < 0.70 -> NPC.
  rg_commit -= RG_COMMIT_INHIBIT_RATE * local_delta_signal * rg_commit * TIME_STEP;
  rg_commit = fminf(fmaxf(rg_commit, 0.0f), 1.0f);

  // -------------------------------------------------------------------------
  // Cell-type switch and k_production sync
  // -------------------------------------------------------------------------
  int new_type = (rg_commit >= RG_COMMIT_THRESHOLD_RG)  ? 2
               : (rg_commit >= RG_COMMIT_THRESHOLD_NPC) ? 1
               : 0;
  // Ratchet: fate commitment is irreversible.  Noise can transiently push
  // rg_commit below a threshold; without this guard the cell_type would
  // oscillate (e.g. NPC -> iPSC -> NPC) which is biologically wrong.
  if (new_type < agent_cell_type) new_type = agent_cell_type;

  if (new_type != agent_cell_type) {
    agent_cell_type = new_type;
    // Update k_production[2] = INIT_CELL_PRODUCTION_RATES[2]
    //                           * CELL_PRODUCTION_MULTIPLIER[new_type]
    const float base_prod = FLAMEGPU->environment.getProperty<float, N_SPECIES>(
        "INIT_CELL_PRODUCTION_RATES", 2);
    const float multiplier = FLAMEGPU->environment.getProperty<float, N_CELL_TYPES>(
        "CELL_PRODUCTION_MULTIPLIER", new_type);
    FLAMEGPU->setVariable<float, N_SPECIES>("k_production", 2, base_prod * multiplier);
    FLAMEGPU->setVariable<int>("cell_type", agent_cell_type);
  }

  // -------------------------------------------------------------------------
  // Epithelialization ODE (logistic, forward Euler)
  //   d_epi/dt = RG_EPITHELIAL_RATE * rg_commit * (1 - epi)
  // Applies to both NPC and RG cells: NPC cells in the pseudo-stratified
  // neuroepithelium do acquire partial apico-basal polarity proportional to
  // their commitment level.  The z-alignment itself (in cell_rg_polarity_update)
  // is additionally scaled by rg_commit for NPC cells, so peripheral NPC cells
  // with low commitment develop much weaker z-polarity than RG cells.
  // -------------------------------------------------------------------------
  if (agent_cell_type >= 1) {
    epi += RG_EPITHELIAL_RATE * rg_commit * (1.0f - epi) * TIME_STEP;
    epi = fminf(fmaxf(epi, 0.0f), 1.0f);
  }

  // -------------------------------------------------------------------------
  // Write back
  // -------------------------------------------------------------------------
  FLAMEGPU->setVariable<float>("rg_commit_level",         rg_commit);
  FLAMEGPU->setVariable<float>("epithelialization_level", epi);
  FLAMEGPU->setVariable<float>("rosette_maturity",        rosette_maturity);
  FLAMEGPU->setVariable<float>("rg_neighbour_density",    local_rg_fraction);
  FLAMEGPU->setVariable<float>("morphogen_local",          local_sp2);

  return flamegpu::ALIVE;
}
