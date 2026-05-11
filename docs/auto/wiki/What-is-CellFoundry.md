<p align="center">
  <img src="https://raw.githubusercontent.com/cborau/cellfoundry/master/assets/logo_cellfoundry.png" alt="Cellfoundry logo" width="360">
</p>

## Overview

Cellfoundry is a multi-physics, agent-based simulation framework for studying the cellular microenvironment. It combines interacting cells, extracellular matrix, fibre networks, diffusing chemical species, mechanical coupling, and more, in a single GPU-accelerated model.

The framework is designed for in vitro and organoid-scale studies where transport, mechanics, and microstructure jointly affect cell behaviour. Its modular structure also makes it suitable for parameter sweeps, digital twin prototyping, and mechanobiology hypothesis testing.

Model structure and initialization are contained in a single Python file (model.py), while agent interaction implementation is separated into single C++ files. Agent functions are fully customizable and can be used to simulate a wide range of biological processes.

## Core Model Components

- **Cells (CELL)**: migration, metabolism, stress updates, and interactions with ECM and adhesions.
- **Extracellular matrix (ECM)**: concentration fields, diffusion, and voxel-level mechanics.
- **Fibre nodes (FNODE)**: network mechanics and boundary interactions.
- **Focal adhesions (FOCAD)**: attachment dynamics and force transmission between cells and fibres.
- **Boundary/corner agents (BCORNER)**: domain constraints and boundary condition enforcement.

## Outputs and Analysis

Cellfoundry produces a range of output data that can be analyzed to extract biological insights:

- **VTK files**: 3D visualization of cell and ECM states over time.
- **Pickle snapshots**: Complete model state at specified intervals for detailed analysis.
- **Custom output functions**: User-defined functions that extract specific metrics or generate reports.

## Built on FLAME GPU 2

Cellfoundry is implemented on top of FLAME GPU 2, which provides high-performance GPU execution for agent-based models.

- FLAME GPU 2 repository: <https://github.com/FLAMEGPU/FLAMEGPU2>
- FLAME GPU 2 documentation: <https://docs.flamegpu.com/>
- FLAME GPU 2 examples: <https://github.com/FLAMEGPU/FLAMEGPU2/tree/master/examples>

## Typical Workflow

1. Configure model parameters in `model.py` (or through the Model Editor UI).
2. Optionally, create a `.json` file to override specific parameters at runtime:
```json
{
  "STEPS": 2400,
  "TIME_STEP": 180,
  "SAVE_EVERY_N_STEPS": 12,
  "INCLUDE_CELLS": true,
  "INCLUDE_CELL_CELL_INTERACTION": true,
  "INCLUDE_CELL_CYCLE": true,
  "ORGANOID_ASSAY": true,
  "ORGANOID_INIT_RADIUS": 20.0,
  "CELL_RADIUS": [10.0, 10.0, 10.0],
  "SAVE_DATA_TO_FILE": true,
  "SAVE_PICKLE": true,
  "N_CELLS": 10,
  "CELL_SPEED_REF": 0.005,
  "ROTATIONAL_DIFFUSION_RATE": 0.0005,
  "DIVISION_RATE_MULTIPLIER": [1.0, 1.5, 2.0],
  "CELL_CELL_ADHESION_K": 9.8,
  "CELL_CELL_REPULSION_K": 60.7
}
```
3. Run simulation to produce VTK outputs and optional pickle snapshots.
4. Analyze dynamics using scripts in `postprocessing/`.
5. Use the generated function reference to inspect model behavior and implementation details.

## Supplementary Videos

![Metabolism showcase](https://raw.githubusercontent.com/cborau/cellfoundry/master/assets/SuppVideo3_Metabolism.gif)

*This video illustrates the integration of chemical and mechanical interactions in the model, where cell motility and matrix remodelling are influenced by local biochemical cues, while cells also actively modify their microenvironment through secretion and consumption of diffusible factors. In this example, 100 cells are shown. The cells consume species 0 (e.g. nutrients present in the environment) and release species 1, generating heterogeneous concentration patterns in the surrounding domain. The blobs are coloured and scaled according to the local concentration values of species 1*

![Matrix degradation](https://raw.githubusercontent.com/cborau/cellfoundry/master/assets/SuppVideo4_Matrix_Degradation.gif)

*Matrix degradation by a migrating cell. Surrounding fibre nodes (FNODEs) within the interaction range are represented as blobs whose size and colour reflect their local degradation state. As degradation progresses and reaches a value of 1, the corresponding FNODE is removed from the simulation, thereby modelling local matrix breakdown, relaxing mechanical constraints and enabling cell invasion through the matrix*

![Matrix reinforcement](https://raw.githubusercontent.com/cborau/cellfoundry/master/assets/SuppVideo5_Matrix_Reinforcement.gif)

*Matrix reinforcement driven by a migrating cell. Surrounding fibre nodes (FNODEs) within the interaction range are shown as blobs whose size and colour indicate the local reinforcement level. Reinforcement promotes the generation of new FNODEs and connected fibres, enabling progressive matrix deposition and remodelling around the cell. This can lead to local stiffening and densification of the matrix, which in turn can influence cell motility, force transmission and diffusion of species.*

![Organoid growth](https://raw.githubusercontent.com/cborau/cellfoundry/master/assets/SuppVideo6_Organoid_Growth.gif)

*Organoid growth driven by the proliferation of three cell types that progressively assemble into a compact three-dimensional structure. Cells are coloured by cell type in the left panel and by cell-cycle phase in the right panel, highlighting both the emerging tissue composition and the heterogeneous proliferative dynamics during organoid formation.*
