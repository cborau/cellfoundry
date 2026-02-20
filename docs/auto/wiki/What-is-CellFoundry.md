# What is CellFoundry

<p align="center">
  <img src="https://raw.githubusercontent.com/cborau/cellfoundry/master/assets/logo_cellfoundry.png" alt="CellFoundry logo" width="360">
</p>

## Overview

CellFoundry is a multi-physics, agent-based simulation framework for studying the cellular microenvironment. It combines interacting cells, extracellular matrix, fibre networks, diffusing chemical species, mechanical coupling, and more, in a single GPU-accelerated model.

The framework is designed for in vitro and organoid-scale studies where transport, mechanics, and microstructure jointly affect cell behaviour. Its modular structure also makes it suitable for parameter sweeps, digital twin prototyping, and mechanobiology hypothesis testing.

Model structure and initialization are contained in a single Python file (model.py), while agent interaction implementation is separated into single C++ files. Agent functions are fully customizable and can be used to simulate a wide range of biological processes## Core Model Components

- **Cells (CELL)**: migration, metabolism, stress updates, and interactions with ECM and adhesions.
- **Extracellular matrix (ECM)**: concentration fields, diffusion, and voxel-level mechanics.
- **Fibre nodes (FNODE)**: network mechanics and boundary interactions.
- **Focal adhesions (FOCAD)**: attachment dynamics and force transmission between cells and fibres.
- **Boundary/corner agents (BCORNER)**: domain constraints and boundary condition enforcement.

## Outputs and Analysis

CellFoundry produces a range of output data that can be analyzed to extract biological insights:

- **VTK files**: 3D visualization of cell and ECM states over time.
- **Pickle snapshots**: Complete model state at specified intervals for detailed analysis.
- **Custom output functions**: User-defined functions that extract specific metrics or generate reports.

## Built on FLAME GPU 2

CellFoundry is implemented on top of FLAME GPU 2, which provides high-performance GPU execution for agent-based models.

- FLAME GPU 2 repository: <https://github.com/FLAMEGPU/FLAMEGPU2>
- FLAME GPU 2 documentation: <https://docs.flamegpu.com/>
- FLAME GPU 2 examples: <https://github.com/FLAMEGPU/FLAMEGPU2/tree/master/examples>

## Typical Workflow

1. Configure model parameters in `model.py` (or through the Model Editor UI).
2. Run simulation to produce VTK outputs and optional pickle snapshots.
3. Analyze dynamics using scripts in `postprocessing/`.
4. Use the generated function reference to inspect model behavior and implementation details.
