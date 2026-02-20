# Model Editor

<p align="left">
  <img src="https://raw.githubusercontent.com/cborau/cellfoundry/master/assets/icon.png" alt="Model Editor icon" width="72">
</p>

## Purpose

`param_ui.py` provides a custom desktop interface to inspect and edit simulation parameters in `model.py` without manually searching through the source file. It is designed to speed up model iteration and reduce editing errors during parameter tuning.

## What it does

- Loads `model.py` and indexes key configuration variables.
- Exposes grouped controls for frequently tuned parameters (time stepping, boundaries, feature toggles, etc.).
- Preserves comments and formatting while patching variable assignments.
- Provides a code editor view with syntax highlighting for quick navigation.
- Supports launching model runs from the same interface to shorten edit-run cycles.

## Why use it

- Faster navigation across large configuration sections.
- Reduced risk of syntax mistakes in manual edits.
- Better reproducibility when adjusting many parameters across experiments.

## Interface Preview

![Model Parameter Editor screenshot](https://raw.githubusercontent.com/cborau/cellfoundry/master/assets/parameter_editor.png)
