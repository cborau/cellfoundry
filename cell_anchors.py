"""Optional CELL anchor storage and RTC specialization for focal adhesions.

The compile-time feature is derived from INCLUDE_FOCAL_ADHESIONS. It is not a
second user parameter. Shared nucleus/stress variables are independent of it.
"""

import json
from pathlib import Path

ANCHOR_ARRAYS = ("x_i", "y_i", "z_i", "u_ref_x_i", "u_ref_y_i", "u_ref_z_i")


def declare_cell_anchors(agent, include_focal_adhesions, count):
    """Omit the arrays entirely when focal adhesions are disabled."""
    if include_focal_adhesions:
        for name in ANCHOR_ARRAYS:
            agent.newVariableArrayFloat(name, count)


def register_cell_rtc(agent, name, filename, include_focal_adhesions):
    """Register an anchor-aware CELL kernel, including variant FILES overrides.

    Compile out every anchor access/local array in a model without focal
    adhesions. #line retains the original filename/line in RTC diagnostics.
    No generated source files or kernel constants are written to disk.
    """
    path = Path(filename).resolve()
    source = (
        f"#define CELLFOUNDRY_CELL_ANCHORS {int(bool(include_focal_adhesions))}\n"
        f"#line 1 {json.dumps(path.as_posix())}\n"
        + path.read_text(encoding="utf-8")
    )
    return agent.newRTCFunction(name, source)
