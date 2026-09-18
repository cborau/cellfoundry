"""Explicit rejection of an infeasible simulation, without hiding programming errors."""

import json
import os
from pathlib import Path


class TrialRejected(RuntimeError):
    """This parameter combination cannot produce a valid simulation/objective."""


def reject_trial(reason):
    """Abort a simulation and, under the optimizer, prune only this trial.

    Host-function exceptions can be wrapped by FLAMEGPU. Persist an explicit
    signal before raising so classification does not depend on exception text.
    The optimizer supplies a fresh token for each subprocess invocation.
    Outside optimization this simply raises a clear RuntimeError subclass.
    Use only for known infeasible states; ordinary errors should propagate.
    """
    path = os.environ.get("CELLFOUNDRY_TRIAL_REJECTION_FILE")
    token = os.environ.get("CELLFOUNDRY_TRIAL_TOKEN")
    if path and token:
        Path(path).write_text(json.dumps({"token": token, "reason": str(reason)}), encoding="utf-8")
    raise TrialRejected(str(reason))
