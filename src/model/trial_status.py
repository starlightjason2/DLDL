"""Trial lifecycle status.

Kept in its own leaf module (importing nothing from ``model`` or ``util``) so both
``model.hp_trial`` and ``util.hptune`` can import it at the top level without a
circular dependency.
"""

from __future__ import annotations

from enum import IntEnum


class TrialStatus(IntEnum):
    """Persisted as integers in the trial CSV."""

    COMPLETED = 0
    RUNNING = -1
    QUEUED = -2
    FAILED = -3
