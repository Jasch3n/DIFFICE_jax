"""Registry of consistency-check functions.

Every check takes `(config, regions, **kwargs)` and returns a
`CheckResult` — mirroring how every provider in `data_sources` returns a
fixed contract regardless of file format. See CONTEXT.md's "Consistency
check" for what distinguishes these from code-correctness tests.
"""

from joint_xpinn_data.checks.hydrostatic import check_hydrostatic_equilibrium
from joint_xpinn_data.checks.velocity_front import check_velocity_vs_front

CHECKS = {
    "velocity_vs_front": check_velocity_vs_front,
    "hydrostatic_equilibrium": check_hydrostatic_equilibrium,
}
