"""MSI Survivor Lab: Python tools for primorial survivor-envelope experiments.

This package exposes the computational core used by the MSI Survivor Lab GUI:
local obstruction profiles, global/windowed survivor envelopes, boundary-channel
completion, layered residual attrition, validation checks, and survivor-polynomial
coefficient laws.

Research status: this software supports finite computational experiments and
preprint verification. It does not claim to prove Goldbach, Hardy--Littlewood,
Bateman--Horn, or any prime-distribution conjecture.
"""

from ._version import __version__
from .core import (
    primes_up_to,
    primorial,
    is_prime,
    parse_H,
    residue_cloud,
    overlap_profile,
    local_profile,
    local_law,
    global_envelope,
    R_value,
    packet_values,
    survives_packet,
    windowed_survivor_count,
    boundary_completion_report,
    layered_exhaustion_summary,
    layered_report,
    polynomial_window_count,
    polynomial_law_report,
    validation_report,
    scan_complete_survivors,
    pattern_summary,
    self_test,
)

