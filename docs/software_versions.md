# MSI Survivor Lab software version history

This repository preserves the Python development sequence because the versions document the research workflow.

- **v01 baseline direct verifier**: first brute-force checks of local profiles, CRT-period counts, and small examples.
- **v02 optimized theoretical engine**: memoization, dynamic-programming multiplicative convolution, global envelopes, complete-window identities.
- **v03 diagram layer**: visual exploration through local-law plots, envelope histograms, CDFs, windowed profiles, threshold plots, and torus projections.
- **v04 validation layer**: Mertens/zeta calibration, affine root-count tests, polynomial local-factor checks, Goldbach local factors, observed-versus-predicted prime packet tests.
- **v05 pattern search**: complete-survivor scans across diagonals, residue-class summaries, and CSV export.
- **v06 layered exhaustion**: partial-period exhaustion and residual-prime attrition certificates.
- **v07 boundary/multi-layer**: small-prime boundary-channel completion and staged residual-layer accounting.
- **v08 polynomial law**: sparse survivor-polynomial supports, coefficient-law verification, survivor gap statistics, zoom/pan diagram support.
- **v09 integrated backend**: Python GUI/CLI front-end connected to a C++ JSON/CSV back-end for large scans.

The public Python package starts at semantic version **0.1.0**.
