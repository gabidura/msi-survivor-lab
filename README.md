# MSI Survivor Lab

**MSI Survivor Lab** is a Python research library and demonstration GUI for finite survivor-envelope structures associated with prime constraints.

It implements computational tools for:

- local obstruction profiles and residue-cloud overlaps;
- finite packet-sieve survivor counts;
- global and windowed survivor envelopes;
- complete-sieve prime-interior tests;
- explicit small-prime boundary-channel completion;
- layered residual-prime attrition;
- localized survivor-polynomial coefficient laws;
- validation checks against classical local products and small examples;
- diagram-based exploration through the GUI scripts.

## Research status

This repository is an experimental research-software companion to the MSI survivor-envelope preprint sequence. It is intended for reproducible exploration of exact finite survivor identities, local obstruction profiles, finite packet-sieve survivor counts, global and windowed survivor envelopes, localized window counts, layered residual-prime attrition, boundary-channel completion, and survivor-polynomial coefficient laws.

The software does **not** claim to prove Goldbach, Hardy--Littlewood, Bateman--Horn, or prime-distribution conjectures. It is not presented as a proof assistant and does not certify any unproved prime-distribution conjecture. Its purpose is to make the finite definitions, identities, examples, diagrams, and numerical workflows inspectable and reproducible.

## Quick start

Clone the repository and run:

```bash
python -m pip install -e .
msi-survivor --self-test
```

Run a windowed survivor count:

```bash
msi-survivor --count --H 0 --z 8 --n 58 --start 0 --stop 59 --json
```

Run a layered exhaustion report:

```bash
msi-survivor --layered --H 0 --z 5 --y 3 --n 58 --start 0 --stop 59
```

Run a survivor-polynomial coefficient-law check:

```bash
msi-survivor --poly-law --H 0 --z 5 --n 58 --start 0 --stop 59 --json
```

Launch the full GUI directly:

```bash
python apps/msi_survivor_gui.py
```

The GUI uses `tkinter` and optionally `matplotlib` for diagrams. To install plotting support:

```bash
python -m pip install -e .[plots]
```

## Historical Python versions

The folder `archive/python_versions/` contains the original single-file Python development sequence:

- `v01`: baseline direct verification;
- `v02`: memoized envelope engine and dynamic-programming convolution;
- `v03`: diagram layer;
- `v04`: known-theory validation checks;
- `v05`: pattern search;
- `v06`: layered interval exhaustion;
- `v07`: boundary-channel and multi-layer residual accounting;
- `v08`: survivor-polynomial law and zoomable diagrams;
- `v09`: integrated Python/C++ backend bridge.

At least v01--v03 are included to document the early development path. Later versions are included for reproducibility of the current research workflow.

## Core mathematical objects

For a finite shift set `H`, diagonal parameter `n`, and sieve level `z`, the program computes local survivor values

\[
Y_p(u,H)=p-c_p(u,H),
\]

global survivor envelopes, windowed counts

\[
R_z(n,H;I),
\]

and the localized survivor-polynomial coefficient law

\[
R_Q(n,H;I)=[X^0]_Q\,W_I(X)S_{Q,n,H}(X^{-1}).
\]

The software is designed to keep exact finite survivor-distribution claims separate from any unproved prime-distribution conjectures.

## Related preprints

- Additive Encoding of Primality Constraints and Local Obstruction Structure in a Symmetric Affine System: <https://zenodo.org/records/20259817>
- Exact Finite Local Sieve Structures for Symmetric Diagonal Affine Systems, previous version: <https://zenodo.org/records/20447546>
- Exact Finite Packet Sieves and Survivor Convolutions for Symmetric Diagonal Affine Systems, revised merged version: DOI `10.5281/zenodo.20573926`

## License

MIT License. See `LICENSE`.
