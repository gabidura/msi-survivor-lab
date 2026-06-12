# MSI Survivor Lab

**MSI Survivor Lab** is an experimental Python research library and demonstration GUI for finite survivor-envelope structures associated with prime constraints.

It implements computational tools for:

* local obstruction profiles and residue-cloud overlaps;
* finite packet-sieve survivor counts;
* global and windowed survivor envelopes;
* complete-sieve prime-interior tests;
* explicit small-prime boundary-channel completion;
* layered residual-prime attrition;
* localized survivor-polynomial coefficient laws;
* validation checks against classical local products, finite identities, and small examples;
* diagram-based exploration through GUI scripts.

## Research status

MSI Survivor Lab is an experimental research-software companion to the MSI survivor-envelope preprint sequence. It is intended for reproducible exploration of exact finite survivor identities, local obstruction profiles, finite packet-sieve survivor counts, global and windowed survivor envelopes, localized window counts, layered residual-prime attrition, boundary-channel completion, and survivor-polynomial coefficient laws.

The software does **not** claim to prove Goldbach, Hardy--Littlewood, Bateman--Horn, or prime-distribution conjectures. It is not a proof assistant and does not certify any unproved prime-distribution conjecture. Its purpose is to make the finite definitions, identities, examples, diagrams, and numerical workflows of the survivor-envelope program inspectable and reproducible.

## Quick start

Clone the repository and install it in editable mode:

```bash
python -m pip install -e .
```

Run the self-test:

```bash
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

Launch the GUI directly:

```bash
python apps/msi_survivor_gui.py
```

The GUI uses `tkinter` and optionally `matplotlib` for diagrams. To install plotting support:

```bash
python -m pip install -e .[plots]
```

## Mathematical objects

For a finite shift set `H`, diagonal parameter `n`, and sieve level `z`, the program computes local survivor values

[
Y_p(u,H)=p-c_p(u,H),
]

where (c_p(u,H)) is the local obstruction count associated with the residue-cloud overlap structure.

It also computes windowed survivor counts

[
R_z(n,H;I),
]

global and local survivor envelopes, and the localized survivor-polynomial coefficient law

[
R_Q(n,H;I)
==========

[X^0]*Q,W_I(X)S*{Q,n,H}(X^{-1}).
]

Here (W_I(X)) is the window polynomial and (S_{Q,n,H}(X)) is the survivor-residue polynomial modulo (Q).

The software is designed to keep exact finite survivor-distribution claims separate from any unproved prime-distribution conjectures.

## Main modules and workflows

MSI Survivor Lab supports the following computational workflows:

1. **Local obstruction analysis**
   Computes residue clouds, overlap profiles, local forbidden classes, and local survivor values.

2. **Global envelope computation**
   Computes global survivor-fiber distributions from local laws.

3. **Windowed survivor counts**
   Computes localized survivor counts inside finite intervals.

4. **Complete-sieve prime interior tests**
   Tests the finite implication that sufficiently deep survivors correspond to large-prime interior packets.

5. **Boundary-channel completion**
   Enumerates explicit small-prime boundary channels of the form
   [
   L_j(i)=q,\qquad q\le z.
   ]

6. **Layered residual-prime attrition**
   Splits a full sieve level into a base exhaustion level and residual prime layers.

7. **Survivor-polynomial coefficient laws**
   Verifies localized survivor counts through cyclic polynomial coefficient extraction.

8. **Validation checks**
   Compares finite local products and small examples with classical calibration targets.

9. **Diagram exploration**
   Supports visual exploration of local laws, global envelopes, windowed profiles, and related finite structures.

## Historical Python versions

The folder `archive/python_versions/` preserves the original single-file Python development sequence:

* `v01`: baseline direct verification;
* `v02`: memoized envelope engine and dynamic-programming convolution;
* `v03`: diagram layer;
* `v04`: known-theory validation checks;
* `v05`: pattern search;
* `v06`: layered interval exhaustion;
* `v07`: boundary-channel and multi-layer residual accounting;
* `v08`: survivor-polynomial law and zoomable diagrams;
* `v09`: integrated Python/C++ backend bridge.

Versions `v01`--`v03` document the early development path. Later versions are included where useful for reproducing the current workflow and research history.

## Related preprints

The software accompanies the following preprints and project records:

1. **Additive Encoding of Primality Constraints and Local Obstruction Structure in a Symmetric Affine System**
   Zenodo record: https://zenodo.org/records/20259817
   DOI: 10.5281/zenodo.20259817

2. Exact Finite Local Sieve Structures for Symmetric Diagonal Affine Systems
   Earlier/superseded preprint version.
   Zenodo record: https://zenodo.org/records/20447546
   DOI: 10.5281/zenodo.20447546

3. **Exact Finite Packet Sieves and Survivor Convolutions for Symmetric Diagonal Affine Systems**
   Revised merged version.
   Zenodo record: https://zenodo.org/records/20573926
   DOI: 10.5281/zenodo.20573926

4. **Global, Windowed, and Layered Survivor Envelopes: Primorial Survivor Spaces, Complete-Sieve Prime Interior, and Residual-Prime Attrition
   Global survivor-envelope and distribution-theoretic framework.**
   Zenodo record: https://zenodo.org/records/20617961
   DOI: 10.5281/zenodo.20617961
   
5. **Goldbach Survivor-Gap Problem: Complete-Sieve Interior, Two-Cloud Survivor Masks, and Computational Validation
   Goldbach-specialized survivor-gap and complete-sieve validation paper.**
   Zenodo record: https://zenodo.org/records/20660792
   DOI: 10.5281/zenodo.20660792

Additional software records, computational reports, and minorant-certificate experiments are being organized for Zenodo release.

## Citation

If you use this software, please cite the repository and the associated Zenodo software release when available. The mathematical background is documented in the companion preprints listed above.

A software DOI for MSI Survivor Lab is planned through Zenodo after the GitHub release is stabilized.

## License

MIT License. See `LICENSE`.

## Disclaimer

This repository is a research prototype. It is intended for experimental computation, reproducibility, and mathematical exploration. Numerical experiments produced by the software should not be interpreted as proofs of unproved prime-distribution conjectures unless accompanied by independent mathematical arguments.
