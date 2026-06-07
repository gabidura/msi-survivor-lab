#!/usr/bin/env python3
"""
MSI Survivor Lab v03 Diagrams
====================
Tabbed GUI + headless verification tool + diagram layer for the Modular Survivor Interface
(MSI), primorial survivor spaces, convolutional survivor envelopes, and
windowed/localized survivor envelopes.

Highlights in v03
-----------------
- Memoized local profiles, local laws, primorials and global exact envelopes.
- Dynamic-programming multiplicative convolution for the global envelope.
- Windowed envelope tab: complete-window identity + arbitrary-window boundary.
- Local/global/positive/log-envelope summaries.
- Prime-interior certification and small-prime boundary warning.
- Headless self-test for formula verification.
- Diagram tab for local laws, global envelopes, CDFs, windows, thresholds, and torus projections.

Run GUI:
    python msi_survivor_lab_v02.py

Run tests:
    python msi_survivor_lab_v02.py --self-test

The core computation is standard-library only. The diagram tab uses matplotlib when available.
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:  # headless mode still works
    tk = None
    ttk = None

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:  # diagrams are optional; headless tests still work
    Figure = None
    FigureCanvasTkAgg = None

# ---------------------------------------------------------------------------
# Core cached arithmetic
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def primes_tuple(z: int) -> Tuple[int, ...]:
    if z < 2:
        return ()
    sieve = [True] * (z + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(z ** 0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start:z + 1:p] = [False] * (((z - start) // p) + 1)
    return tuple(i for i, is_p in enumerate(sieve) if is_p)


def primes_up_to(z: int) -> List[int]:
    return list(primes_tuple(int(z)))


@lru_cache(maxsize=None)
def primorial_cached(z: int) -> int:
    q = 1
    for p in primes_tuple(int(z)):
        q *= p
    return q


def primorial(z: int) -> int:
    return primorial_cached(int(z))


def is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def factorization(n: int) -> Dict[int, int]:
    n = abs(int(n))
    out: Dict[int, int] = {}
    if n < 2:
        return out
    while n % 2 == 0:
        out[2] = out.get(2, 0) + 1
        n //= 2
    d = 3
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def omega(n: int) -> int:
    return sum(factorization(n).values())


def parse_H(text: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in text.replace(';', ',').split(',') if p.strip()]
    if not parts:
        raise ValueError("H must contain at least one integer, e.g. 0,2,6")
    return tuple(int(p) for p in parts)


@lru_cache(maxsize=None)
def residue_cloud_tuple(H: Tuple[int, ...], p: int) -> Tuple[int, ...]:
    return tuple(sorted({(-h - 1) % p for h in H}))


def residue_cloud(H: Sequence[int], p: int) -> List[int]:
    return list(residue_cloud_tuple(tuple(H), int(p)))


@lru_cache(maxsize=None)
def overlap_profile_tuple(H: Tuple[int, ...], p: int) -> Tuple[int, ...]:
    A = set(residue_cloud_tuple(H, p))
    vals = []
    for u in range(p):
        shifted = {((a + u) % p) for a in A}
        vals.append(len(A.intersection(shifted)))
    return tuple(vals)


def overlap_profile(H: Sequence[int], p: int) -> Dict[int, int]:
    return {u: v for u, v in enumerate(overlap_profile_tuple(tuple(H), int(p)))}


@lru_cache(maxsize=None)
def local_profile_tuple(H: Tuple[int, ...], p: int) -> Tuple[Tuple[int, int, int, int, int, int, int], ...]:
    A = residue_cloud_tuple(H, p)
    a = len(A)
    rho = overlap_profile_tuple(H, p)
    rows = []
    for u, rho_u in enumerate(rho):
        c = 2 * a - rho_u
        y = p - c
        rows.append((p, u, a, rho_u, c, y, 1 if y > 0 else 0))
    return tuple(rows)


def local_profile(H: Sequence[int], p: int) -> List[Dict[str, int]]:
    return [
        {"p": p0, "u": u, "a": a, "rho": rho, "c": c, "Y": y, "admissible": adm}
        for (p0, u, a, rho, c, y, adm) in local_profile_tuple(tuple(H), int(p))
    ]


@lru_cache(maxsize=None)
def local_law_tuple(H: Tuple[int, ...], p: int) -> Tuple[Tuple[int, int], ...]:
    law = Counter(row[5] for row in local_profile_tuple(H, p))
    return tuple(sorted(law.items()))


def local_law(H: Sequence[int], p: int) -> Counter:
    return Counter(dict(local_law_tuple(tuple(H), int(p))))


@lru_cache(maxsize=None)
def local_positive_law_tuple(H: Tuple[int, ...], p: int) -> Tuple[Tuple[int, int], ...]:
    law = Counter(row[5] for row in local_profile_tuple(H, p) if row[5] > 0)
    return tuple(sorted(law.items()))


def local_positive_law(H: Sequence[int], p: int) -> Counter:
    return Counter(dict(local_positive_law_tuple(tuple(H), int(p))))


def multiplicative_convolve(a: Mapping[int, int], b: Mapping[int, int]) -> Counter:
    out: Counter = Counter()
    for x, cx in a.items():
        for y, cy in b.items():
            out[x * y] += cx * cy
    return out


@lru_cache(maxsize=None)
def global_envelope_tuple(H: Tuple[int, ...], z: int) -> Tuple[Tuple[int, int], ...]:
    """Exact global multiplicative convolution envelope as sorted (R,count)."""
    dist: Counter = Counter({1: 1})
    for p in primes_tuple(int(z)):
        dist = multiplicative_convolve(dist, dict(local_law_tuple(H, p)))
    return tuple(sorted(dist.items()))


def global_envelope(H: Sequence[int], z: int) -> Counter:
    return Counter(dict(global_envelope_tuple(tuple(H), int(z))))


@lru_cache(maxsize=None)
def positive_global_envelope_tuple(H: Tuple[int, ...], z: int) -> Tuple[Tuple[int, int], ...]:
    dist: Counter = Counter({1: 1})
    for p in primes_tuple(int(z)):
        law = dict(local_positive_law_tuple(H, p))
        if not law:
            return tuple()
        dist = multiplicative_convolve(dist, law)
    return tuple(sorted(dist.items()))


def positive_global_envelope(H: Sequence[int], z: int) -> Counter:
    return Counter(dict(positive_global_envelope_tuple(tuple(H), int(z))))


def local_y_by_u(H: Sequence[int], p: int) -> Dict[int, int]:
    return {row[1]: row[5] for row in local_profile_tuple(tuple(H), int(p))}


def R_value(H: Sequence[int], z: int, n: int) -> int:
    prod = 1
    Ht = tuple(H)
    for p in primes_tuple(int(z)):
        y = {row[1]: row[5] for row in local_profile_tuple(Ht, p)}[(n + 2) % p]
        prod *= y
    return prod


def moment_from_distribution(dist: Mapping[int, int], s: int) -> int:
    if s < 1:
        raise ValueError("moments are defined here for s >= 1 to avoid 0^0 ambiguity")
    return sum((value ** s) * count for value, count in dist.items())


def distribution_cdf(dist: Mapping[int, int]) -> List[Tuple[int, int]]:
    total = 0
    out = []
    for value, count in sorted(dist.items()):
        total += count
        out.append((value, total))
    return out


# ---------------------------------------------------------------------------
# Windowed/localized envelope tools
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowResult:
    n: int
    start: int
    stop: int
    length: int
    q: int
    complete_blocks: int
    remainder: int
    period_fiber: int
    complete_contribution: int
    remainder_exact: int | None
    lower_bound: int
    upper_bound: int
    boundary_note: str


def packet_values(H: Sequence[int], n: int, i: int) -> List[int]:
    vals: List[int] = []
    for h in H:
        vals.append(i + h + 1)
        vals.append(n - i - h + 1)
    return vals


def survives_packet(H: Sequence[int], n: int, i: int, z: int) -> bool:
    vals = packet_values(H, n, i)
    for p in primes_tuple(int(z)):
        if any(v % p == 0 for v in vals):
            return False
    return True


def windowed_survivor_count(
    H: Sequence[int], z: int, n: int, start: int, stop: int, *, max_remainder_direct: int = 200_000
) -> WindowResult:
    """Count or bound survivors in i-window [start, stop] for fixed n.

    Complete blocks of length Q(z) contribute exactly R_z(n,H). The remainder
    is enumerated only if small enough; otherwise a rigorous interval bound is
    returned.
    """
    if stop < start:
        raise ValueError("stop must be >= start")
    q = primorial(z)
    length = stop - start + 1
    complete = length // q
    rem = length % q
    period_fiber = R_value(H, z, n)
    complete_contribution = complete * period_fiber
    rem_exact = None
    boundary_note = "complete window; no boundary" if rem == 0 else "remainder enumerated exactly"
    lower = complete_contribution
    upper = complete_contribution + rem
    if rem > 0 and rem <= max_remainder_direct:
        rem_start = start + complete * q
        rem_exact = sum(1 for i in range(rem_start, rem_start + rem) if survives_packet(H, n, i, z))
        lower = complete_contribution + rem_exact
        upper = lower
    elif rem > 0:
        boundary_note = f"remainder length {rem} too large for direct enumeration; exact count lies in [lower, upper]"
    return WindowResult(
        n=n,
        start=start,
        stop=stop,
        length=length,
        q=q,
        complete_blocks=complete,
        remainder=rem,
        period_fiber=period_fiber,
        complete_contribution=complete_contribution,
        remainder_exact=rem_exact,
        lower_bound=lower,
        upper_bound=upper,
        boundary_note=boundary_note,
    )


def interval_moment_summary(H: Sequence[int], z: int, N: int, s: int, *, start_n: int = 1, max_remainder_direct: int = 200_000) -> Dict[str, int | str | None]:
    if s < 1:
        raise ValueError("s must be >= 1")
    q = primorial(z)
    dist = global_envelope(H, z)
    period_sum = moment_from_distribution(dist, s)
    complete = N // q
    rem = N % q
    complete_contribution = complete * period_sum
    rem_exact = None
    lower = complete_contribution
    upper = complete_contribution + rem * (q ** s)
    note = "complete n-interval; no boundary" if rem == 0 else "n-remainder enumerated exactly"
    if rem > 0 and rem <= max_remainder_direct:
        rem_start = start_n + complete * q
        rem_exact = sum(R_value(H, z, n) ** s for n in range(rem_start, rem_start + rem))
        lower = complete_contribution + rem_exact
        upper = lower
    elif rem > 0:
        note = f"n-remainder length {rem} too large; exact moment lies in [lower, upper]"
    return {
        "Q": q,
        "N": N,
        "s": s,
        "complete_blocks": complete,
        "remainder": rem,
        "period_sum": period_sum,
        "complete_contribution": complete_contribution,
        "remainder_exact": rem_exact,
        "lower_bound": lower,
        "upper_bound": upper,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Prime interior and additive hyperplanes
# ---------------------------------------------------------------------------


def ceil_sqrt(n: int) -> int:
    r = math.isqrt(n)
    return r if r * r == n else r + 1


def omega_bound(B: int, z: int) -> float:
    return float("inf") if z <= 1 else math.log(max(B, 2)) / math.log(z)


def prime_interior_status(vals: Sequence[int], z: int) -> str:
    if any(v <= 1 for v in vals):
        return "outside positive range"
    B = max(vals)
    if z >= ceil_sqrt(B):
        if all(is_prime(v) and v > z for v in vals):
            return "large-prime interior packet"
        if all(is_prime(v) for v in vals):
            return "prime packet with small-prime boundary coordinate"
        return "contradiction/check assumptions"
    return "rough survivor / incomplete level"


def additive_hyperplane_local_count(p: int, d: int, N: int) -> int:
    if d < 1:
        raise ValueError("d must be >= 1")
    if N % p == 0:
        return ((p - 1) ** d + (p - 1) * ((-1) ** d)) // p
    return ((p - 1) ** d - ((-1) ** d)) // p


# ---------------------------------------------------------------------------
# Safe expression evaluator for later general polynomial experiments
# ---------------------------------------------------------------------------

ALLOWED_AST = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Name,
    ast.Load, ast.Div
)


def eval_expr(expr: str, variables: Dict[str, int]) -> int:
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST):
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in variables:
            raise ValueError(f"Unknown variable: {node.id}")
    return int(eval(compile(tree, '<expr>', 'eval'), {"__builtins__": {}}, variables))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test(verbose: bool = True) -> bool:
    tests: List[Tuple[bool, str]] = []
    examples = [
        ((0,), 5, {0: 15, 3: 8, 4: 2, 6: 4, 8: 1}),
        ((0, 2), 5, {0: 25, 1: 2, 2: 2, 3: 1}),
        ((0, 2, 6), 7, {0: 189, 2: 12, 4: 8, 8: 1}),
    ]
    for H, z, expected in examples:
        dist = global_envelope(H, z)
        tests.append((dict(dist) == expected, f"global envelope H={H}, z={z}: {dict(dist)}"))
        q = primorial(z)
        m1_direct = sum(R_value(H, z, n) for n in range(q))
        m1_dist = moment_from_distribution(dist, 1)
        tests.append((m1_direct == m1_dist, f"moment 1 H={H}, z={z}: direct={m1_direct}, dist={m1_dist}"))
        m2_direct = sum(R_value(H, z, n) ** 2 for n in range(q))
        m2_dist = moment_from_distribution(dist, 2)
        tests.append((m2_direct == m2_dist, f"moment 2 H={H}, z={z}: direct={m2_direct}, dist={m2_dist}"))

    # Dynamic interval exactness for complete n-periods.
    summary = interval_moment_summary((0, 2), 5, N=3 * primorial(5), s=2)
    tests.append((summary["lower_bound"] == 3 * moment_from_distribution(global_envelope((0, 2), 5), 2), f"dynamic interval complete: {summary}"))

    # Window complete identity: length M*Q gives M*R.
    H = (0,)
    z = 5
    n = 58
    q = primorial(z)
    wr = windowed_survivor_count(H, z, n, 0, 2 * q - 1)
    tests.append((wr.lower_bound == 2 * R_value(H, z, n) and wr.upper_bound == wr.lower_bound, f"complete i-window identity: {wr}"))

    # Complete-sieve endpoint: z=8 removes 7^2 composite in sum 60 example.
    H = (0,)
    n = 58
    z = 8
    bad = []
    good = []
    for i in range(0, 59):
        if survives_packet(H, n, i, z):
            vals = packet_values(H, n, i)
            if all(v > 1 for v in vals):
                good.append((i, vals))
                if not all(is_prime(v) and v > z for v in vals):
                    bad.append((i, vals))
    tests.append((not bad and bool(good), f"large-prime interior endpoint bad={bad}, good={good[:3]}"))

    ok = all(t[0] for t in tests)
    if verbose:
        for passed, msg in tests:
            print(("PASS" if passed else "FAIL") + ": " + msg)
        print("OVERALL:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# Tk GUI
# ---------------------------------------------------------------------------

if tk is not None:
    class SurvivorLabApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("MSI Survivor Lab v03 Diagrams - Local, Global, and Windowed Envelopes")
            self.geometry("1240x820")
            self.H_var = tk.StringVar(value="0,2")
            self.z_var = tk.IntVar(value=5)
            self.status_var = tk.StringVar(value="Ready")
            self._build_ui()

        def _build_ui(self) -> None:
            top = ttk.Frame(self, padding=8)
            top.pack(fill=tk.X)
            ttk.Label(top, text="H =").pack(side=tk.LEFT)
            ttk.Entry(top, textvariable=self.H_var, width=26).pack(side=tk.LEFT, padx=4)
            ttk.Label(top, text="z =").pack(side=tk.LEFT)
            ttk.Entry(top, textvariable=self.z_var, width=8).pack(side=tk.LEFT, padx=4)
            ttk.Button(top, text="Refresh all", command=self.refresh_all).pack(side=tk.LEFT, padx=6)
            ttk.Button(top, text="Clear caches", command=self.clear_caches).pack(side=tk.LEFT, padx=6)
            ttk.Button(top, text="Run self-test", command=self.run_self_test_gui).pack(side=tk.LEFT, padx=6)
            ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

            self.nb = ttk.Notebook(self)
            self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

            self.tab_overview = ttk.Frame(self.nb)
            self.tab_local = ttk.Frame(self.nb)
            self.tab_global = ttk.Frame(self.nb)
            self.tab_window = ttk.Frame(self.nb)
            self.tab_dynamic = ttk.Frame(self.nb)
            self.tab_prime = ttk.Frame(self.nb)
            self.tab_additive = ttk.Frame(self.nb)
            self.tab_optimization = ttk.Frame(self.nb)
            self.tab_diagrams = ttk.Frame(self.nb)
            self.tab_roadmap = ttk.Frame(self.nb)
            for frame, title in [
                (self.tab_overview, "Overview"),
                (self.tab_local, "Local envelope"),
                (self.tab_global, "Global envelope"),
                (self.tab_window, "Windowed envelope"),
                (self.tab_dynamic, "Dynamic n-intervals"),
                (self.tab_prime, "Prime interior"),
                (self.tab_additive, "Additive/Goldbach"),
                (self.tab_optimization, "Optimization"),
                (self.tab_diagrams, "Diagrams"),
                (self.tab_roadmap, "Roadmap"),
            ]:
                self.nb.add(frame, text=title)

            self._build_overview()
            self._build_local()
            self._build_global()
            self._build_window()
            self._build_dynamic()
            self._build_prime()
            self._build_additive()
            self._build_optimization()
            self._build_diagrams()
            self._build_roadmap()
            self.refresh_all()

        def get_params(self) -> Tuple[Tuple[int, ...], int]:
            H = parse_H(self.H_var.get())
            z = int(self.z_var.get())
            if z < 2:
                raise ValueError("z must be at least 2")
            return H, z

        def _text(self, parent):
            txt = tk.Text(parent, wrap=tk.WORD, font=("Consolas", 10))
            scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=txt.yview)
            txt.configure(yscrollcommand=scroll.set)
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            return txt

        def clear_tree(self, tree) -> None:
            for item in tree.get_children():
                tree.delete(item)

        def _build_overview(self) -> None:
            self.overview_text = self._text(self.tab_overview)

        def _build_local(self) -> None:
            cols = ("p", "u", "a", "rho", "c", "Y", "admissible")
            self.local_tree = ttk.Treeview(self.tab_local, columns=cols, show="headings")
            for c in cols:
                self.local_tree.heading(c, text=c)
                self.local_tree.column(c, width=90, anchor=tk.CENTER)
            self.local_tree.pack(fill=tk.BOTH, expand=True)

        def _build_global(self) -> None:
            frame = ttk.Frame(self.tab_global, padding=6)
            frame.pack(fill=tk.X)
            ttk.Button(frame, text="Export global CSV", command=self.export_global_distribution).pack(side=tk.LEFT)
            cols = ("R", "count", "probability", "cdf")
            self.global_tree = ttk.Treeview(self.tab_global, columns=cols, show="headings")
            for c in cols:
                self.global_tree.heading(c, text=c)
                self.global_tree.column(c, width=170, anchor=tk.CENTER)
            self.global_tree.pack(fill=tk.BOTH, expand=True)

        def _build_window(self) -> None:
            controls = ttk.Frame(self.tab_window, padding=6)
            controls.pack(fill=tk.X)
            self.win_n_var = tk.IntVar(value=58)
            self.win_a_var = tk.IntVar(value=0)
            self.win_b_var = tk.IntVar(value=59)
            ttk.Label(controls, text="n=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.win_n_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="i start=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.win_a_var, width=9).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="i stop=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.win_b_var, width=9).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Compute window", command=self.refresh_window).pack(side=tk.LEFT, padx=5)
            self.window_text = self._text(self.tab_window)

        def _build_dynamic(self) -> None:
            controls = ttk.Frame(self.tab_dynamic, padding=6)
            controls.pack(fill=tk.X)
            self.N_var = tk.IntVar(value=90)
            self.s_var = tk.IntVar(value=2)
            ttk.Label(controls, text="N length=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.N_var, width=10).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="moment s>=1=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.s_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Compute n-interval moment", command=self.refresh_dynamic).pack(side=tk.LEFT, padx=5)
            self.dynamic_text = self._text(self.tab_dynamic)

        def _build_prime(self) -> None:
            controls = ttk.Frame(self.tab_prime, padding=6)
            controls.pack(fill=tk.X)
            self.prime_n_var = tk.IntVar(value=58)
            self.i_min_var = tk.IntVar(value=0)
            self.i_max_var = tk.IntVar(value=58)
            ttk.Label(controls, text="n=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.prime_n_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="i min=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.i_min_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="i max=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.i_max_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="List survivor packets", command=self.refresh_prime).pack(side=tk.LEFT, padx=5)
            cols = ("i", "packet", "B", "ceil sqrt B", "omega max", "status")
            self.prime_tree = ttk.Treeview(self.tab_prime, columns=cols, show="headings")
            for c in cols:
                self.prime_tree.heading(c, text=c)
                self.prime_tree.column(c, width=170, anchor=tk.CENTER)
            self.prime_tree.pack(fill=tk.BOTH, expand=True)

        def _build_additive(self) -> None:
            controls = ttk.Frame(self.tab_additive, padding=6)
            controls.pack(fill=tk.X)
            self.add_N_var = tk.IntVar(value=60)
            self.add_d_var = tk.IntVar(value=2)
            ttk.Label(controls, text="N=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.add_N_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="d=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.add_d_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Compute additive local counts", command=self.refresh_additive).pack(side=tk.LEFT, padx=5)
            self.additive_text = self._text(self.tab_additive)

        def _build_optimization(self) -> None:
            self.optimization_text = self._text(self.tab_optimization)

        def _build_diagrams(self) -> None:
            outer = ttk.Frame(self.tab_diagrams, padding=6)
            outer.pack(fill=tk.X)
            self.diagram_type_var = tk.StringVar(value="Global envelope histogram")
            self.diagram_p_var = tk.IntVar(value=5)
            self.diagram_q_var = tk.IntVar(value=7)
            self.diagram_n0_var = tk.IntVar(value=0)
            self.diagram_count_var = tk.IntVar(value=60)
            self.diagram_win_n_var = tk.IntVar(value=58)
            self.diagram_win_a_var = tk.IntVar(value=0)
            self.diagram_win_b_var = tk.IntVar(value=59)
            self.diagram_B_var = tk.IntVar(value=1000)
            ttk.Label(outer, text="Plot:").pack(side=tk.LEFT)
            combo = ttk.Combobox(outer, textvariable=self.diagram_type_var, width=34, state="readonly",
                                 values=("Local survivor law", "Global envelope histogram", "Envelope CDF", "Windowed survivor profile", "Prime-selection threshold", "Two-prime torus projection"))
            combo.pack(side=tk.LEFT, padx=4)
            combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_diagram())
            ttk.Label(outer, text="p=").pack(side=tk.LEFT)
            ttk.Entry(outer, textvariable=self.diagram_p_var, width=5).pack(side=tk.LEFT)
            ttk.Label(outer, text="q=").pack(side=tk.LEFT)
            ttk.Entry(outer, textvariable=self.diagram_q_var, width=5).pack(side=tk.LEFT)
            ttk.Label(outer, text="n0=").pack(side=tk.LEFT)
            ttk.Entry(outer, textvariable=self.diagram_n0_var, width=7).pack(side=tk.LEFT)
            ttk.Label(outer, text="count=").pack(side=tk.LEFT)
            ttk.Entry(outer, textvariable=self.diagram_count_var, width=7).pack(side=tk.LEFT)
            ttk.Label(outer, text="I=[").pack(side=tk.LEFT)
            ttk.Entry(outer, textvariable=self.diagram_win_a_var, width=6).pack(side=tk.LEFT)
            ttk.Label(outer, text=",").pack(side=tk.LEFT)
            ttk.Entry(outer, textvariable=self.diagram_win_b_var, width=6).pack(side=tk.LEFT)
            ttk.Label(outer, text="] B=").pack(side=tk.LEFT)
            ttk.Entry(outer, textvariable=self.diagram_B_var, width=8).pack(side=tk.LEFT)
            ttk.Button(outer, text="Draw", command=self.refresh_diagram).pack(side=tk.LEFT, padx=4)
            ttk.Button(outer, text="Save PNG", command=self.save_diagram).pack(side=tk.LEFT, padx=4)

            if Figure is None or FigureCanvasTkAgg is None:
                self.diagram_text = self._text(self.tab_diagrams)
                self.diagram_text.insert(tk.END, "matplotlib is not available. The computational engine still works.\n")
                self.diagram_canvas = None
                self.diagram_fig = None
                self.diagram_ax = None
                return
            self.diagram_fig = Figure(figsize=(8.6, 5.4), dpi=100)
            self.diagram_ax = self.diagram_fig.add_subplot(111)
            self.diagram_canvas = FigureCanvasTkAgg(self.diagram_fig, master=self.tab_diagrams)
            self.diagram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def refresh_diagram(self) -> None:
            if not hasattr(self, 'diagram_canvas') or self.diagram_canvas is None:
                return
            H, z = self.get_params()
            ax = self.diagram_ax
            ax.clear()
            kind = self.diagram_type_var.get()
            if kind == "Local survivor law":
                p = int(self.diagram_p_var.get())
                rows = local_profile(H, p)
                xs = [r["u"] for r in rows]
                ys = [r["Y"] for r in rows]
                ax.bar(xs, ys)
                ax.set_title(f"Local survivor law Y_p(u), H={list(H)}, p={p}")
                ax.set_xlabel("u mod p")
                ax.set_ylabel("Y_p(u)=p-c_p(u,H)")
            elif kind == "Global envelope histogram":
                dist = global_envelope(H, z)
                xs = list(sorted(dist.keys()))
                ys = [dist[x] for x in xs]
                ax.bar(range(len(xs)), ys)
                ax.set_xticks(range(len(xs)))
                labels = [str(x) for x in xs]
                if len(labels) > 30:
                    labels = [labels[i] if i % max(1, len(labels)//20) == 0 else "" for i in range(len(labels))]
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_title(f"Global convolutional envelope, H={list(H)}, z={z}")
                ax.set_xlabel("survivor fiber value R")
                ax.set_ylabel("count over n mod Q(z)")
            elif kind == "Envelope CDF":
                dist = global_envelope(H, z)
                qtotal = primorial(z)
                cdf = distribution_cdf(dist)
                xs = [x for x, _ in cdf]
                ys = [c / qtotal for _, c in cdf]
                ax.step(xs, ys, where='post')
                ax.set_ylim(-0.02, 1.02)
                ax.set_title(f"Envelope CDF F_z(t), H={list(H)}, z={z}")
                ax.set_xlabel("t")
                ax.set_ylabel("P(R_z <= t)")
            elif kind == "Windowed survivor profile":
                n0 = int(self.diagram_n0_var.get())
                count = max(1, min(int(self.diagram_count_var.get()), 500))
                a = int(self.diagram_win_a_var.get())
                b = int(self.diagram_win_b_var.get())
                xs = list(range(n0, n0 + count))
                ys = [windowed_survivor_count(H, z, n, a, b).lower_bound for n in xs]
                ax.plot(xs, ys, marker='o', markersize=3, linewidth=1)
                ax.set_title(f"Windowed survivor profile, I=[{a},{b}], H={list(H)}, z={z}")
                ax.set_xlabel("n")
                ax.set_ylabel("R_z(n,H;I)")
            elif kind == "Prime-selection threshold":
                B = max(3, int(self.diagram_B_var.get()))
                zmax = max(10, 2 * ceil_sqrt(B))
                xs = list(range(2, zmax + 1))
                ys = [omega_bound(B, zz) for zz in xs]
                ax.plot(xs, ys)
                root = math.sqrt(B)
                ax.axvline(root, linestyle='--')
                ax.set_title(f"Large-prime-factor bound log(B)/log(z), B={B}")
                ax.set_xlabel("sieve level z")
                ax.set_ylabel("upper bound for Omega(L)")
            elif kind == "Two-prime torus projection":
                p = int(self.diagram_p_var.get())
                qv = int(self.diagram_q_var.get())
                length = p * qv // math.gcd(p, qv)
                ns = list(range(length))
                xs = [(n + 2) % p for n in ns]
                ys = [(n + 2) % qv for n in ns]
                colors = [R_value(H, z, n) for n in ns]
                sc = ax.scatter(xs, ys, c=colors, s=45)
                ax.set_title(f"Torus projection (mod {p}, mod {qv}), colored by R_z")
                ax.set_xlabel(f"n+2 mod {p}")
                ax.set_ylabel(f"n+2 mod {qv}")
                ax.set_xticks(range(p))
                ax.set_yticks(range(qv))
                self.diagram_fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.grid(True, alpha=0.25)
            self.diagram_fig.tight_layout()
            self.diagram_canvas.draw_idle()

        def save_diagram(self) -> None:
            if not hasattr(self, 'diagram_fig') or self.diagram_fig is None:
                messagebox.showwarning("No diagram", "matplotlib is not available")
                return
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if not path:
                return
            self.diagram_fig.savefig(path, dpi=160, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Saved {path}")

        def _build_roadmap(self) -> None:
            self.roadmap_text = self._text(self.tab_roadmap)
            self.roadmap_text.insert(tk.END, """MSI roadmap\n\n1. Local envelope: local laws mu_p for Y_p=p-c_p.\n2. Global envelope: multiplicative convolution over p<=z.\n3. Dynamic n-intervals: exact over complete Q(z)-periods.\n4. Windowed envelope: localized survivor count in actual i-windows.\n5. Prime interior: z>=sqrt(B) certifies large-prime coordinates >z.\n6. Boundary channels: small prime coordinates <=z must be handled separately.\n7. Next computational paper: sparse local representation, DP convolution, and optimized experiments.\n""")
            self.roadmap_text.configure(state=tk.DISABLED)

        def refresh_all(self) -> None:
            try:
                self.refresh_overview()
                self.refresh_local()
                self.refresh_global()
                self.refresh_window()
                self.refresh_dynamic()
                self.refresh_prime()
                self.refresh_additive()
                self.refresh_optimization()
                self.refresh_diagram()
                self.status_var.set("Updated")
            except Exception as exc:
                self.status_var.set("Error")
                messagebox.showerror("Error", str(exc))

        def refresh_overview(self) -> None:
            H, z = self.get_params()
            t0 = time.perf_counter()
            ps = primes_up_to(z)
            q = primorial(z)
            dist = global_envelope(H, z)
            dt = time.perf_counter() - t0
            mean = moment_from_distribution(dist, 1) / q
            second = moment_from_distribution(dist, 2) / q
            var = second - mean * mean
            positive = sum(c for r, c in dist.items() if r > 0)
            pos_dist = positive_global_envelope(H, z)
            self.overview_text.configure(state=tk.NORMAL)
            self.overview_text.delete("1.0", tk.END)
            self.overview_text.insert(tk.END, f"H = {list(H)}\nz = {z}\nprimes <= z = {ps}\nQ(z) = {q}\n\n")
            self.overview_text.insert(tk.END, f"Global envelope support size = {len(dist)} distinct fiber values\n")
            self.overview_text.insert(tk.END, f"Positive envelope support size = {len(pos_dist)}\n")
            self.overview_text.insert(tk.END, f"Positive probability = {positive}/{q} = {positive/q:.8f}\n")
            self.overview_text.insert(tk.END, f"Mean R = {mean:.8f}\nVariance R = {var:.8f}\n")
            self.overview_text.insert(tk.END, f"Envelope computation time = {dt:.4f} seconds\n\n")
            self.overview_text.insert(tk.END, "Envelope hierarchy:\n")
            self.overview_text.insert(tk.END, "  local laws mu_p -> global multiplicative convolution -> windowed/localized envelope -> complete-sieve prime interior.\n\n")
            self.overview_text.insert(tk.END, "Important boundary:\n")
            self.overview_text.insert(tk.END, "  Complete sieving certifies only large-prime interior coordinates > z; prime coordinates <= z need boundary channels.\n")
            self.overview_text.configure(state=tk.DISABLED)

        def refresh_local(self) -> None:
            H, z = self.get_params()
            self.clear_tree(self.local_tree)
            for p in primes_up_to(z):
                for row in local_profile(H, p):
                    self.local_tree.insert("", tk.END, values=(row["p"], row["u"], row["a"], row["rho"], row["c"], row["Y"], row["admissible"]))

        def refresh_global(self) -> None:
            H, z = self.get_params()
            q = primorial(z)
            dist = global_envelope(H, z)
            self.clear_tree(self.global_tree)
            cdf = 0
            for r, count in sorted(dist.items()):
                cdf += count
                self.global_tree.insert("", tk.END, values=(r, count, f"{count/q:.10f}", f"{cdf/q:.10f}"))

        def refresh_window(self) -> None:
            H, z = self.get_params()
            n = int(self.win_n_var.get())
            start = int(self.win_a_var.get())
            stop = int(self.win_b_var.get())
            wr = windowed_survivor_count(H, z, n, start, stop)
            self.window_text.configure(state=tk.NORMAL)
            self.window_text.delete("1.0", tk.END)
            self.window_text.insert(tk.END, f"H={list(H)}, z={z}, n={n}\n")
            self.window_text.insert(tk.END, f"i-window=[{start},{stop}], length={wr.length}\n")
            self.window_text.insert(tk.END, f"Q(z)={wr.q}\ncomplete blocks={wr.complete_blocks}, remainder={wr.remainder}\n")
            self.window_text.insert(tk.END, f"period fiber R_z(n,H)={wr.period_fiber}\n")
            self.window_text.insert(tk.END, f"complete-block contribution={wr.complete_contribution}\n")
            self.window_text.insert(tk.END, f"remainder exact={wr.remainder_exact}\n")
            self.window_text.insert(tk.END, f"windowed survivor count/bounds=[{wr.lower_bound},{wr.upper_bound}]\n")
            self.window_text.insert(tk.END, f"note: {wr.boundary_note}\n\n")
            if wr.length % wr.q == 0:
                self.window_text.insert(tk.END, "Complete-window identity verified: R_z(n,H;I)=M R_z(n,H).\n")
            else:
                self.window_text.insert(tk.END, "Arbitrary window: global envelope is localized with an explicit boundary/remainder term.\n")
            self.window_text.configure(state=tk.DISABLED)

        def refresh_dynamic(self) -> None:
            H, z = self.get_params()
            N = int(self.N_var.get())
            s = int(self.s_var.get())
            summary = interval_moment_summary(H, z, N, s)
            self.dynamic_text.configure(state=tk.NORMAL)
            self.dynamic_text.delete("1.0", tk.END)
            for k, v in summary.items():
                self.dynamic_text.insert(tk.END, f"{k}: {v}\n")
            self.dynamic_text.insert(tk.END, "\nIf remainder=0, this is exact interval exhaustion by full CRT periods.\n")
            self.dynamic_text.configure(state=tk.DISABLED)

        def refresh_prime(self) -> None:
            H, z = self.get_params()
            n = int(self.prime_n_var.get())
            i_min = int(self.i_min_var.get())
            i_max = int(self.i_max_var.get())
            self.clear_tree(self.prime_tree)
            for i in range(i_min, i_max + 1):
                if survives_packet(H, n, i, z):
                    vals = packet_values(H, n, i)
                    if all(v > 1 for v in vals):
                        B = max(vals)
                        omega_max = max(omega(v) for v in vals)
                        sqrt_b = ceil_sqrt(B)
                    else:
                        B = max(vals)
                        omega_max = "-"
                        sqrt_b = "-"
                    status = prime_interior_status(vals, z)
                    self.prime_tree.insert("", tk.END, values=(i, str(vals), B, sqrt_b, omega_max, status))

        def refresh_additive(self) -> None:
            z = int(self.z_var.get())
            N = int(self.add_N_var.get())
            d = int(self.add_d_var.get())
            ps = primes_up_to(z)
            q = primorial(z)
            prod = 1
            self.additive_text.configure(state=tk.NORMAL)
            self.additive_text.delete("1.0", tk.END)
            self.additive_text.insert(tk.END, f"Additive hyperplane x_1+...+x_d=N with nonzero coordinates mod p\nN={N}, d={d}, z={z}, Q={q}\n\n")
            for p in ps:
                c = additive_hyperplane_local_count(p, d, N)
                prod *= c
                self.additive_text.insert(tk.END, f"p={p:3d}  C_p,d(N)={c}\n")
            self.additive_text.insert(tk.END, f"\nCRT product count over Q(z): {prod}\n")
            self.additive_text.configure(state=tk.DISABLED)

        def refresh_optimization(self) -> None:
            H, z = self.get_params()
            t0 = time.perf_counter()
            dist = global_envelope(H, z)
            t1 = time.perf_counter()
            q = primorial(z)
            ps = primes_up_to(z)
            local_sizes = [len(local_law(H, p)) for p in ps]
            self.optimization_text.configure(state=tk.NORMAL)
            self.optimization_text.delete("1.0", tk.END)
            self.optimization_text.insert(tk.END, "Optimization summary\n\n")
            self.optimization_text.insert(tk.END, f"H={list(H)}, z={z}\n")
            self.optimization_text.insert(tk.END, f"#primes={len(ps)}, Q(z)={q}\n")
            self.optimization_text.insert(tk.END, f"Naive full-period enumeration would inspect Q(z) states.\n")
            self.optimization_text.insert(tk.END, f"DP convolution used local support sizes {local_sizes}.\n")
            self.optimization_text.insert(tk.END, f"Final exact support size={len(dist)}.\n")
            self.optimization_text.insert(tk.END, f"Exact envelope time={t1-t0:.6f} seconds.\n\n")
            self.optimization_text.insert(tk.END, "Techniques used:\n")
            self.optimization_text.insert(tk.END, "- lru_cache memoization for primes, primorials, local profiles, and envelopes;\n")
            self.optimization_text.insert(tk.END, "- dynamic-programming multiplicative convolution of local laws;\n")
            self.optimization_text.insert(tk.END, "- complete-window identities avoiding enumeration of long intervals;\n")
            self.optimization_text.insert(tk.END, "- direct enumeration only for small remainders/boundary checks.\n")
            self.optimization_text.configure(state=tk.DISABLED)

        def export_global_distribution(self) -> None:
            H, z = self.get_params()
            dist = global_envelope(H, z)
            q = primorial(z)
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["R", "count", "probability"])
                for r, c in sorted(dist.items()):
                    w.writerow([r, c, c / q])
            messagebox.showinfo("Exported", f"Saved {path}")

        def run_self_test_gui(self) -> None:
            ok = self_test(verbose=False)
            messagebox.showinfo("Self-test", "PASS" if ok else "FAIL")

        def clear_caches(self) -> None:
            for f in [primes_tuple, primorial_cached, residue_cloud_tuple, overlap_profile_tuple, local_profile_tuple, local_law_tuple, local_positive_law_tuple, global_envelope_tuple, positive_global_envelope_tuple]:
                f.cache_clear()
            self.status_var.set("Caches cleared")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MSI Survivor Lab v03 Diagrams")
    parser.add_argument("--self-test", action="store_true", help="run headless verification tests")
    parser.add_argument("--envelope", action="store_true", help="print exact global envelope and exit")
    parser.add_argument("--H", default="0,2", help="comma-separated shift set, e.g. 0,2,6")
    parser.add_argument("--z", type=int, default=5, help="sieve level")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test(verbose=True) else 1
    if args.envelope:
        H = parse_H(args.H)
        dist = global_envelope(H, args.z)
        q = primorial(args.z)
        print(f"H={list(H)} z={args.z} Q={q}")
        for r, c in sorted(dist.items()):
            print(f"R={r}\tcount={c}\tprob={c/q:.12g}")
        return 0
    if tk is None:
        print("tkinter is not available; use --self-test or --envelope in headless mode", file=sys.stderr)
        return 2
    app = SurvivorLabApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
