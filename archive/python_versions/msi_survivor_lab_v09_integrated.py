#!/usr/bin/env python3
"""
MSI Survivor Lab v09 Integrated Python/C++ Polynomial Laboratory
====================================================================
Tabbed GUI + headless verification tool for the Modular Survivor Interface
(MSI), primorial survivor spaces, survivor envelopes, localized/windowed
counts, layered interval exhaustion, boundary-channel completion, and the
localized survivor-polynomial coefficient law.

Version history
---------------
v01  Baseline direct verifier.  Brute-force local profiles, CRT-period checks,
     small examples for H={0}, H={0,2}, H={0,2,6}.
v02  Optimized theoretical engine.  Added memoization, dynamic-programming
     multiplicative convolution, exact global envelopes, complete-window
     identities, and headless self-tests.
v03  Diagram layer.  Added Local Law, Global Envelope Histogram, CDF,
     Windowed Profile, Prime Threshold, and Two-Prime Torus Projection plots.
v04  Known Theory Validation.  Added Mertens/zeta calibration, affine root-count
     checks, polynomial local-factor checks, Goldbach local factors, observed
     prime packet vs log-weight predictor, and localization discrepancy.
v05  Pattern Search.  Added localized complete-survivor scans over N ranges,
     residue-class summaries, zero/positive statistics, and CSV export.
v06  Layered Exhaustion.  Added partial-period exhaustion by Q(y), residual-prime
     attrition bounds, and certification of localized nonemptiness.
v07  Boundary and Multi-Layer.  Added explicit small-prime boundary-channel
     completion L_j=q and staged residual-layer accounting y0<...<z.
v08  Polynomial Law + Diagram Navigation.  Added sparse localized survivor
     polynomials, CRT-mask support construction, coefficient-law verification,
     survivor gap statistics, a Polynomial Law tab/CLI, and Matplotlib
     navigation toolbar for zoom/pan/save diagrams.
v09  Integrated Python/C++ Backend. Added C++ JSON/CSV machine-readable output,
     Python subprocess bridge, GUI backend tab, backend selector, and benchmark
     support. Python remains the front-end for diagrams and exploration; C++ is
     the fast back-end for large scans.

Run GUI:
    python msi_survivor_lab_v09_integrated.py

Run tests:
    python msi_survivor_lab_v09_integrated.py --self-test

Example polynomial-law report:
    python msi_survivor_lab_v09_integrated.py --poly-law --H 0 --z 5 --n 58 --start 0 --stop 59

The core computation is standard-library only. The diagram tab uses matplotlib
when available.  Large primorial support-polynomial construction is intentionally
bounded by --max-support / max_q to avoid accidental explosion; for larger scans
use the C++ engine or the envelope/layered methods.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except Exception:  # diagrams are optional; headless tests still work
    Figure = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None

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


# ---------------------------------------------------------------------------
# Small-prime boundary channels and multi-layer residual grouping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryCandidate:
    i: int
    side: str
    h: int
    q: int
    values: Tuple[int, ...]
    all_prime: bool
    survives_interior: bool
    note: str


def small_prime_boundary_candidates(
    H: Sequence[int], z: int, n: int, start: int, stop: int
) -> Tuple[BoundaryCandidate, ...]:
    """Enumerate explicit small-prime boundary channels for the symmetric packet.

    Boundary channels are L_j^+(i)=q or L_j^-(i)=q with q<=z prime.
    For L_j^+(i)=i+h+1=q, i=q-h-1.
    For L_j^-(i)=n-i-h+1=q, i=n-h+1-q.
    """
    out: List[BoundaryCandidate] = []
    seen = set()
    Ht = tuple(H)
    for q in primes_tuple(int(z)):
        for h in Ht:
            candidates = [
                ("plus", q - h - 1),
                ("minus", int(n) - h + 1 - q),
            ]
            for side, i in candidates:
                if i < start or i > stop:
                    continue
                key = (i, side, h, q)
                if key in seen:
                    continue
                seen.add(key)
                vals = tuple(packet_values(Ht, int(n), int(i)))
                allp = all(v > 1 and is_prime(v) for v in vals)
                interior = survives_packet(Ht, int(n), int(i), int(z))
                if allp and not interior:
                    note = "boundary prime packet missed by large-prime interior sieve"
                elif allp and interior:
                    note = "prime packet also survives interior sieve (unexpected when boundary coordinate <= z)"
                else:
                    note = "boundary candidate but remaining coordinates are not all prime"
                out.append(BoundaryCandidate(int(i), side, int(h), int(q), vals, allp, interior, note))
    return tuple(sorted(out, key=lambda c: (c.i, c.q, c.side, c.h)))


def direct_window_survivor_count(H: Sequence[int], z: int, n: int, start: int, stop: int, *, max_direct: int = 500_000) -> int | None:
    length = int(stop) - int(start) + 1
    if length < 0:
        return 0
    if length > max_direct:
        return None
    return sum(1 for i in range(int(start), int(stop) + 1) if survives_packet(H, int(n), i, int(z)))


def boundary_completion_report(H: Sequence[int], z: int, n: int, start: int, stop: int) -> str:
    Ht = tuple(H)
    direct_count = direct_window_survivor_count(Ht, z, n, start, stop)
    candidates = small_prime_boundary_candidates(Ht, z, n, start, stop)
    positives = [c for c in candidates if c.all_prime]
    if direct_count is None:
        classification = "interior count not directly enumerated; boundary candidates listed"
    elif direct_count > 0:
        classification = "interior positive"
    elif positives:
        classification = "boundary-only positive"
    else:
        classification = "no interior or boundary prime packet found in this window"
    lines: List[str] = []
    lines.append("Small-prime boundary completion report")
    lines.append(f"H={list(Ht)}, n={n}, z={z}, window=[{start},{stop}]")
    lines.append(f"direct large-prime interior survivor count={direct_count}")
    lines.append(f"boundary candidates={len(candidates)}, boundary prime packets={len(positives)}")
    lines.append(f"classification={classification}")
    lines.append("")
    lines.append("Boundary candidates:")
    lines.append("i\tside\th\tq\tall_prime\tsurvives_interior\tvalues\tnote")
    for c in candidates:
        lines.append(f"{c.i}\t{c.side}\t{c.h}\t{c.q}\t{c.all_prime}\t{c.survives_interior}\t{list(c.values)}\t{c.note}")
    lines.append("")
    lines.append("Rule: complete sieving certifies only prime coordinates > z. True prime packets with a coordinate q<=z lie on the explicit boundary channels L_j(i)=q and are recovered here.")
    return "\n".join(lines)


@dataclass(frozen=True)
class MultiLayerGroup:
    lower: int
    upper: int
    primes: Tuple[int, ...]
    attrition_bound: int


@dataclass(frozen=True)
class MultiLayerResult:
    base: LayeredResult
    levels: Tuple[int, ...]
    groups: Tuple[MultiLayerGroup, ...]
    total_attrition_bound: int
    certified_lower_bound: int


def parse_levels_text(text: str, z: int) -> Tuple[int, ...]:
    if not str(text).strip() or str(text).strip().lower() == "auto":
        return tuple()
    vals = sorted({int(x.strip()) for x in str(text).replace(';', ',').split(',') if x.strip()})
    vals = [v for v in vals if v < int(z)]
    return tuple(vals)


def multi_layer_attrition_summary(
    H: Sequence[int], z: int, n: int, start: int, stop: int, levels: Sequence[int] | None = None
) -> MultiLayerResult:
    """Group residual attrition by several sieve layers.

    The rigorous bound is still the union-bound attrition from the base level to z;
    grouping is diagnostic and helps select future refinement levels. It is exact as
    an accounting of the same bound, not a stronger theorem by itself.
    """
    length = int(stop) - int(start) + 1
    if levels:
        ys = tuple(sorted({int(v) for v in levels if int(v) < int(z)}))
        y0 = ys[0] if ys else recommended_base_level(length, z)
    else:
        y0 = recommended_base_level(length, z)
        ys = (y0,)
    if not ys or ys[0] != y0:
        ys = (y0,) + tuple(v for v in ys if v > y0)
    if ys[-1] != int(z):
        all_levels = tuple(ys) + (int(z),)
    else:
        all_levels = tuple(ys)
    base = layered_exhaustion_summary(H, z, n, start, stop, y=y0)
    groups: List[MultiLayerGroup] = []
    total = 0
    for lo, hi in zip(all_levels[:-1], all_levels[1:]):
        ps = tuple(p for p in residual_primes_tuple(lo, hi))
        subtotal = 0
        for p in ps:
            c_p = c_formula_state(H, p, n)
            full_cycles = base.complete_Qy_blocks // p
            remainder_blocks = base.complete_Qy_blocks % p
            subtotal += base.base_period_fiber * (full_cycles * c_p + min(remainder_blocks, c_p))
        groups.append(MultiLayerGroup(int(lo), int(hi), ps, int(subtotal)))
        total += subtotal
    lower = max(0, base.base_survivor_count - total)
    return MultiLayerResult(base=base, levels=all_levels, groups=tuple(groups), total_attrition_bound=total, certified_lower_bound=lower)


def multi_layer_report(result: MultiLayerResult) -> str:
    lines: List[str] = []
    b = result.base
    lines.append("Multi-layer residual attrition report")
    lines.append(f"H={list(b.H)}, n={b.n}, z={b.z}, base y0={b.y}")
    lines.append(f"window=[{b.start},{b.stop}], length={b.length}")
    lines.append(f"levels={list(result.levels)}")
    lines.append(f"Q(y0)={b.Q_y}, complete Q(y0)-blocks M={b.complete_Qy_blocks}, base survivors={b.base_survivor_count}")
    lines.append(f"total grouped attrition bound={result.total_attrition_bound}")
    lines.append(f"certified lower bound={result.certified_lower_bound}")
    lines.append(f"certified nonempty={result.certified_lower_bound > 0}")
    lines.append("")
    lines.append("Layer groups:")
    lines.append("lower\tupper\tprimes\tattrition_bound")
    for g in result.groups:
        lines.append(f"{g.lower}\t{g.upper}\t{list(g.primes)}\t{g.attrition_bound}")
    lines.append("")
    lines.append("Note: grouping is a diagnostic accounting of the rigorous union-bound attrition from the base level. It helps choose layers for later sharper inclusion-exclusion or exact residual tests.")
    return "\n".join(lines)


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
# Optional C++ backend bridge (v09)
# ---------------------------------------------------------------------------

def default_cpp_executable() -> str:
    """Best-effort default executable name/path for the C++ backend."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "msi_survivor_engine_v09.exe",
        here / "msi_survivor_engine_v09",
        Path.cwd() / "msi_survivor_engine_v09.exe",
        Path.cwd() / "msi_survivor_engine_v09",
        Path("msi_survivor_engine_v09.exe"),
        Path("msi_survivor_engine_v09"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "msi_survivor_engine_v09.exe"


def run_cpp_backend(exe_path: str, args: Sequence[str], timeout: int = 120) -> dict:
    """Run the C++ backend with --json and parse its result.

    The C++ engine is intentionally optional. If it is not available, all Python
    computations still work, but large scans will be slower.
    """
    exe = str(exe_path or default_cpp_executable())
    cmd = [exe] + list(args)
    if "--json" not in cmd:
        cmd.append("--json")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"C++ backend exited with {proc.returncode}").strip())
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("C++ backend produced no output")
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"C++ backend did not produce valid JSON:\n{out[:1000]}") from exc


def cpp_args_common(H: Sequence[int], z: int, n: int, start: int, stop: int) -> List[str]:
    return ["--H", ",".join(str(x) for x in H), "--z", str(int(z)), "--n", str(int(n)), "--start", str(int(start)), "--stop", str(int(stop))]


def backend_count_report(exe: str, H: Sequence[int], z: int, n: int, start: int, stop: int) -> str:
    data = run_cpp_backend(exe, ["--count"] + cpp_args_common(H, z, n, start, stop))
    lines = ["C++ backend localized count", "=" * 34]
    lines.append(json.dumps(data, indent=2))
    return "\n".join(lines)


def backend_layered_report(exe: str, H: Sequence[int], z: int, y: int, n: int, start: int, stop: int) -> str:
    data = run_cpp_backend(exe, ["--layered", "--y", str(int(y))] + cpp_args_common(H, z, n, start, stop))
    lines = ["C++ backend layered report", "=" * 32]
    lines.append(json.dumps(data, indent=2))
    return "\n".join(lines)


def backend_boundary_report(exe: str, H: Sequence[int], z: int, n: int, start: int, stop: int) -> str:
    data = run_cpp_backend(exe, ["--boundary"] + cpp_args_common(H, z, n, start, stop))
    lines = ["C++ backend boundary completion", "=" * 34]
    lines.append(json.dumps(data, indent=2))
    return "\n".join(lines)


def backend_polynomial_report(exe: str, H: Sequence[int], z: int, n: int, start: int, stop: int, max_q: int = 300000) -> str:
    data = run_cpp_backend(exe, ["--poly-law", "--max-q", str(int(max_q))] + cpp_args_common(H, z, n, start, stop))
    lines = ["C++ backend polynomial-law report", "=" * 37]
    lines.append(json.dumps(data, indent=2))
    return "\n".join(lines)


def backend_benchmark(exe: str, H: Sequence[int], z: int, n: int, start: int, stop: int) -> str:
    lines = ["Python vs C++ backend benchmark", "=" * 33]
    t0 = time.perf_counter()
    py_count = windowed_survivor_count(H, z, n, start, stop).lower_bound
    py_dt = time.perf_counter() - t0
    t1 = time.perf_counter()
    data = run_cpp_backend(exe, ["--count"] + cpp_args_common(H, z, n, start, stop))
    cpp_dt = time.perf_counter() - t1
    cpp_count = data.get("count")
    lines.append(f"H={list(H)}, z={z}, n={n}, I=[{start},{stop}]")
    lines.append(f"Python count={py_count}, time={py_dt:.6f} s")
    lines.append(f"C++ count={cpp_count}, time={cpp_dt:.6f} s")
    lines.append(f"match={py_count == cpp_count}")
    if cpp_dt > 0:
        lines.append(f"speed ratio Python/C++={py_dt/cpp_dt:.3f}")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Known-theory validation and localized nonemptiness experiments
# ---------------------------------------------------------------------------

EULER_GAMMA = 0.5772156649015328606


def unary_mertens_ratio(z: int) -> Tuple[float, float]:
    """Return phi(Q)/Q and (phi(Q)/Q)*e^gamma*log(z)."""
    z = int(z)
    density = 1.0
    for p in primes_tuple(z):
        density *= (1.0 - 1.0 / p)
    ratio = density * math.exp(EULER_GAMMA) * math.log(z) if z > 1 else float('nan')
    return density, ratio


def c_formula_state(H: Sequence[int], p: int, n: int) -> int:
    u = (n + 2) % p
    for row in local_profile_tuple(tuple(H), int(p)):
        if row[1] == u:
            return row[4]
    raise RuntimeError("state not found")


def affine_direct_root_count(H: Sequence[int], p: int, n: int) -> int:
    """Count forbidden i mod p directly from the packet polynomial roots."""
    count = 0
    for i in range(p):
        if any(((i + h + 1) % p == 0) or ((n - i - h + 1) % p == 0) for h in H):
            count += 1
    return count


def affine_root_count_table(H: Sequence[int], z: int, n: int) -> List[Tuple[int, int, int, bool]]:
    rows = []
    for p in primes_tuple(int(z)):
        cf = c_formula_state(H, p, n)
        cd = affine_direct_root_count(H, p, n)
        rows.append((p, cf, cd, cf == cd))
    return rows


def finite_singular_product(H: Sequence[int], z: int, n: int) -> float:
    """Finite Hardy--Littlewood-style normalized product for the affine packet."""
    r = 2 * len(H)
    prod = 1.0
    for p in primes_tuple(int(z)):
        c = c_formula_state(H, p, n)
        numerator = 1.0 - c / p
        denominator = (1.0 - 1.0 / p) ** r
        prod *= numerator / denominator
    return prod


def polynomial_root_count(expr: str, p: int) -> int:
    roots = 0
    for x in range(p):
        if eval_expr(expr, {"x": x}) % p == 0:
            roots += 1
    return roots


def polynomial_validation_table(expr: str, z: int) -> List[Tuple[int, int, int, float]]:
    """Return p, roots, local survivors, normalized BH-type local factor for one polynomial."""
    rows = []
    for p in primes_tuple(int(z)):
        roots = polynomial_root_count(expr, p)
        survivors = p - roots
        factor = (1.0 - roots / p) / (1.0 - 1.0 / p) if p > 1 else float('nan')
        rows.append((p, roots, survivors, factor))
    return rows


def goldbach_sigma_p(p: int, N: int) -> float:
    c = additive_hyperplane_local_count(p, 2, N)
    return p * c / ((p - 1) ** 2)


def goldbach_singular_trunc(N: int, z: int) -> Tuple[float, List[Tuple[int, int, float]]]:
    prod = 1.0
    rows = []
    for p in primes_tuple(int(z)):
        c = additive_hyperplane_local_count(p, 2, N)
        sigma = goldbach_sigma_p(p, N)
        prod *= sigma
        rows.append((p, c, sigma))
    return prod, rows


def actual_prime_packet_count(H: Sequence[int], n: int, start: int, stop: int) -> Tuple[int, List[Tuple[int, Tuple[int, ...]]]]:
    count = 0
    examples: List[Tuple[int, Tuple[int, ...]]] = []
    for i in range(start, stop + 1):
        vals = tuple(packet_values(H, n, i))
        if all(v > 1 and is_prime(v) for v in vals):
            count += 1
            if len(examples) < 10:
                examples.append((i, vals))
    return count, examples


def log_weight_prediction(H: Sequence[int], z: int, n: int, start: int, stop: int) -> float:
    sig = finite_singular_product(H, z, n)
    total = 0.0
    for i in range(start, stop + 1):
        vals = packet_values(H, n, i)
        if all(v > 1 for v in vals):
            denom = 1.0
            ok = True
            for v in vals:
                lv = math.log(v)
                if lv <= 0:
                    ok = False
                    break
                denom *= lv
            if ok:
                total += 1.0 / denom
    return sig * total


def localization_discrepancy(H: Sequence[int], z: int, n: int, start: int, stop: int) -> Dict[str, float | int | str | None]:
    wr = windowed_survivor_count(H, z, n, start, stop)
    predicted = (wr.length / wr.q) * wr.period_fiber
    exact = wr.lower_bound if wr.lower_bound == wr.upper_bound else None
    discrepancy = None if exact is None else exact - predicted
    return {
        "Q": wr.q,
        "length": wr.length,
        "period_fiber": wr.period_fiber,
        "global_scaled_prediction": predicted,
        "exact_window_count": exact,
        "lower_bound": wr.lower_bound,
        "upper_bound": wr.upper_bound,
        "discrepancy": discrepancy,
        "note": wr.boundary_note,
    }


def goldbach_complete_survivor_scan(N_min: int, N_max: int, *, max_rows: int = 500) -> List[Tuple[int, int, int, int | None, Tuple[int, int] | None]]:
    """Scan even N for H={0}. Returns N,z,count,first_i,first_pair.

    This is intentionally direct, because the target is localized complete-survivor
    nonemptiness in the actual Goldbach window, not full-period distribution.
    """
    rows = []
    if N_min % 2:
        N_min += 1
    for N in range(N_min, N_max + 1, 2):
        z = ceil_sqrt(N)
        n = N - 2
        start_i = 1  # x=i+1 >= 2
        stop_i = N - 3  # y=N-(i+1) >= 2 -> i <= N-3
        count = 0
        first_i = None
        first_pair = None
        for i in range(start_i, stop_i + 1):
            if survives_packet((0,), n, i, z):
                vals = packet_values((0,), n, i)
                if all(v > 1 for v in vals):
                    count += 1
                    if first_i is None:
                        first_i = i
                        first_pair = (vals[0], vals[1])
        rows.append((N, z, count, first_i, first_pair))
        if len(rows) >= max_rows:
            break
    return rows



@dataclass(frozen=True)
class PatternRow:
    N: int
    z: int
    B: int
    start_i: int
    stop_i: int
    count: int
    first_i: int | None
    first_packet: Tuple[int, ...] | None
    residue: int | None = None


def feasible_index_window(H: Sequence[int], N: int) -> Tuple[int, int]:
    """Return the integer i-window where all packet coordinates are > 1.

    The diagonal parameter is n=N-2, so each pair has sum N.
    For every h in H we need
        i+h+1 > 1 and n-i-h+1 > 1.
    Hence i >= 1-h and i <= n-h-1 = N-h-3.
    """
    if not H:
        return (1, 0)
    n = N - 2
    start = max(1 - h for h in H)
    stop = min(n - h - 1 for h in H)
    return start, stop


def coordinate_bound_on_window(H: Sequence[int], N: int, start: int, stop: int) -> int:
    """Exact maximum coordinate size on a finite window."""
    if start > stop:
        return 0
    n = N - 2
    B = 0
    # For affine coordinates the max is at endpoints, but we keep this explicit
    # and safe because the windows used experimentally are modest.
    for i in (start, stop):
        for v in packet_values(H, n, i):
            B = max(B, int(v))
    return B


def scan_complete_survivors(
    H: Sequence[int],
    N_min: int,
    N_max: int,
    *,
    parity: int = 0,
    modulus: int = 6,
    max_rows: int | None = None,
) -> List[PatternRow]:
    """Scan localized complete-survivor nonemptiness over diagonal sums N.

    For each N, choose the actual positivity window for the symmetric packet,
    compute B=max coordinate on that window, set z=ceil(sqrt(B)), and count
    survivors in that window.  Surviving packets are then prime-certified in
    the large-prime interior, except for explicit small-prime boundary channels
    that the sieve excludes.
    """
    Ht = tuple(H)
    rows: List[PatternRow] = []
    for N in range(N_min, N_max + 1):
        if N % 2 != parity:
            continue
        start, stop = feasible_index_window(Ht, N)
        if start > stop:
            B = 0
            z = 2
            rows.append(PatternRow(N, z, B, start, stop, 0, None, None, N % modulus if modulus else None))
        else:
            B = coordinate_bound_on_window(Ht, N, start, stop)
            z = max(2, ceil_sqrt(B))
            n = N - 2
            count = 0
            first_i = None
            first_packet = None
            for i in range(start, stop + 1):
                if survives_packet(Ht, n, i, z):
                    vals = tuple(packet_values(Ht, n, i))
                    if all(v > 1 for v in vals):
                        count += 1
                        if first_i is None:
                            first_i = i
                            first_packet = vals
            rows.append(PatternRow(N, z, B, start, stop, count, first_i, first_packet, N % modulus if modulus else None))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def pattern_summary(rows: Sequence[PatternRow], *, modulus: int = 6) -> str:
    if not rows:
        return "No rows.\n"
    positives = [r for r in rows if r.count > 0]
    zeros = [r for r in rows if r.count == 0]
    counts = [r.count for r in rows]
    pos_counts = [r.count for r in positives]
    residues_pos = sorted({r.N % modulus for r in positives}) if modulus else []
    residues_zero = sorted({r.N % modulus for r in zeros}) if modulus else []
    large_zero = [r for r in zeros if r.N >= 50]
    lines = []
    lines.append("Localized complete-survivor nonemptiness pattern summary")
    lines.append("=" * 64)
    lines.append(f"Rows scanned: {len(rows)}")
    lines.append(f"Positive rows: {len(positives)}")
    lines.append(f"Zero rows: {len(zeros)}")
    lines.append(f"Positive residues mod {modulus}: {residues_pos}")
    lines.append(f"Zero residues mod {modulus}: {residues_zero}")
    if counts:
        lines.append(f"Count min/mean/median/max: {min(counts)} / {sum(counts)/len(counts):.4g} / {sorted(counts)[len(counts)//2]} / {max(counts)}")
    if pos_counts:
        lines.append(f"Positive count min/mean/max: {min(pos_counts)} / {sum(pos_counts)/len(pos_counts):.4g} / {max(pos_counts)}")
    lines.append(f"Zeros with N>=50: {len(large_zero)}")
    if zeros:
        lines.append("First zero rows:")
        for r in zeros[:20]:
            lines.append(f"  N={r.N}, z={r.z}, B={r.B}, window=[{r.start_i},{r.stop_i}], residue={r.residue}")
    if positives:
        lines.append("First positive rows:")
        for r in positives[:10]:
            lines.append(f"  N={r.N}, z={r.z}, count={r.count}, first_i={r.first_i}, first_packet={r.first_packet}, residue={r.residue}")
    lines.append("")
    lines.append("Interpretation note:")
    lines.append("  Small zero cases often reflect boundary effects: the positivity window is tiny, complete sieving excludes small prime coordinates <= z, and explicit small-prime boundary channels are not included in the large-prime interior count.")
    lines.append("  Persistent zero classes at larger N are more significant; they suggest local admissibility restrictions or deeper localized nonemptiness structure.")
    return "\n".join(lines) + "\n"


def write_pattern_csv(rows: Sequence[PatternRow], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "z", "B", "start_i", "stop_i", "count", "first_i", "first_packet", "residue"])
        for r in rows:
            w.writerow([r.N, r.z, r.B, r.start_i, r.stop_i, r.count, r.first_i, r.first_packet, r.residue])

def validation_report(H: Sequence[int], z: int, *, n: int = 58, start: int = 0, stop: int = 59, N: int = 60, poly: str = "x*x+1") -> str:
    lines: List[str] = []
    lines.append("Known Theory Validation / Calibration Report")
    lines.append("=" * 58)
    lines.append(f"H={list(H)}, z={z}, n={n}, i-window=[{start},{stop}], N={N}, polynomial f(x)={poly!r}")
    lines.append("")

    density, ratio = unary_mertens_ratio(z)
    lines.append("1. Unary zeta/Mertens calibration")
    lines.append(f"   phi(Q(z))/Q(z) = prod_{{p<=z}}(1-1/p) = {density:.12g}")
    lines.append(f"   Mertens ratio density * exp(gamma) * log(z) = {ratio:.12g}  (expected to drift toward 1 as z grows)")
    lines.append("")

    lines.append("2. Affine packet root-count calibration")
    all_ok = True
    for p, cf, cd, ok in affine_root_count_table(H, z, n):
        all_ok = all_ok and ok
        lines.append(f"   p={p:3d}: overlap formula c_p={cf:3d}, direct packet-root count={cd:3d}, match={ok}")
    lines.append(f"   overall affine root-count check: {'PASS' if all_ok else 'FAIL'}")
    lines.append("")

    lines.append("3. Finite normalized affine singular product")
    try:
        sig = finite_singular_product(H, z, n)
        lines.append(f"   S_z^form(n,H) = {sig:.12g}")
    except Exception as exc:
        lines.append(f"   could not compute product: {exc}")
    lines.append("")

    lines.append("4. Polynomial/Bateman--Horn local input calibration")
    try:
        prod = 1.0
        for p, roots, survivors, factor in polynomial_validation_table(poly, z):
            prod *= factor
            lines.append(f"   p={p:3d}: roots={roots:3d}, survivors={survivors:3d}, normalized local factor={factor:.8g}")
        lines.append(f"   finite product of normalized polynomial local factors = {prod:.12g}")
    except Exception as exc:
        lines.append(f"   polynomial check failed: {exc}")
    lines.append("")

    lines.append("5. Goldbach additive local-factor calibration")
    try:
        gprod, grows = goldbach_singular_trunc(N, z)
        for p, c, sigma in grows:
            lines.append(f"   p={p:3d}: C_p,2(N)={c:3d}, sigma_p={sigma:.8g}")
        lines.append(f"   finite Goldbach local product = {gprod:.12g}")
    except Exception as exc:
        lines.append(f"   Goldbach check failed: {exc}")
    lines.append("")

    lines.append("6. Observed prime packets versus survivor/log-weight prediction")
    try:
        actual, examples = actual_prime_packet_count(H, n, start, stop)
        pred = log_weight_prediction(H, z, n, start, stop)
        lines.append(f"   actual prime packets in window = {actual}")
        lines.append(f"   finite singular-product/log-weight prediction = {pred:.12g}")
        lines.append(f"   ratio actual/prediction = {actual / pred if pred > 0 else float('nan'):.12g}")
        lines.append(f"   first examples = {examples}")
    except Exception as exc:
        lines.append(f"   observed-vs-predicted check failed: {exc}")
    lines.append("")

    lines.append("7. Localization discrepancy")
    try:
        disc = localization_discrepancy(H, z, n, start, stop)
        for k, v in disc.items():
            lines.append(f"   {k}: {v}")
    except Exception as exc:
        lines.append(f"   localization discrepancy check failed: {exc}")
    lines.append("")

    lines.append("Interpretation")
    lines.append("   These are calibration checks, not borrowed proofs. Passing them means the MSI survivor-envelope layer reproduces the standard local arithmetic objects in classical cases.")
    lines.append("   The independent frontier remains localized complete-survivor nonemptiness inside the actual bounded window.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layered interval exhaustion and residual-prime attrition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayeredResidualRow:
    p: int
    c_p: int
    full_cycles: int
    remainder_blocks: int
    killed_bound_per_base_class: int
    killed_bound_total: int


@dataclass(frozen=True)
class LayeredResult:
    H: Tuple[int, ...]
    n: int
    z: int
    y: int
    start: int
    stop: int
    length: int
    Q_y: int
    Q_z: int
    complete_Qy_blocks: int
    unused_remainder: int
    base_period_fiber: int
    base_survivor_count: int
    residual_primes: Tuple[int, ...]
    residual_attrition_bound: int
    certified_lower_bound: int
    expected_main: float
    exact_count: int | None
    certified_nonempty: bool
    rows: Tuple[LayeredResidualRow, ...]
    note: str


def recommended_base_level(length: int, z: int) -> int:
    """Largest integer y<=z with Q(y)<=length, using prime thresholds.

    If even Q(2)>length, returns 1. Then Q(1)=1 and the base period is trivial.
    """
    length = int(length)
    z = int(z)
    best = 1
    for p in primes_tuple(z):
        if primorial(p) <= max(length, 1):
            best = p
        else:
            break
    return best


def residual_primes_tuple(y: int, z: int) -> Tuple[int, ...]:
    return tuple(p for p in primes_tuple(int(z)) if p > int(y))


def layered_exhaustion_summary(
    H: Sequence[int],
    z: int,
    n: int,
    start: int,
    stop: int,
    y: int | None = None,
    *,
    max_direct: int = 200_000,
) -> LayeredResult:
    """Rigorous layered exhaustion certificate for an i-window.

    The window [start,stop] is decomposed into complete Q(y)-blocks plus a
    leftover remainder. Complete Q(y)-blocks produce M*R_y base survivors.
    Residual primes y<p<=z remove at most the union-bound attrition given by
    the cycling lemma. If the lower bound is positive, full z-level localized
    nonemptiness is certified.
    """
    if stop < start:
        raise ValueError("stop must be >= start")
    Ht = tuple(H)
    z = int(z)
    length = stop - start + 1
    if y is None:
        y = recommended_base_level(length, z)
    y = int(y)
    if y >= z:
        # Keep y as an intermediate level; if user chooses y>=z, there is no residual layer.
        y = z
    Q_y = primorial(y)
    Q_z = primorial(z)
    M = length // Q_y
    rem = length % Q_y
    R_y = R_value(Ht, y, n) if y >= 2 else 1
    base_count = M * R_y
    residuals = residual_primes_tuple(y, z)
    rows = []
    total_attrition = 0
    expected = float(base_count)
    for p in residuals:
        c_p = c_formula_state(Ht, p, n)
        full_cycles = M // p
        remainder_blocks = M % p
        killed_per_class = full_cycles * c_p + min(remainder_blocks, c_p)
        killed_total = R_y * killed_per_class
        rows.append(LayeredResidualRow(p, c_p, full_cycles, remainder_blocks, killed_per_class, killed_total))
        total_attrition += killed_total
        expected *= (1.0 - c_p / p)
    lower = max(0, base_count - total_attrition)
    exact_count = None
    note = "certificate uses complete Q(y)-blocks; leftover remainder is not used"
    complete_len = M * Q_y
    if complete_len > 0 and complete_len <= max_direct:
        exact_count = sum(1 for i in range(start, start + complete_len) if survives_packet(Ht, n, i, z))
        note += "; exact full z-count on complete base segment was enumerated"
    elif complete_len == 0:
        note = "window contains no complete Q(y)-block; layered base production is zero"
    else:
        note += "; exact enumeration skipped because complete base segment is large"
    return LayeredResult(
        H=Ht,
        n=int(n),
        z=z,
        y=y,
        start=int(start),
        stop=int(stop),
        length=length,
        Q_y=Q_y,
        Q_z=Q_z,
        complete_Qy_blocks=M,
        unused_remainder=rem,
        base_period_fiber=R_y,
        base_survivor_count=base_count,
        residual_primes=residuals,
        residual_attrition_bound=total_attrition,
        certified_lower_bound=lower,
        expected_main=expected,
        exact_count=exact_count,
        certified_nonempty=lower > 0,
        rows=tuple(rows),
        note=note,
    )


def layered_report(result: LayeredResult) -> str:
    lines = []
    lines.append("Layered interval exhaustion report")
    lines.append(f"H={list(result.H)}, n={result.n}, z={result.z}, y={result.y}")
    lines.append(f"window=[{result.start},{result.stop}], length={result.length}")
    lines.append(f"Q(y)={result.Q_y}, Q(z)={result.Q_z}")
    lines.append(f"complete Q(y)-blocks M={result.complete_Qy_blocks}, unused remainder={result.unused_remainder}")
    lines.append(f"base period fiber R_y={result.base_period_fiber}")
    lines.append(f"base survivor count M*R_y={result.base_survivor_count}")
    lines.append(f"residual primes y<p<=z: {list(result.residual_primes)}")
    lines.append(f"residual attrition upper bound={result.residual_attrition_bound}")
    lines.append(f"certified lower bound={result.certified_lower_bound}")
    lines.append(f"expected residual main term={result.expected_main:.6g}")
    lines.append(f"certified nonempty={result.certified_nonempty}")
    lines.append(f"exact count on complete base segment={result.exact_count}")
    lines.append(f"note: {result.note}")
    lines.append("")
    lines.append("Residual attrition rows:")
    lines.append("p\tc_p\tfull_cycles\tremainder_blocks\tkilled/class\tkilled_total")
    for r in result.rows:
        lines.append(f"{r.p}\t{r.c_p}\t{r.full_cycles}\t{r.remainder_blocks}\t{r.killed_bound_per_base_class}\t{r.killed_bound_total}")
    lines.append("")
    lines.append("Criterion: if certified_lower_bound > 0, then the window contains at least one full z-level survivor in its complete Q(y)-block portion.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Localized survivor polynomial / coefficient-law tools
# ---------------------------------------------------------------------------

def crt_idempotents(primes: Sequence[int]) -> Dict[int, int]:
    """CRT idempotents e_p modulo Q for squarefree Q=prod primes.

    e_p == 1 mod p and e_p == 0 mod ell for ell != p.
    """
    Q = 1
    for p in primes:
        Q *= int(p)
    out: Dict[int, int] = {}
    for p in primes:
        m = Q // p
        inv = pow(m % p, -1, p)
        out[int(p)] = (m * inv) % Q
    return out


def local_survivor_residues(H: Sequence[int], p: int, n: int) -> Tuple[int, ...]:
    """Residues i mod p that survive the symmetric packet at state n."""
    forbidden = set()
    for h in H:
        forbidden.add((-h - 1) % p)              # i+h+1 == 0
        forbidden.add((n - h + 1) % p)          # n-i-h+1 == 0 => i == n-h+1
    return tuple(a for a in range(p) if a not in forbidden)


def survivor_support_crt(H: Sequence[int], z: int, n: int, *, max_q: int = 300000) -> Tuple[int, Tuple[int, ...]]:
    """Return Q and sorted support of S_{Q,n,H}(X) if Q<=max_q.

    This builds the support of the sparse survivor polynomial
    S_Q(X)=sum_{r in R_Q} X^r in Z[X]/(X^Q-1).
    """
    ps = primes_tuple(int(z))
    Q = primorial(int(z))
    if Q > max_q:
        raise ValueError(f"Q(z)={Q} exceeds max_q={max_q}; use envelope/layered/C++ methods")
    e = crt_idempotents(ps)
    support = {0}
    for p in ps:
        residues = local_survivor_residues(tuple(H), p, int(n))
        next_support = set()
        ep = e[p]
        for r in support:
            for a in residues:
                next_support.add((r + a * ep) % Q)
        support = next_support
    return Q, tuple(sorted(support))


def window_residue_counter(start: int, stop: int, Q: int) -> Counter:
    """Residue multiplicities of an integer interval modulo Q."""
    if stop < start:
        return Counter()
    length = stop - start + 1
    full, rem = divmod(length, Q)
    out: Counter = Counter({r: full for r in range(Q)}) if full else Counter()
    base = start % Q
    for k in range(rem):
        out[(base + k) % Q] += 1
    return out


def polynomial_window_count(H: Sequence[int], z: int, n: int, start: int, stop: int, *, max_q: int = 300000) -> Dict[str, object]:
    """Verify R_z(n,H;I)=[X^0] W_I(X) S_Q(X^{-1})."""
    Q, support = survivor_support_crt(H, z, n, max_q=max_q)
    wc = window_residue_counter(start, stop, Q)
    coeff_count = sum(wc.get(r, 0) for r in support)
    direct = sum(1 for i in range(start, stop + 1) if survives_packet(H, n, i, z)) if (stop - start + 1) <= max_q else None
    gaps = survivor_gaps_from_support(support, Q)
    return {
        "H": tuple(H),
        "z": int(z),
        "n": int(n),
        "Q": Q,
        "window": (int(start), int(stop)),
        "window_length": int(stop - start + 1),
        "support_size": len(support),
        "R_period": R_value(H, z, n),
        "coefficient_count": coeff_count,
        "direct_count": direct,
        "coefficient_matches_direct": (direct is None or direct == coeff_count),
        "max_gap": gaps["max_gap"],
        "mean_gap": gaps["mean_gap"],
        "first_support": support[:20],
    }


def survivor_gaps_from_support(support: Sequence[int], Q: int) -> Dict[str, object]:
    """Gap statistics for survivor residues on the cyclic group Z/QZ."""
    if not support:
        return {"max_gap": None, "mean_gap": None, "gaps": ()}
    ss = sorted(set(int(x) % Q for x in support))
    gaps = []
    for a, b in zip(ss, ss[1:]):
        gaps.append(b - a)
    gaps.append((ss[0] + Q) - ss[-1])
    return {
        "max_gap": max(gaps),
        "mean_gap": sum(gaps) / len(gaps),
        "gaps": tuple(gaps),
    }


def polynomial_law_report(H: Sequence[int], z: int, n: int, start: int, stop: int, *, max_q: int = 300000) -> str:
    lines = []
    lines.append("Localized survivor-polynomial coefficient-law report")
    lines.append("=" * 66)
    try:
        data = polynomial_window_count(H, z, n, start, stop, max_q=max_q)
    except Exception as exc:
        return f"Polynomial law report failed: {exc}\n"
    lines.append(f"H={list(data['H'])}, z={data['z']}, n={data['n']}, Q={data['Q']}")
    lines.append(f"window={data['window']}, length={data['window_length']}")
    lines.append(f"support size |R_Q|={data['support_size']} ; R_period={data['R_period']}")
    lines.append(f"coefficient count [X^0] W_I(X) S_Q(X^-1) = {data['coefficient_count']}")
    lines.append(f"direct enumeration count = {data['direct_count']}")
    lines.append(f"coefficient matches direct = {data['coefficient_matches_direct']}")
    lines.append(f"max survivor gap modulo Q = {data['max_gap']}")
    lines.append(f"mean survivor gap modulo Q = {data['mean_gap']}")
    lines.append(f"first support residues = {list(data['first_support'])}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  The global envelope is the evaluation S_Q(1).")
    lines.append("  The localized/windowed count is the cyclic constant-coefficient pairing")
    lines.append("      R_Q(n,H;I) = [X^0]_Q W_I(X) S_Q(X^{-1}).")
    lines.append("  Positivity of this coefficient is exactly localized survivor nonemptiness.")
    return "\n".join(lines) + "\n"


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

    layered = layered_exhaustion_summary((0,), 5, 58, 0, 59, y=3)
    tests.append((layered.base_survivor_count == 20 and layered.residual_attrition_bound == 4 and layered.certified_lower_bound == 16 and layered.exact_count == 16, f"layered exhaustion H={{0}}, z=5, y=3: lower={layered.certified_lower_bound}, exact={layered.exact_count}"))

    bc = small_prime_boundary_candidates((0,), 2, 2, 1, 1)
    tests.append((any(c.all_prime and c.i == 1 for c in bc), f"boundary channel N=4/H={{0}} candidates={bc}"))
    ml = multi_layer_attrition_summary((0,), 5, 58, 0, 59, levels=(3,))
    tests.append((ml.certified_lower_bound == 16 and ml.total_attrition_bound == 4, f"multi-layer grouped attrition lower={ml.certified_lower_bound}"))

    # Polynomial coefficient-law check.
    pdata = polynomial_window_count((0,), 5, 58, 0, 59, max_q=100000)
    tests.append((pdata["coefficient_count"] == pdata["direct_count"] == 16, f"polynomial coefficient law count={pdata["coefficient_count"]}, direct={pdata["direct_count"]}"))

    # Known-theory validation checks.
    density, ratio = unary_mertens_ratio(30)
    tests.append((0 < density < 1 and ratio > 0, f"Mertens calibration density={density}, ratio={ratio}"))
    tests.append((all(ok for *_rest, ok in affine_root_count_table((0, 2), 7, 58)), "affine root-count calibration H=(0,2), z=7, n=58"))
    gprod, grows = goldbach_singular_trunc(60, 7)
    tests.append((gprod > 0 and len(grows) == len(primes_up_to(7)), f"Goldbach local calibration product={gprod}"))
    poly_rows = polynomial_validation_table("x*x+1", 7)
    tests.append((len(poly_rows) == len(primes_up_to(7)), f"polynomial validation rows={poly_rows}"))

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
            self.title("MSI Survivor Lab v09 Integrated Python/C++ - Local, Global, Windowed, Polynomial")
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
            self.tab_validation = ttk.Frame(self.nb)
            self.tab_pattern = ttk.Frame(self.nb)
            self.tab_layered = ttk.Frame(self.nb)
            self.tab_boundary = ttk.Frame(self.nb)
            self.tab_multilayer = ttk.Frame(self.nb)
            self.tab_polynomial = ttk.Frame(self.nb)
            self.tab_backend = ttk.Frame(self.nb)
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
                (self.tab_validation, "Known Theory Validation"),
                (self.tab_pattern, "Pattern Search"),
                (self.tab_layered, "Layered Exhaustion"),
                (self.tab_boundary, "Boundary Completion"),
                (self.tab_multilayer, "Multi-Layer"),
                (self.tab_polynomial, "Polynomial Law"),
                (self.tab_backend, "C++ Backend"),
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
            self._build_validation()
            self._build_pattern()
            self._build_layered()
            self._build_boundary()
            self._build_multilayer()
            self._build_polynomial()
            self._build_backend()
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
            self.diagram_toolbar = NavigationToolbar2Tk(self.diagram_canvas, self.tab_diagrams) if NavigationToolbar2Tk is not None else None
            if self.diagram_toolbar is not None:
                self.diagram_toolbar.update()
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

        def _build_validation(self) -> None:
            controls = ttk.Frame(self.tab_validation, padding=6)
            controls.pack(fill=tk.X)
            self.val_n_var = tk.IntVar(value=58)
            self.val_N_var = tk.IntVar(value=60)
            self.val_a_var = tk.IntVar(value=0)
            self.val_b_var = tk.IntVar(value=59)
            self.val_poly_var = tk.StringVar(value="x*x+1")
            ttk.Label(controls, text="n=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.val_n_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="N=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.val_N_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="I=[").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.val_a_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text=",").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.val_b_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="]  f(x)=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.val_poly_var, width=20).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Run validation", command=self.refresh_validation).pack(side=tk.LEFT, padx=5)
            self.validation_text = self._text(self.tab_validation)


        def _build_pattern(self) -> None:
            controls = ttk.Frame(self.tab_pattern, padding=6)
            controls.pack(fill=tk.X)
            self.pattern_Nmin_var = tk.IntVar(value=4)
            self.pattern_Nmax_var = tk.IntVar(value=500)
            self.pattern_mod_var = tk.IntVar(value=6)
            self.pattern_parity_var = tk.IntVar(value=0)
            ttk.Label(controls, text="N min=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.pattern_Nmin_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="N max=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.pattern_Nmax_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="mod=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.pattern_mod_var, width=6).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="parity=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.pattern_parity_var, width=5).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Run pattern scan", command=self.refresh_pattern).pack(side=tk.LEFT, padx=5)
            ttk.Button(controls, text="Export CSV", command=self.export_pattern_csv).pack(side=tk.LEFT, padx=5)
            paned = ttk.PanedWindow(self.tab_pattern, orient=tk.VERTICAL)
            paned.pack(fill=tk.BOTH, expand=True)
            top = ttk.Frame(paned)
            bottom = ttk.Frame(paned)
            paned.add(top, weight=1)
            paned.add(bottom, weight=3)
            self.pattern_text = self._text(top)
            cols = ("N", "z", "B", "window", "count", "first_i", "first_packet", "residue")
            self.pattern_tree = ttk.Treeview(bottom, columns=cols, show="headings")
            for c in cols:
                self.pattern_tree.heading(c, text=c)
                self.pattern_tree.column(c, width=130, anchor=tk.CENTER)
            self.pattern_tree.pack(fill=tk.BOTH, expand=True)
            self._last_pattern_rows = []

        def refresh_pattern(self) -> None:
            H, _z = self.get_params()
            Nmin = int(self.pattern_Nmin_var.get())
            Nmax = int(self.pattern_Nmax_var.get())
            modulus = int(self.pattern_mod_var.get())
            parity = int(self.pattern_parity_var.get()) % 2
            rows = scan_complete_survivors(H, Nmin, Nmax, parity=parity, modulus=modulus)
            self._last_pattern_rows = rows
            self.pattern_text.configure(state=tk.NORMAL)
            self.pattern_text.delete("1.0", tk.END)
            self.pattern_text.insert(tk.END, f"H={list(H)}, N range=[{Nmin},{Nmax}], parity={parity}, residue modulus={modulus}\n\n")
            self.pattern_text.insert(tk.END, pattern_summary(rows, modulus=modulus))
            self.pattern_text.configure(state=tk.DISABLED)
            self.clear_tree(self.pattern_tree)
            for r in rows[:1000]:
                self.pattern_tree.insert("", tk.END, values=(r.N, r.z, r.B, f"[{r.start_i},{r.stop_i}]", r.count, r.first_i, str(r.first_packet), r.residue))

        def export_pattern_csv(self) -> None:
            rows = getattr(self, '_last_pattern_rows', [])
            if not rows:
                self.refresh_pattern()
                rows = getattr(self, '_last_pattern_rows', [])
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            write_pattern_csv(rows, path)
            messagebox.showinfo("Exported", f"Saved {path}")

        def _build_layered(self) -> None:
            controls = ttk.Frame(self.tab_layered, padding=6)
            controls.pack(fill=tk.X)
            self.layered_n_var = tk.IntVar(value=58)
            self.layered_start_var = tk.IntVar(value=0)
            self.layered_stop_var = tk.IntVar(value=59)
            self.layered_y_var = tk.StringVar(value="auto")
            ttk.Label(controls, text="n=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.layered_n_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="I=[").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.layered_start_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text=",").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.layered_stop_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="]  y=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.layered_y_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Compute layered certificate", command=self.refresh_layered).pack(side=tk.LEFT, padx=5)
            self.layered_text = self._text(self.tab_layered)

        def refresh_layered(self) -> None:
            H, z = self.get_params()
            n = int(self.layered_n_var.get())
            start = int(self.layered_start_var.get())
            stop = int(self.layered_stop_var.get())
            y_text = self.layered_y_var.get().strip().lower()
            y = None if y_text in ("", "auto") else int(y_text)
            result = layered_exhaustion_summary(H, z, n, start, stop, y=y)
            self.layered_text.configure(state=tk.NORMAL)
            self.layered_text.delete("1.0", tk.END)
            self.layered_text.insert(tk.END, layered_report(result))
            self.layered_text.configure(state=tk.DISABLED)

        def _build_boundary(self) -> None:
            controls = ttk.Frame(self.tab_boundary, padding=6)
            controls.pack(fill=tk.X)
            self.boundary_n_var = tk.IntVar(value=58)
            self.boundary_start_var = tk.IntVar(value=0)
            self.boundary_stop_var = tk.IntVar(value=59)
            ttk.Label(controls, text="n").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.boundary_n_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="start i").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.boundary_start_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="stop i").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.boundary_stop_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Compute boundary completion", command=self.refresh_boundary).pack(side=tk.LEFT, padx=5)
            self.boundary_text = self._text(self.tab_boundary)

        def refresh_boundary(self) -> None:
            H, z = self.get_params()
            n = int(self.boundary_n_var.get())
            start = int(self.boundary_start_var.get())
            stop = int(self.boundary_stop_var.get())
            report = boundary_completion_report(H, z, n, start, stop)
            self.boundary_text.configure(state=tk.NORMAL)
            self.boundary_text.delete("1.0", tk.END)
            self.boundary_text.insert(tk.END, report)
            self.boundary_text.configure(state=tk.DISABLED)

        def _build_multilayer(self) -> None:
            controls = ttk.Frame(self.tab_multilayer, padding=6)
            controls.pack(fill=tk.X)
            self.multi_n_var = tk.IntVar(value=58)
            self.multi_start_var = tk.IntVar(value=0)
            self.multi_stop_var = tk.IntVar(value=59)
            self.multi_levels_var = tk.StringVar(value="auto")
            ttk.Label(controls, text="n").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.multi_n_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="start i").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.multi_start_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="stop i").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.multi_stop_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="levels").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.multi_levels_var, width=16).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Compute multi-layer", command=self.refresh_multilayer).pack(side=tk.LEFT, padx=5)
            self.multilayer_text = self._text(self.tab_multilayer)

        def refresh_multilayer(self) -> None:
            H, z = self.get_params()
            n = int(self.multi_n_var.get())
            start = int(self.multi_start_var.get())
            stop = int(self.multi_stop_var.get())
            levels = parse_levels_text(self.multi_levels_var.get(), z)
            result = multi_layer_attrition_summary(H, z, n, start, stop, levels=levels)
            self.multilayer_text.configure(state=tk.NORMAL)
            self.multilayer_text.delete("1.0", tk.END)
            self.multilayer_text.insert(tk.END, multi_layer_report(result))
            self.multilayer_text.configure(state=tk.DISABLED)

        def _build_polynomial(self) -> None:
            controls = ttk.Frame(self.tab_polynomial, padding=6)
            controls.pack(fill=tk.X)
            self.poly_n_var = tk.IntVar(value=58)
            self.poly_a_var = tk.IntVar(value=0)
            self.poly_b_var = tk.IntVar(value=59)
            self.poly_maxq_var = tk.IntVar(value=300000)
            ttk.Label(controls, text="n=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.poly_n_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="I=[").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.poly_a_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text=",").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.poly_b_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="] max_Q=").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.poly_maxq_var, width=10).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Verify coefficient law", command=self.refresh_polynomial).pack(side=tk.LEFT, padx=5)
            self.polynomial_text = self._text(self.tab_polynomial)


        def _build_backend(self) -> None:
            controls = ttk.Frame(self.tab_backend, padding=6)
            controls.pack(fill=tk.X)
            self.cpp_path_var = tk.StringVar(value=default_cpp_executable())
            self.cpp_mode_var = tk.StringVar(value="count")
            self.cpp_n_var = tk.IntVar(value=58)
            self.cpp_start_var = tk.IntVar(value=0)
            self.cpp_stop_var = tk.IntVar(value=59)
            self.cpp_y_var = tk.IntVar(value=3)
            self.cpp_maxq_var = tk.IntVar(value=300000)
            ttk.Label(controls, text="C++ exe:").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.cpp_path_var, width=42).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Browse", command=self.browse_cpp_backend).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="mode:").pack(side=tk.LEFT)
            combo = ttk.Combobox(controls, textvariable=self.cpp_mode_var, width=12, state="readonly", values=("count", "layered", "boundary", "poly-law", "benchmark"))
            combo.pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="n").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.cpp_n_var, width=7).pack(side=tk.LEFT, padx=2)
            ttk.Label(controls, text="I=[").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.cpp_start_var, width=7).pack(side=tk.LEFT, padx=2)
            ttk.Label(controls, text=",").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.cpp_stop_var, width=7).pack(side=tk.LEFT, padx=2)
            ttk.Label(controls, text="] y").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.cpp_y_var, width=6).pack(side=tk.LEFT, padx=2)
            ttk.Label(controls, text="maxQ").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.cpp_maxq_var, width=9).pack(side=tk.LEFT, padx=2)
            ttk.Button(controls, text="Run C++", command=self.refresh_backend).pack(side=tk.LEFT, padx=5)
            self.backend_text = self._text(self.tab_backend)
            self.backend_text.insert(tk.END, "This tab calls the optional C++ v09 backend with --json and displays the result.\nPython remains the GUI/diagram layer; C++ is the fast scan engine.\n")

        def browse_cpp_backend(self) -> None:
            path = filedialog.askopenfilename(title="Select C++ backend executable")
            if path:
                self.cpp_path_var.set(path)

        def refresh_backend(self) -> None:
            try:
                H, z = self.get_params()
                exe = self.cpp_path_var.get()
                mode = self.cpp_mode_var.get()
                n = int(self.cpp_n_var.get())
                start = int(self.cpp_start_var.get())
                stop = int(self.cpp_stop_var.get())
                if mode == "count":
                    report = backend_count_report(exe, H, z, n, start, stop)
                elif mode == "layered":
                    report = backend_layered_report(exe, H, z, int(self.cpp_y_var.get()), n, start, stop)
                elif mode == "boundary":
                    report = backend_boundary_report(exe, H, z, n, start, stop)
                elif mode == "poly-law":
                    report = backend_polynomial_report(exe, H, z, n, start, stop, int(self.cpp_maxq_var.get()))
                elif mode == "benchmark":
                    report = backend_benchmark(exe, H, z, n, start, stop)
                else:
                    report = "Unknown backend mode"
                self.backend_text.configure(state=tk.NORMAL)
                self.backend_text.delete("1.0", tk.END)
                self.backend_text.insert(tk.END, report)
                self.backend_text.configure(state=tk.DISABLED)
            except Exception as exc:
                self.backend_text.configure(state=tk.NORMAL)
                self.backend_text.delete("1.0", tk.END)
                self.backend_text.insert(tk.END, f"C++ backend error:\n{exc}\n\nCheck that the executable path is correct and that the C++ source was built with -std=c++17 -O2.")
                self.backend_text.configure(state=tk.DISABLED)

        def _build_roadmap(self) -> None:
            self.roadmap_text = self._text(self.tab_roadmap)
            self.roadmap_text.insert(tk.END, """MSI roadmap\n\n1. Local envelope: local laws mu_p for Y_p=p-c_p.\n2. Global envelope: multiplicative convolution over p<=z.\n3. Dynamic n-intervals: exact over complete Q(z)-periods.\n4. Windowed envelope: localized survivor count in actual i-windows.\n5. Prime interior: z>=sqrt(B) certifies large-prime coordinates >z.\n6. Boundary channels: small prime coordinates <=z must be handled separately.\n7. Polynomial law: localized coefficient pairing remembers survivor positions.
8. C++ backend: optional fast engine called by Python via JSON/CSV for large scans.\n""")
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
                self.refresh_validation()
                self.refresh_pattern()
                self.refresh_layered()
                self.refresh_boundary()
                self.refresh_multilayer()
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

        def refresh_validation(self) -> None:
            H, z = self.get_params()
            n = int(self.val_n_var.get())
            N = int(self.val_N_var.get())
            start = int(self.val_a_var.get())
            stop = int(self.val_b_var.get())
            poly = self.val_poly_var.get().strip() or "x*x+1"
            report = validation_report(H, z, n=n, start=start, stop=stop, N=N, poly=poly)
            self.validation_text.configure(state=tk.NORMAL)
            self.validation_text.delete("1.0", tk.END)
            self.validation_text.insert(tk.END, report)
            self.validation_text.configure(state=tk.DISABLED)

        def refresh_polynomial(self) -> None:
            H, z = self.get_params()
            n = int(self.poly_n_var.get())
            start = int(self.poly_a_var.get())
            stop = int(self.poly_b_var.get())
            max_q = int(self.poly_maxq_var.get())
            report = polynomial_law_report(H, z, n, start, stop, max_q=max_q)
            self.polynomial_text.configure(state=tk.NORMAL)
            self.polynomial_text.delete("1.0", tk.END)
            self.polynomial_text.insert(tk.END, report)
            self.polynomial_text.configure(state=tk.DISABLED)

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
    parser = argparse.ArgumentParser(description="MSI Survivor Lab v09 Integrated Python/C++ Polynomial Laboratory")
    parser.add_argument("--self-test", action="store_true", help="run headless verification tests")
    parser.add_argument("--envelope", action="store_true", help="print exact global envelope and exit")
    parser.add_argument("--validate", action="store_true", help="print known-theory validation report and exit")
    parser.add_argument("--scan-goldbach", action="store_true", help="scan localized complete survivors for even Goldbach sums and exit")
    parser.add_argument("--pattern-scan", action="store_true", help="scan localized complete-survivor nonemptiness for the selected H and exit")
    parser.add_argument("--layered", action="store_true", help="print layered interval-exhaustion certificate and exit")
    parser.add_argument("--boundary-complete", action="store_true", help="print small-prime boundary completion report and exit")
    parser.add_argument("--multi-layer", action="store_true", help="print multi-layer residual attrition report and exit")
    parser.add_argument("--poly-law", action="store_true", help="print localized survivor-polynomial coefficient-law report and exit")
    parser.add_argument("--cpp", action="store_true", help="run the optional C++ backend instead of Python for selected modes")
    parser.add_argument("--cpp-exe", default=default_cpp_executable(), help="path to C++ backend executable")
    parser.add_argument("--cpp-mode", default="count", choices=["count", "layered", "boundary", "poly-law", "benchmark"], help="C++ backend mode")
    parser.add_argument("--max-q", type=int, default=300000, help="maximum Q for sparse survivor-polynomial support construction")
    parser.add_argument("--y", default="auto", help="base level y for --layered, or auto")
    parser.add_argument("--levels", default="auto", help="comma-separated residual layer levels for --multi-layer, or auto")
    parser.add_argument("--csv-out", default="", help="optional CSV output path for --pattern-scan")
    parser.add_argument("--mod", type=int, default=6, help="residue modulus for pattern summary")
    parser.add_argument("--N-min", type=int, default=4, help="minimum N for --pattern-scan")
    parser.add_argument("--H", default="0,2", help="comma-separated shift set, e.g. 0,2,6")
    parser.add_argument("--z", type=int, default=5, help="sieve level")
    parser.add_argument("--n", type=int, default=58, help="diagonal parameter for validation")
    parser.add_argument("--N", type=int, default=60, help="additive/Goldbach sum for validation or scan upper bound")
    parser.add_argument("--start", type=int, default=0, help="window start i")
    parser.add_argument("--stop", type=int, default=59, help="window stop i")
    parser.add_argument("--poly", default="x*x+1", help="polynomial expression in x for validation")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test(verbose=True) else 1
    if args.cpp:
        H = parse_H(args.H)
        if args.cpp_mode == "count":
            print(backend_count_report(args.cpp_exe, H, args.z, args.n, args.start, args.stop))
        elif args.cpp_mode == "layered":
            y_arg = recommended_base_level(args.stop - args.start + 1, args.z) if str(args.y).lower() in ("", "auto") else int(args.y)
            print(backend_layered_report(args.cpp_exe, H, args.z, y_arg, args.n, args.start, args.stop))
        elif args.cpp_mode == "boundary":
            print(backend_boundary_report(args.cpp_exe, H, args.z, args.n, args.start, args.stop))
        elif args.cpp_mode == "poly-law":
            print(backend_polynomial_report(args.cpp_exe, H, args.z, args.n, args.start, args.stop, args.max_q))
        elif args.cpp_mode == "benchmark":
            print(backend_benchmark(args.cpp_exe, H, args.z, args.n, args.start, args.stop))
        return 0
    if args.validate:
        H = parse_H(args.H)
        print(validation_report(H, args.z, n=args.n, start=args.start, stop=args.stop, N=args.N, poly=args.poly))
        return 0
    if args.layered:
        H = parse_H(args.H)
        y_arg = None if str(args.y).lower() in ("", "auto") else int(args.y)
        result = layered_exhaustion_summary(H, args.z, args.n, args.start, args.stop, y=y_arg)
        print(layered_report(result))
        return 0
    if args.boundary_complete:
        H = parse_H(args.H)
        print(boundary_completion_report(H, args.z, args.n, args.start, args.stop))
        return 0
    if args.multi_layer:
        H = parse_H(args.H)
        levels = parse_levels_text(args.levels, args.z)
        result = multi_layer_attrition_summary(H, args.z, args.n, args.start, args.stop, levels=levels)
        print(multi_layer_report(result))
        return 0
    if args.poly_law:
        H = parse_H(args.H)
        print(polynomial_law_report(H, args.z, args.n, args.start, args.stop, max_q=args.max_q))
        return 0
    if args.pattern_scan:
        H = parse_H(args.H)
        rows = scan_complete_survivors(H, args.N_min, args.N, parity=0, modulus=args.mod)
        print(pattern_summary(rows, modulus=args.mod))
        if args.csv_out:
            write_pattern_csv(rows, args.csv_out)
            print(f"CSV saved to {args.csv_out}")
        return 0
    if args.scan_goldbach:
        rows = goldbach_complete_survivor_scan(4, args.N)
        print("N	z	complete_survivors	first_i	first_pair")
        for N, z, cnt, first_i, pair in rows:
            print(f"{N}	{z}	{cnt}	{first_i}	{pair}")
        return 0
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
