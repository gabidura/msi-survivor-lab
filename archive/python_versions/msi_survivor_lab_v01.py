#!/usr/bin/env python3
"""
MSI Survivor Lab
================
A small tabbed GUI and command-line self-test tool for the Modular Survivor
Interface / primorial survivor framework.

The program verifies and explores:
- residue clouds A_p(H)
- overlap profiles rho_p(u;H)
- obstruction counts c_p(u,H)
- local survivor laws Y_p(u,H)=p-c_p(u,H)
- global convolutional survivor envelopes
- dynamic interval exhaustion
- coordinatewise large-prime-factor control and complete-sieve endpoint

Run GUI:
    python msi_survivor_lab.py

Run tests/headless:
    python msi_survivor_lab.py --self-test
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:  # headless self-test still works
    tk = None
    ttk = None


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------

def primes_up_to(z: int) -> List[int]:
    if z < 2:
        return []
    sieve = [True] * (z + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(z ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:z + 1:step] = [False] * (((z - start) // step) + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]


def is_prime(n: int) -> bool:
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
    n = abs(n)
    result: Dict[int, int] = {}
    if n < 2:
        return result
    d = 2
    while d * d <= n:
        while n % d == 0:
            result[d] = result.get(d, 0) + 1
            n //= d
        d = 3 if d == 2 else d + 2
    if n > 1:
        result[n] = result.get(n, 0) + 1
    return result


def omega(n: int) -> int:
    return sum(factorization(n).values())


def primorial(z: int) -> int:
    return math.prod(primes_up_to(z)) if z >= 2 else 1


def parse_H(text: str) -> List[int]:
    parts = [p.strip() for p in text.replace(';', ',').split(',') if p.strip()]
    if not parts:
        raise ValueError("H must contain at least one integer, e.g. 0,2,6")
    return [int(p) for p in parts]


def residue_cloud(H: Sequence[int], p: int) -> List[int]:
    return sorted({(-h - 1) % p for h in H})


def overlap_profile(H: Sequence[int], p: int) -> Dict[int, int]:
    A = set(residue_cloud(H, p))
    return {u: len(A.intersection({(a + u) % p for a in A})) for u in range(p)}


def local_profile(H: Sequence[int], p: int) -> List[Dict[str, int]]:
    A = residue_cloud(H, p)
    a = len(A)
    rho = overlap_profile(H, p)
    rows = []
    for u in range(p):
        c = 2 * a - rho[u]
        y = p - c
        rows.append({"p": p, "u": u, "a": a, "rho": rho[u], "c": c, "Y": y, "admissible": 1 if y > 0 else 0})
    return rows


def local_law(H: Sequence[int], p: int) -> Counter:
    return Counter(row["Y"] for row in local_profile(H, p))


def global_envelope(H: Sequence[int], z: int) -> Counter:
    dist = Counter({1: 1})
    for p in primes_up_to(z):
        law = local_law(H, p)
        new = Counter()
        for a, ca in dist.items():
            for b, cb in law.items():
                new[a * b] += ca * cb
        dist = new
    return Counter(dict(sorted(dist.items())))


def R_value(H: Sequence[int], z: int, n: int) -> int:
    prod = 1
    for p in primes_up_to(z):
        rows = local_profile(H, p)
        y_by_u = {row["u"]: row["Y"] for row in rows}
        prod *= y_by_u[(n + 2) % p]
    return prod


def moment_from_distribution(dist: Counter, s: int) -> int:
    return sum((value ** s) * count for value, count in dist.items())


def dynamic_interval_moment(H: Sequence[int], z: int, M: int, s: int) -> Tuple[int, int, int]:
    Q = primorial(z)
    dist = global_envelope(H, z)
    period_sum = moment_from_distribution(dist, s)
    formula = M * period_sum
    # direct only for moderate sizes
    N = M * Q
    direct = sum(R_value(H, z, n) ** s for n in range(1, N + 1)) if N <= 20000 else formula
    return formula, direct, period_sum


def packet_values(H: Sequence[int], n: int, i: int) -> List[int]:
    vals: List[int] = []
    for h in H:
        vals.append(i + h + 1)
        vals.append(n - i - h + 1)
    return vals


def survives_packet(H: Sequence[int], n: int, i: int, z: int) -> bool:
    vals = packet_values(H, n, i)
    for p in primes_up_to(z):
        if any(v % p == 0 for v in vals):
            return False
    return True


def prime_interior_status(vals: Sequence[int], z: int) -> str:
    if any(v <= 1 for v in vals):
        return "outside positive range"
    B = max(vals)
    if z >= math.isqrt(B) + (0 if math.isqrt(B) ** 2 == B else 1):
        if all(is_prime(v) and v > z for v in vals):
            return "large-prime interior packet"
        return "fails interior theorem assumptions or boundary"
    return "rough survivor / incomplete level"


def omega_bound(B: int, z: int) -> float:
    if z <= 1:
        return float('inf')
    return math.log(B) / math.log(z)


# ---------------------------------------------------------------------------
# Simple safe expression evaluator for optional future use
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
# Headless self-test
# ---------------------------------------------------------------------------

def self_test(verbose: bool = True) -> bool:
    tests = []
    examples = [
        ([0], 5, {0: 15, 3: 8, 4: 2, 6: 4, 8: 1}),
        ([0, 2], 5, {0: 25, 1: 2, 2: 2, 3: 1}),
        ([0, 2, 6], 7, {0: 189, 2: 12, 4: 8, 8: 1}),
    ]
    for H, z, expected in examples:
        dist = global_envelope(H, z)
        tests.append((dict(dist) == expected, f"global envelope H={H}, z={z}: {dict(dist)}"))
        Q = primorial(z)
        m1_direct = sum(R_value(H, z, n) for n in range(Q))
        m1_dist = moment_from_distribution(dist, 1)
        tests.append((m1_direct == m1_dist, f"moment 1 H={H}, z={z}: direct={m1_direct}, dist={m1_dist}"))
        m2_direct = sum(R_value(H, z, n) ** 2 for n in range(Q))
        m2_dist = moment_from_distribution(dist, 2)
        tests.append((m2_direct == m2_dist, f"moment 2 H={H}, z={z}: direct={m2_direct}, dist={m2_dist}"))

    H = [0]
    n = 58
    z = 8
    survivors = []
    composite_survivors = []
    for i in range(0, 59):
        if survives_packet(H, n, i, z):
            vals = packet_values(H, n, i)
            # The large-prime interior theorem assumes every coordinate is > 1.
            if any(v <= 1 for v in vals):
                continue
            survivors.append((i, vals))
            if not all(is_prime(v) and v > z for v in vals):
                composite_survivors.append((i, vals))
    tests.append((not composite_survivors, f"complete-sieve endpoint z=8, n=58, composites={composite_survivors}"))
    tests.append((survivors != [], f"complete-sieve endpoint survivors={survivors[:5]}"))

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
            self.title("MSI Survivor Lab - Primorial Survivor Framework")
            self.geometry("1120x760")
            self.H_var = tk.StringVar(value="0,2")
            self.z_var = tk.IntVar(value=5)
            self.status_var = tk.StringVar(value="Ready")
            self._build_ui()

        def _build_ui(self):
            top = ttk.Frame(self, padding=8)
            top.pack(fill=tk.X)
            ttk.Label(top, text="H =").pack(side=tk.LEFT)
            ttk.Entry(top, textvariable=self.H_var, width=24).pack(side=tk.LEFT, padx=4)
            ttk.Label(top, text="z =").pack(side=tk.LEFT)
            ttk.Entry(top, textvariable=self.z_var, width=8).pack(side=tk.LEFT, padx=4)
            ttk.Button(top, text="Refresh all", command=self.refresh_all).pack(side=tk.LEFT, padx=6)
            ttk.Button(top, text="Run self-test", command=self.run_self_test_gui).pack(side=tk.LEFT, padx=6)
            ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

            self.nb = ttk.Notebook(self)
            self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

            self.tab_overview = ttk.Frame(self.nb)
            self.tab_local = ttk.Frame(self.nb)
            self.tab_global = ttk.Frame(self.nb)
            self.tab_dynamic = ttk.Frame(self.nb)
            self.tab_prime = ttk.Frame(self.nb)
            self.tab_goldbach = ttk.Frame(self.nb)
            self.tab_roadmap = ttk.Frame(self.nb)
            for frame, title in [
                (self.tab_overview, "Overview"),
                (self.tab_local, "Local profiles"),
                (self.tab_global, "Global envelope"),
                (self.tab_dynamic, "Dynamic intervals"),
                (self.tab_prime, "Prime interior"),
                (self.tab_goldbach, "Additive/Goldbach"),
                (self.tab_roadmap, "Roadmap"),
            ]:
                self.nb.add(frame, text=title)

            self._build_overview()
            self._build_local()
            self._build_global()
            self._build_dynamic()
            self._build_prime()
            self._build_goldbach()
            self._build_roadmap()
            self.refresh_all()

        def get_params(self) -> Tuple[List[int], int]:
            H = parse_H(self.H_var.get())
            z = int(self.z_var.get())
            if z < 2:
                raise ValueError("z must be at least 2")
            return H, z

        def clear_tree(self, tree):
            for item in tree.get_children():
                tree.delete(item)

        def _text(self, parent):
            txt = tk.Text(parent, wrap=tk.WORD, font=("Consolas", 10))
            scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=txt.yview)
            txt.configure(yscrollcommand=scroll.set)
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            return txt

        def _build_overview(self):
            self.overview_text = self._text(self.tab_overview)

        def _build_local(self):
            cols = ("p", "u", "a", "rho", "c", "Y", "admissible")
            self.local_tree = ttk.Treeview(self.tab_local, columns=cols, show="headings")
            for c in cols:
                self.local_tree.heading(c, text=c)
                self.local_tree.column(c, width=90, anchor=tk.CENTER)
            self.local_tree.pack(fill=tk.BOTH, expand=True)

        def _build_global(self):
            frame = ttk.Frame(self.tab_global, padding=6)
            frame.pack(fill=tk.X)
            ttk.Button(frame, text="Export distribution CSV", command=self.export_distribution).pack(side=tk.LEFT)
            cols = ("R", "count", "probability", "cdf")
            self.global_tree = ttk.Treeview(self.tab_global, columns=cols, show="headings")
            for c in cols:
                self.global_tree.heading(c, text=c)
                self.global_tree.column(c, width=150, anchor=tk.CENTER)
            self.global_tree.pack(fill=tk.BOTH, expand=True)

        def _build_dynamic(self):
            controls = ttk.Frame(self.tab_dynamic, padding=6)
            controls.pack(fill=tk.X)
            self.M_var = tk.IntVar(value=3)
            self.s_var = tk.IntVar(value=2)
            ttk.Label(controls, text="M =").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.M_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="moment s =").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.s_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Check dynamic interval", command=self.refresh_dynamic).pack(side=tk.LEFT, padx=5)
            self.dynamic_text = self._text(self.tab_dynamic)

        def _build_prime(self):
            controls = ttk.Frame(self.tab_prime, padding=6)
            controls.pack(fill=tk.X)
            self.n_var = tk.IntVar(value=58)
            self.i_min_var = tk.IntVar(value=0)
            self.i_max_var = tk.IntVar(value=58)
            ttk.Label(controls, text="n =").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.n_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="i min =").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.i_min_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="i max =").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.i_max_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="List survivor packets", command=self.refresh_prime).pack(side=tk.LEFT, padx=5)
            cols = ("i", "packet", "B", "omega max", "status")
            self.prime_tree = ttk.Treeview(self.tab_prime, columns=cols, show="headings")
            for c in cols:
                self.prime_tree.heading(c, text=c)
                self.prime_tree.column(c, width=180, anchor=tk.CENTER)
            self.prime_tree.pack(fill=tk.BOTH, expand=True)

        def _build_goldbach(self):
            controls = ttk.Frame(self.tab_goldbach, padding=6)
            controls.pack(fill=tk.X)
            self.N_var = tk.IntVar(value=60)
            self.d_var = tk.IntVar(value=2)
            ttk.Label(controls, text="N =").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.N_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Label(controls, text="d =").pack(side=tk.LEFT)
            ttk.Entry(controls, textvariable=self.d_var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Compute local additive counts", command=self.refresh_goldbach).pack(side=tk.LEFT, padx=5)
            self.goldbach_text = self._text(self.tab_goldbach)

        def _build_roadmap(self):
            self.roadmap_text = self._text(self.tab_roadmap)
            self.roadmap_text.insert(tk.END, """MSI program roadmap\n\n1. Paper IV: global survivor-fiber distribution.\n2. Paper V: computational refinement and software.\n3. Paper VI: weighted Maynard/Selberg interface.\n\nKey principle:\n  local laws -> convolutional envelope -> dynamic intervals -> prime interior selection.\n\nBoundary:\n  complete sieving certifies large-prime interiors, but small prime coordinates <= z require boundary channels.\n""")
            self.roadmap_text.configure(state=tk.DISABLED)

        def refresh_all(self):
            try:
                self.refresh_overview()
                self.refresh_local()
                self.refresh_global()
                self.refresh_dynamic()
                self.refresh_prime()
                self.refresh_goldbach()
                self.status_var.set("Updated")
            except Exception as e:
                self.status_var.set("Error")
                messagebox.showerror("Error", str(e))

        def refresh_overview(self):
            H, z = self.get_params()
            ps = primes_up_to(z)
            Q = primorial(z)
            dist = global_envelope(H, z)
            mean = moment_from_distribution(dist, 1) / Q
            second = moment_from_distribution(dist, 2) / Q
            var = second - mean * mean
            positive = sum(c for r, c in dist.items() if r > 0)
            self.overview_text.configure(state=tk.NORMAL)
            self.overview_text.delete("1.0", tk.END)
            self.overview_text.insert(tk.END, f"H = {H}\nz = {z}\nprimes <= z = {ps}\nQ(z) = {Q}\n\n")
            self.overview_text.insert(tk.END, f"Envelope support size = {len(dist)} distinct fiber values\n")
            self.overview_text.insert(tk.END, f"Positive probability = {positive}/{Q} = {positive/Q:.6f}\n")
            self.overview_text.insert(tk.END, f"Mean R = {mean:.6f}\nVariance R = {var:.6f}\n\n")
            self.overview_text.insert(tk.END, "Interpretation:\n")
            self.overview_text.insert(tk.END, "- R=0 means at least one local obstruction collapses the diagonal.\n")
            self.overview_text.insert(tk.END, "- Positive R means the diagonal has survivor index classes at this sieve level.\n")
            self.overview_text.insert(tk.END, "- The distribution below is obtained by multiplicative convolution of local laws.\n")
            self.overview_text.configure(state=tk.DISABLED)

        def refresh_local(self):
            H, z = self.get_params()
            self.clear_tree(self.local_tree)
            for p in primes_up_to(z):
                for row in local_profile(H, p):
                    self.local_tree.insert("", tk.END, values=(row["p"], row["u"], row["a"], row["rho"], row["c"], row["Y"], row["admissible"]))

        def refresh_global(self):
            H, z = self.get_params()
            Q = primorial(z)
            dist = global_envelope(H, z)
            self.clear_tree(self.global_tree)
            cdf = 0
            for r, count in sorted(dist.items()):
                cdf += count
                self.global_tree.insert("", tk.END, values=(r, count, f"{count/Q:.8f}", f"{cdf/Q:.8f}"))

        def refresh_dynamic(self):
            H, z = self.get_params()
            M = int(self.M_var.get())
            s = int(self.s_var.get())
            Q = primorial(z)
            formula, direct, period_sum = dynamic_interval_moment(H, z, M, s)
            self.dynamic_text.configure(state=tk.NORMAL)
            self.dynamic_text.delete("1.0", tk.END)
            self.dynamic_text.insert(tk.END, f"H={H}, z={z}, Q={Q}, M={M}, N=M*Q={M*Q}, s={s}\n\n")
            self.dynamic_text.insert(tk.END, f"Period moment sum = {period_sum}\n")
            self.dynamic_text.insert(tk.END, f"Formula M * period = {formula}\n")
            if M * Q <= 20000:
                self.dynamic_text.insert(tk.END, f"Direct enumeration = {direct}\n")
                self.dynamic_text.insert(tk.END, f"Match = {formula == direct}\n")
            else:
                self.dynamic_text.insert(tk.END, "Direct enumeration skipped because N is large; formula is exact by theorem.\n")
            self.dynamic_text.configure(state=tk.DISABLED)

        def refresh_prime(self):
            H, z = self.get_params()
            n = int(self.n_var.get())
            i_min = int(self.i_min_var.get())
            i_max = int(self.i_max_var.get())
            self.clear_tree(self.prime_tree)
            for i in range(i_min, i_max + 1):
                if survives_packet(H, n, i, z):
                    vals = packet_values(H, n, i)
                    B = max(vals)
                    om = max(omega(v) for v in vals if v > 1) if all(v > 1 for v in vals) else "-"
                    status = prime_interior_status(vals, z)
                    self.prime_tree.insert("", tk.END, values=(i, str(vals), B, om, status))

        def refresh_goldbach(self):
            N = int(self.N_var.get())
            d = int(self.d_var.get())
            z = int(self.z_var.get())
            ps = primes_up_to(z)
            self.goldbach_text.configure(state=tk.NORMAL)
            self.goldbach_text.delete("1.0", tk.END)
            self.goldbach_text.insert(tk.END, f"Additive hyperplane x_1+...+x_d=N with nonzero coordinates mod p\nN={N}, d={d}, z={z}\n\n")
            global_count = 1
            for p in ps:
                if N % p == 0:
                    count = ((p - 1) ** d + (p - 1) * ((-1) ** d)) // p
                else:
                    count = ((p - 1) ** d - ((-1) ** d)) // p
                global_count *= count
                self.goldbach_text.insert(tk.END, f"p={p:3d}  local count={count}\n")
            self.goldbach_text.insert(tk.END, f"\nCRT product count over Q(z): {global_count}\n")
            self.goldbach_text.configure(state=tk.DISABLED)

        def export_distribution(self):
            H, z = self.get_params()
            dist = global_envelope(H, z)
            Q = primorial(z)
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["R", "count", "probability"])
                for r, c in sorted(dist.items()):
                    w.writerow([r, c, c / Q])
            messagebox.showinfo("Exported", f"Saved {path}")

        def run_self_test_gui(self):
            ok = self_test(verbose=False)
            messagebox.showinfo("Self-test", "PASS" if ok else "FAIL")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MSI Survivor Lab")
    parser.add_argument("--self-test", action="store_true", help="run headless verification tests")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test(verbose=True) else 1
    if tk is None:
        print("tkinter is not available; run with --self-test for headless mode", file=sys.stderr)
        return 2
    app = SurvivorLabApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

