"""Command-line interface for MSI Survivor Lab."""
from __future__ import annotations

import argparse
import json
from .core import (
    parse_H, windowed_survivor_count, layered_exhaustion_summary,
    polynomial_window_count, validation_report, self_test, pattern_summary,
    scan_complete_survivors, write_pattern_csv,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MSI Survivor Lab command line tools")
    p.add_argument("--self-test", action="store_true", help="run internal tests")
    p.add_argument("--count", action="store_true", help="windowed survivor count")
    p.add_argument("--layered", action="store_true", help="layered exhaustion report")
    p.add_argument("--poly-law", action="store_true", help="survivor-polynomial coefficient law")
    p.add_argument("--validate", action="store_true", help="known-theory validation report")
    p.add_argument("--pattern-scan", action="store_true", help="scan complete survivor counts over N")
    p.add_argument("--H", default="0", help="comma-separated shift set, e.g. 0,2,6")
    p.add_argument("--z", type=int, default=5, help="sieve level")
    p.add_argument("--y", type=int, default=None, help="base layer for layered exhaustion")
    p.add_argument("--n", type=int, default=58, help="diagonal parameter n; Goldbach sum is N=n+2")
    p.add_argument("--start", type=int, default=0, help="window start")
    p.add_argument("--stop", type=int, default=59, help="window stop inclusive")
    p.add_argument("--N", type=int, default=500, help="maximum N for pattern scans")
    p.add_argument("--N-min", type=int, default=4, help="minimum N for pattern scans")
    p.add_argument("--mod", type=int, default=6, help="modulus for pattern summary")
    p.add_argument("--csv-out", default="", help="optional CSV output path for pattern scans")
    p.add_argument("--json", action="store_true", help="emit JSON for count/poly-law where supported")
    args = p.parse_args(argv)

    H = parse_H(args.H)
    if args.self_test:
        return 0 if self_test(verbose=True) else 1
    if args.count:
        result = windowed_survivor_count(H, args.z, args.n, args.start, args.stop)
        payload = {"mode":"count", "H":list(H), "z":args.z, "n":args.n, "start":args.start, "stop":args.stop, **result.__dict__}
        print(json.dumps(payload, indent=2) if args.json else payload)
        return 0
    if args.layered:
        y = args.y if args.y is not None else 0
        result = layered_exhaustion_summary(H, args.z, args.n, args.start, args.stop, y=y)
        print(layered_report(result))
        return 0
    if args.poly_law:
        result = polynomial_window_count(H, args.z, args.n, args.start, args.stop)
        print(json.dumps(result, indent=2, default=str) if args.json else result)
        return 0
    if args.validate:
        print(validation_report(H, args.z, n=args.n, start=args.start, stop=args.stop))
        return 0
    if args.pattern_scan:
        rows = scan_complete_survivors(H, args.N_min, args.N)
        print(pattern_summary(rows, modulus=args.mod))
        if args.csv_out:
            write_pattern_csv(rows, args.csv_out)
            print(f"CSV written to {args.csv_out}")
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
