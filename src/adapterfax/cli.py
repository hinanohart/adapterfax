"""Command-line interface: ``adapterfax audit`` and ``adapterfax gate``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _cmd_audit(args: argparse.Namespace) -> int:
    from .core import audit
    from .loader import load_adapters

    adapters = load_adapters(args.files)
    report = audit(
        adapters,
        tol=args.tol,
        aggregation=args.aggregation,
        estimator=args.estimator,
        tw_refine=args.tw_refine,
    )
    if args.json:
        print(report.to_json())
    else:
        print(
            f"effective_capacity_used: {report.effective_capacity_used} "
            f"(95% CI {report.effective_capacity_ci[0]:.1f}..{report.effective_capacity_ci[1]:.1f})"
        )
        print(f"active_mode: {report.active_mode}")
        print(f"redundant subsets (census): {len(report.dependency_census)}")
        for ci in report.dependency_census[:10]:
            print(f"  - {{{', '.join(ci.members)}}} (margin {ci.margin:.3f})")
        print(
            f"baselines: erank={report.baselines['erank']:.2f} "
            f"para_total={report.baselines['para_total']} "
            f"cosine_redundancy={report.baselines['cosine_redundancy']:.3f}"
        )
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from .gates import run_gates

    res = run_gates(fast=not args.full)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        for name, g in res["gates"].items():
            mark = "PASS" if g["passed"] else "FAIL"
            print(f"  [{mark}] {name}: {g['detail']}")
        print(f"active_mode={res['active_mode']} all_pass={res['all_pass']}")
    return 0 if res["all_pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adapterfax", description=__doc__)
    p.add_argument("--version", action="version", version=f"adapterfax {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="audit a stack of adapter .safetensors files")
    a.add_argument("files", nargs="+", help="adapter .safetensors files")
    a.add_argument("--tol", type=float, default=None, help="census energy-fraction tolerance")
    a.add_argument("--aggregation", choices=("layerwise",), default="layerwise")
    a.add_argument("--estimator", choices=("bulk_mean", "median_match"), default="bulk_mean")
    a.add_argument(
        "--tw-refine",
        dest="tw_refine",
        action="store_true",
        help="use scipy [tw] extra to sharpen the analytic TW cross-check",
    )
    a.add_argument("--json", action="store_true", help="emit the full JSON report")
    a.set_defaults(func=_cmd_audit)

    g = sub.add_parser("gate", help="run the synthetic sensitivity gates G1..G9")
    g.add_argument("--full", action="store_true", help="1000 null trials (default: fast)")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=_cmd_gate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
