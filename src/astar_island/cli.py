"""CLI for local runs (requires AINM_TOKEN)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from dotenv import load_dotenv

from astar_island import config
from astar_island.api_client import AstarClient
from astar_island.runner import run_active_round


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Astar Island — NM i AI 2026")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # run-round  — adaptive simulate + predict + submit
    # ------------------------------------------------------------------
    pr = sub.add_parser("run-round", help="Adaptive simulate + predict + submit active round")
    pr.add_argument("--no-submit", action="store_true", help="Only simulate and build predictions")
    pr.add_argument("--no-log", action="store_true", help="Do not write data/observations/run_*.json")
    pr.add_argument("--max-queries", type=int, default=None, help="Override query budget cap")
    pr.add_argument("--round-id", type=str, default=None, help="Fixed round UUID (optional)")

    # ------------------------------------------------------------------
    # budget / active  — informational
    # ------------------------------------------------------------------
    sub.add_parser("budget", help="Print query budget for active round")
    sub.add_parser("active", help="Print active round summary as JSON")
    sub.add_parser("my-rounds", help="List all team rounds with scores")

    # ------------------------------------------------------------------
    # refit  — train regime priors from accumulated GT data
    # ------------------------------------------------------------------
    refit_p = sub.add_parser(
        "refit",
        help="Refit regime priors from data/rounds/*.json (run backfill first)",
    )
    refit_p.add_argument(
        "--no-chaotic",
        action="store_true",
        help="Exclude chaotic rounds (high spatial_decay) from training",
    )

    # ------------------------------------------------------------------
    # eval  — LOO cross-validation
    # ------------------------------------------------------------------
    eval_p = sub.add_parser(
        "eval",
        help="LOO cross-validation over completed rounds in data/rounds/",
    )
    eval_p.add_argument(
        "--rounds-dir",
        type=str,
        default=None,
        help="Override path to rounds directory",
    )

    args = p.parse_args(argv)

    # ------------------------------------------------------------------
    # Commands that need a token
    # ------------------------------------------------------------------
    if args.cmd in ("run-round", "budget", "active", "my-rounds"):
        try:
            config.require_token()
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        client = AstarClient()

    # ------------------------------------------------------------------
    # run-round
    # ------------------------------------------------------------------
    if args.cmd == "run-round":
        print("[astar-island] run-round starting…", file=sys.stderr, flush=True)
        try:
            out = run_active_round(
                client,
                submit=not args.no_submit,
                max_queries=args.max_queries,
                round_id=args.round_id,
                write_log=not args.no_log,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[astar-island] failed: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return 1
        print(
            json.dumps({k: v for k, v in out.items() if k != "predictions"}, indent=2, default=str),
            flush=True,
        )
        return 0

    # ------------------------------------------------------------------
    # budget
    # ------------------------------------------------------------------
    if args.cmd == "budget":
        b = client.get_budget()
        print(b.model_dump_json(indent=2))
        return 0

    # ------------------------------------------------------------------
    # active
    # ------------------------------------------------------------------
    if args.cmd == "active":
        a = client.get_active_round()
        print(json.dumps(a, indent=2, default=str))
        return 0

    # ------------------------------------------------------------------
    # my-rounds
    # ------------------------------------------------------------------
    if args.cmd == "my-rounds":
        rounds = client.my_rounds()
        print(json.dumps(rounds, indent=2, default=str))
        return 0

    # ------------------------------------------------------------------
    # refit
    # ------------------------------------------------------------------
    if args.cmd == "refit":
        try:
            from pathlib import Path
            import importlib.util, sys as _sys
            scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
            spec = importlib.util.spec_from_file_location(
                "refit_regime_priors", scripts_dir / "refit_regime_priors.py"
            )
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            exclude_chaotic = getattr(args, "no_chaotic", False)
            mod.main(exclude_chaotic=exclude_chaotic)
        except Exception as e:  # noqa: BLE001
            print(f"[astar-island] refit failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 1
        return 0

    # ------------------------------------------------------------------
    # eval
    # ------------------------------------------------------------------
    if args.cmd == "eval":
        try:
            from pathlib import Path
            import importlib.util
            scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
            spec = importlib.util.spec_from_file_location(
                "evaluate_loo", scripts_dir / "evaluate_loo.py"
            )
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            rounds_dir = getattr(args, "rounds_dir", None)
            mod.main(rounds_dir=rounds_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[astar-island] eval failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 1
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
