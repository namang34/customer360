"""
Run every scenario end to end and score the output against ground truth.

    python run_evaluation.py                    # all three, offline (no API calls)
    python run_evaluation.py --live             # use the LLMs configured in .env
    python run_evaluation.py --scenario data/scenario_03 --speed 30

Offline is the DEFAULT on purpose. The deterministic path is what the test suite
exercises and what an examiner can reproduce without keys, quota or a network;
`--live` layers the language models on top and should produce the same or better
numbers, never different plumbing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from c360.pipeline import Pipeline  # noqa: E402
from c360.scoring import overall_report, score_scenario  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and score the Customer 360 system.")
    parser.add_argument("--scenario", type=Path, action="append", dest="scenarios")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--speed", type=float, default=0)
    parser.add_argument("--live", action="store_true", help="use the LLMs configured in .env")
    parser.add_argument("--trace", action="store_true", default=True)
    parser.add_argument("--hitl", action="store_true", help="prompt for human approval on each action")
    args = parser.parse_args()

    scenarios = args.scenarios or sorted(Path("data").glob("scenario_*"))
    scenarios = [s for s in scenarios if (s / "replay_config.json").exists()]
    if not scenarios:
        print("no scenarios found under data/")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    scores = []

    for scenario in scenarios:
        print(f"\n{'=' * 72}\nRUN {scenario.name}{'  [live LLM]' if args.live else '  [offline]'}\n{'=' * 72}")
        pipeline = Pipeline(
            scenario,
            speed=args.speed,
            offline=not args.live,
            trace_path=args.out / f"{scenario.name}.trace.jsonl" if args.trace else None,
            interactive_hitl=args.hitl,
        )
        writer, stats = pipeline.run()
        output_path = writer.write(args.out / f"{scenario.name}_inferred_events.json")
        pipeline.close()

        print(f"  {stats.summary()}")
        print(f"  wrote {output_path}")
        print()
        score = score_scenario(scenario, output_path)
        print(score.report())
        scores.append(score)

    print(overall_report(scores))

    all_checkpoints = [c for s in scores for c in s.checkpoints]
    all_fp = [f for s in scores for f in s.false_positives]
    clean = all(c.fully_correct for c in all_checkpoints) and all(f.passed for f in all_fp)
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
