"""
Visibility layer -- watch the system think, live, in the terminal.

    python watch.py data/scenario_03                 # 30s per simulated day
    python watch.py data/scenario_03 --speed 2       # a ~2.5 minute demo
    python watch.py data/scenario_03 --speed 0       # instant

The problem statement says the visibility layer must be real, not decorative, and
is not a priority. So this is a terminal view over the SAME pipeline the graded
run uses -- not a separate code path that reimplements the logic for show. Every
line printed is read from the actual decision that was actually made and written
to the actual output file.

What it shows that a log file does not: the shape of the narrative over time. You
can watch scenario_03 sit quiet through February, tick over to medium the day the
cancellation click arrives, and escalate three days before the deadline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from c360.pipeline import Pipeline  # noqa: E402
from c360.replay import ClockTick, EventTick  # noqa: E402
from c360.schema import Action, ConfidenceBand  # noqa: E402

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    RICH = True
except ImportError:  # rich is a nice-to-have, not a dependency of correctness
    RICH = False

BAND_COLOUR = {"low": "dim", "medium": "yellow", "high": "bold red"}
ACTION_COLOUR = {
    "no_action": "dim",
    "support_intervention": "cyan",
    "personalized_offer": "green",
    "proactive_retention_outreach": "yellow",
    "relationship_manager_escalation": "bold red",
    "compliance_fraud_hold": "bold magenta",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a scenario run.")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--speed", type=float, default=None,
                        help="wall-clock seconds per simulated day; 0 = instant")
    parser.add_argument("--live", action="store_true", help="use the LLMs in .env")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    console = Console() if RICH else None

    def out(text, style=None):
        if console:
            console.print(text, style=style)
        else:
            print(text if isinstance(text, str) else str(text))

    pipeline = Pipeline(
        args.scenario,
        speed=args.speed if args.speed is not None else pipelines_default(args.scenario),
        offline=not args.live,
        trace_path=args.out / f"{args.scenario.name}.trace.jsonl",
    )
    entities = pipeline.entities
    profile = entities.get("profile", {})

    out(f"\n[bold]{pipeline.engine.config.scenario_id}[/bold]  "
        f"{entities.get('customer_id')}  "
        f"{profile.get('customer_value_tier','?')} tier, {profile.get('tenure_months','?')} months"
        if RICH else
        f"\n{pipeline.engine.config.scenario_id}  {entities.get('customer_id')}")
    out(f"{pipeline.engine.config.simulated_start:%d %b %Y} -> "
        f"{pipeline.engine.config.simulated_end:%d %b %Y}   "
        f"embedder={pipeline.semantic.embedder_name}   "
        f"{'LIVE LLM' if args.live else 'offline (deterministic)'}")
    out("-" * 96)

    # Drive the pipeline's own internals so the display and the graded output can
    # never diverge -- there is exactly one implementation of the decision logic.
    pipeline.memory.record_events(pipeline.engine.history)
    previous = None
    for tick in pipeline.engine.stream():
        if isinstance(tick, EventTick):
            pipeline._on_event(tick)
            event = tick.event
            if event.is_late_arrival:
                out(f"  {tick.as_of:%d %b}  LATE ARRIVAL {event.event_id} "
                    f"(occurred {event.event_time:%d %b}, "
                    f"{event.arrival_lag_seconds / 3600:.0f}h ago)",
                    style="magenta" if RICH else None)
        elif isinstance(tick, ClockTick):
            pipeline._on_checkpoint(tick)
            row = pipeline.writer.checkpoints[-1]
            # Print only when the ASSESSMENT CHANGES. A checkpoint is emitted
            # every day and most of them repeat the previous one; printing all 74
            # would bury the three moments that matter in a wall of identical
            # rows. The output file still contains every day.
            signature = (row.inferred_state, row.confidence_band, row.action)
            if signature == previous:
                continue
            previous = signature

            band = row.confidence_band.value
            findings = pipeline.board.findings(tick.as_of)
            signals = sorted({f"{f.agent}/{f.signal}" for f in findings})

            if RICH:
                line = Text()
                line.append(f"  {tick.as_of:%d %b}  ")
                line.append(f"{row.inferred_state.value:<28}", style="bold")
                line.append(f"{band:<8}", style=BAND_COLOUR.get(band, ""))
                line.append(f"{row.action.value:<32}",
                            style=ACTION_COLOUR.get(row.action.value, ""))
                line.append(row.hitl_status.value, style="dim")
                console.print(line)
                if row.action is not Action.NO_ACTION:
                    console.print(f"        subtype  {row.action_subtype}", style="dim")
                    console.print(f"        evidence {', '.join(row.citations[:6])}", style="dim")
                    console.print(f"        signals  {', '.join(signals[:6])}", style="dim")
            else:
                print(f"  {tick.as_of:%d %b}  {row.inferred_state.value:<28}{band:<8}"
                      f"{row.action.value:<32}{row.hitl_status.value}")

    pipeline._collect_llm_stats()
    path = pipeline.writer.write(args.out / f"{args.scenario.name}_inferred_events.json")
    pipeline.close()

    out("-" * 96)
    out(pipeline.stats.summary())
    out(f"wrote {path}")

    # Score it, so the demo ends on a number rather than a vibe.
    truth = args.scenario / "ground_truth.json"
    if truth.exists():
        from c360.scoring import score_scenario

        score = score_scenario(args.scenario, path)
        out("")
        out(score.report())
    return 0


def pipelines_default(scenario: Path) -> float:
    """Use the pacing the scenario config asks for unless told otherwise."""
    import json

    config = json.loads((scenario / "replay_config.json").read_text(encoding="utf-8"))
    return float(config.get("replay_speed_seconds_per_simulated_day", 0))


if __name__ == "__main__":
    raise SystemExit(main())
