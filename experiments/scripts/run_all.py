"""Batch experiment runner — runs all configs in a directory.

Usage:
    python experiments/scripts/run_all.py --config-dir experiments/configs/a100
    python experiments/scripts/run_all.py --config-dir experiments/configs --pattern "*llama8b*"
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from xrag.experiment.config import load_config
from xrag.experiment.runner import run_experiment, save_result


def find_configs(config_dir: str, pattern: str = "*.yaml") -> list[Path]:
    """Find all YAML configs in a directory (recursively)."""
    return sorted(Path(config_dir).rglob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments in a directory")
    parser.add_argument("--config-dir", required=True, help="Directory with YAML configs")
    parser.add_argument("--pattern", default="*.yaml", help="Glob pattern for config files")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if result exists")
    parser.add_argument("--dry-run", action="store_true", help="List configs without running")
    args = parser.parse_args()

    configs = find_configs(args.config_dir, args.pattern)
    if not configs:
        print(f"No configs found in {args.config_dir} matching {args.pattern}")
        sys.exit(1)

    print(f"Found {len(configs)} experiment configs:")
    for c in configs:
        print(f"  {c}")

    if args.dry_run:
        print("\n[DRY RUN] — would run the above configs. Exiting.")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    successes = 0
    failures = 0

    for i, config_path in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] Loading {config_path.name} ...")
        try:
            config = load_config(str(config_path))

            # Check if result already exists
            if args.output_dir:
                out_dir = Path(args.output_dir)
            else:
                out_dir = Path(config.output.results_dir)
            out_path = out_dir / f"{config.name}.json"

            if args.skip_existing and out_path.exists():
                print(f"  SKIP (result exists: {out_path})")
                successes += 1
                continue

            print(f"  Running: {config.name}")
            print(f"    Dataset: {config.dataset.name} ({config.dataset.max_samples} samples)")
            print(f"    Model: {config.generation.model_path}")
            if config.pipeline.ablation:
                print(f"    Ablation: {config.pipeline.ablation}")
            if config.corruption.type:
                print(f"    Corruption: {config.corruption.type} @ {config.corruption.level}")

            result = run_experiment(config)
            save_result(result, str(out_path))

            print(f"  DONE in {result.timing['total']:.1f}s")
            for method, m in result.metrics.items():
                print(f"    {method}: EM={m.get('em', 0):.3f} F1={m.get('f1', 0):.3f}")
            successes += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failures += 1

    print(f"\n{'=' * 60}")
    print(f"Finished: {successes} succeeded, {failures} failed out of {len(configs)}")


if __name__ == "__main__":
    main()
