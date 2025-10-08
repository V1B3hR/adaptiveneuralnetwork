#!/usr/bin/env python3
"""
Robustness Guard CLI - Robustness threshold guard.

Checks robustness scores against thresholds and alerts if robustness is below acceptable levels.
Exits 0 when data is missing or within thresholds, exits 1 when violations are detected.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class RobustnessGuard:
    """Guards against robustness degradation."""

    def __init__(self, results_dir: Path = Path(".")):
        self.results_dir = results_dir
        self.robustness_data = None
        self.adversarial_data = None
        self.load_robustness_data()

    def load_robustness_data(self) -> None:
        """Load robustness and adversarial test data."""
        # Load enhanced robustness results
        robustness_file = self.results_dir / "enhanced_robustness_results.json"
        if robustness_file.exists():
            try:
                with open(robustness_file) as f:
                    self.robustness_data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load {robustness_file}: {e}")

        # Load adversarial results
        adversarial_file = self.results_dir / "adversarial_results.json"
        if adversarial_file.exists():
            try:
                with open(adversarial_file) as f:
                    self.adversarial_data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load {adversarial_file}: {e}")

    def extract_metric_value(self, data: dict[str, Any], metric_path: str) -> float | None:
        """Extract metric value from nested dictionary using dot notation."""
        keys = metric_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None

            if isinstance(current, (int, float)):
                return float(current)
            return None
        except (KeyError, TypeError):
            return None

    def check_robustness_thresholds(
        self,
        min_robustness_score: float = 50.0,
        min_adversarial_score: float = 30.0,
        min_scenario_pass_rate: float = 0.6
    ) -> tuple[bool, str]:
        """
        Check robustness metrics against thresholds.
        
        Args:
            min_robustness_score: Minimum overall robustness score (0-100)
            min_adversarial_score: Minimum adversarial resilience score (0-100)  
            min_scenario_pass_rate: Minimum scenario pass rate (0.0-1.0)
            
        Returns:
            Tuple of (is_violation, message)
        """
        violations = []
        warnings = []

        # Check enhanced robustness results
        if self.robustness_data:
            # Overall robustness score
            robustness_score = self.extract_metric_value(
                self.robustness_data, "overall_robustness_score"
            )
            if robustness_score is not None:
                if robustness_score < min_robustness_score:
                    violations.append(
                        f"Robustness score {robustness_score:.1f} below threshold {min_robustness_score}"
                    )
                else:
                    warnings.append(
                        f"Robustness score {robustness_score:.1f} meets threshold {min_robustness_score}"
                    )

            # Scenario pass rate
            scenarios_tested = self.extract_metric_value(
                self.robustness_data, "scenario_validation.scenarios_tested"
            )
            scenarios_passed = self.extract_metric_value(
                self.robustness_data, "scenario_validation.scenarios_passed"
            )

            if scenarios_tested and scenarios_passed:
                pass_rate = scenarios_passed / scenarios_tested
                if pass_rate < min_scenario_pass_rate:
                    violations.append(
                        f"Scenario pass rate {pass_rate:.2f} below threshold {min_scenario_pass_rate}"
                    )
                else:
                    warnings.append(
                        f"Scenario pass rate {pass_rate:.2f} meets threshold {min_scenario_pass_rate}"
                    )
        else:
            warnings.append("Enhanced robustness data not available")

        # Check adversarial results
        if self.adversarial_data:
            # Adversarial resilience score
            adversarial_score = self.extract_metric_value(
                self.adversarial_data, "adversarial_resilience_score"
            )
            if adversarial_score is not None:
                if adversarial_score < min_adversarial_score:
                    violations.append(
                        f"Adversarial resilience {adversarial_score:.1f} below threshold {min_adversarial_score}"
                    )
                else:
                    warnings.append(
                        f"Adversarial resilience {adversarial_score:.1f} meets threshold {min_adversarial_score}"
                    )
        else:
            warnings.append("Adversarial test data not available")

        # Prepare result message
        if violations:
            message = "Robustness violations detected:\n" + "\n".join(f"  - {v}" for v in violations)
            if warnings:
                message += "\nAdditional info:\n" + "\n".join(f"  - {w}" for w in warnings)
            return True, message
        else:
            if warnings:
                message = "Robustness check passed:\n" + "\n".join(f"  - {w}" for w in warnings)
            else:
                message = "Robustness data not available - skipping robustness guard"
            return False, message


def main() -> None:
    """Main entry point for the robustness guard CLI."""
    parser = argparse.ArgumentParser(
        description="Robustness guard - Check robustness metrics against thresholds"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=".",
        help="Directory containing robustness results (default: current directory)"
    )
    parser.add_argument(
        "--min-robustness",
        type=float,
        default=50.0,
        help="Minimum overall robustness score (0-100, default: 50.0)"
    )
    parser.add_argument(
        "--min-adversarial",
        type=float,
        default=30.0,
        help="Minimum adversarial resilience score (0-100, default: 30.0)"
    )
    parser.add_argument(
        "--min-scenario-pass-rate",
        type=float,
        default=0.6,
        help="Minimum scenario pass rate (0.0-1.0, default: 0.6)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except for violations"
    )

    args = parser.parse_args()

    # Create robustness guard
    guard = RobustnessGuard(Path(args.results_dir))

    # Check for violations
    is_violation, message = guard.check_robustness_thresholds(
        args.min_robustness,
        args.min_adversarial,
        args.min_scenario_pass_rate
    )

    if not args.quiet or is_violation:
        print(message)

    # Exit with appropriate code
    sys.exit(1 if is_violation else 0)


if __name__ == "__main__":
    main()
