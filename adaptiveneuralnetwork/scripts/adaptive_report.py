#!/usr/bin/env python3
"""
Adaptive Report CLI - Consolidates metrics into Markdown reports.

This script reads JSON artifacts from benchmarks, robustness tests, and adversarial
evaluations to generate comprehensive Markdown reports.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Import the existing table generator to reuse its functionality
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
from generate_benchmark_table import BenchmarkTableGenerator


class AdaptiveReportGenerator:
    """Generates comprehensive Markdown reports from benchmark and validation results."""

    def __init__(self, results_dir: str | Path = "."):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.load_all_artifacts()

    def load_all_artifacts(self) -> None:
        """Load all JSON artifacts from the results directory."""
        # JSON files to look for
        artifact_files = [
            "benchmark_results.json",
            "enhanced_robustness_results.json",
            "adversarial_results.json",
            "final_validation.json"
        ]

        # Load main artifacts from the results directory
        for artifact_file in artifact_files:
            file_path = self.results_dir / artifact_file
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    self.results[artifact_file.replace('.json', '')] = data
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {file_path}: {e}")

        # Load additional files from benchmark_results/ subdirectory
        benchmark_results_dir = self.results_dir / "benchmark_results"
        if benchmark_results_dir.exists():
            for result_file in benchmark_results_dir.glob("*.json"):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    self.results[f"benchmark_results_{result_file.stem}"] = data
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {result_file}: {e}")

    def extract_metric_value(self, data: dict[str, Any], metric_path: str, default: str = "N/A") -> str:
        """Extract metric value from nested dictionary using dot notation."""
        keys = metric_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return default

            # Format the value appropriately
            if isinstance(current, float):
                if current < 0.01:
                    return f"{current:.4f}"
                else:
                    return f"{current:.2f}"
            elif isinstance(current, int):
                return str(current)
            elif isinstance(current, str):
                return current
            else:
                return str(current)

        except (KeyError, IndexError, ValueError, TypeError):
            return default

    def generate_benchmarks_summary(self) -> str:
        """Generate benchmarks summary table using existing BenchmarkTableGenerator logic."""
        # Create a BenchmarkTableGenerator instance with the same data
        table_generator = BenchmarkTableGenerator(self.results_dir / "benchmark_results")

        # Override its results with our loaded data to ensure consistency
        table_generator.results = self.results

        # Generate the main benchmark table
        return table_generator.generate_main_benchmark_table()

    def generate_robustness_adversarial_summary(self) -> str:
        """Generate robustness and adversarial summary section."""
        summary_lines = []

        # Enhanced robustness results
        if "enhanced_robustness_results" in self.results:
            robustness_data = self.results["enhanced_robustness_results"]
            summary_lines.append("### Enhanced Robustness Results")
            summary_lines.append("")

            # Overall metrics
            overall_score = self.extract_metric_value(robustness_data, "overall_robustness_score", "N/A")
            total_duration = self.extract_metric_value(robustness_data, "total_duration_seconds", "N/A")

            if overall_score != "N/A":
                summary_lines.append(f"- **Overall Robustness Score**: {overall_score}/100")
            if total_duration != "N/A":
                summary_lines.append(f"- **Total Test Duration**: {total_duration}s")

            # Scenario validation
            if "scenario_validation" in robustness_data:
                scenario_data = robustness_data["scenario_validation"]
                scenarios_tested = self.extract_metric_value(scenario_data, "scenarios_tested", "N/A")
                scenarios_passed = self.extract_metric_value(scenario_data, "scenarios_passed", "N/A")

                if scenarios_tested != "N/A" and scenarios_passed != "N/A":
                    try:
                        pass_rate = (float(scenarios_passed) / float(scenarios_tested)) * 100
                        summary_lines.append(f"- **Scenario Validation**: {scenarios_passed}/{scenarios_tested} ({pass_rate:.1f}% passed)")
                    except (ValueError, ZeroDivisionError):
                        summary_lines.append(f"- **Scenario Validation**: {scenarios_passed}/{scenarios_tested}")

            # Relative robustness if available
            relative_robustness = self.extract_metric_value(robustness_data, "relative_robustness", None)
            if relative_robustness:
                summary_lines.append(f"- **Relative Robustness**: {relative_robustness}")

            summary_lines.append("")

        # Adversarial results
        if "adversarial_results" in self.results:
            adversarial_data = self.results["adversarial_results"]
            summary_lines.append("### Adversarial Resilience Results")
            summary_lines.append("")

            # Key metrics
            resilience_score = self.extract_metric_value(adversarial_data, "adversarial_resilience_score", "N/A")
            tests_passed = self.extract_metric_value(adversarial_data, "tests_passed", "N/A")
            total_tests = self.extract_metric_value(adversarial_data, "total_tests", "N/A")
            avg_degradation = self.extract_metric_value(adversarial_data, "average_performance_degradation", "N/A")

            if resilience_score != "N/A":
                summary_lines.append(f"- **Adversarial Resilience Score**: {resilience_score}/100")
            if tests_passed != "N/A" and total_tests != "N/A":
                try:
                    pass_rate = (float(tests_passed) / float(total_tests)) * 100
                    summary_lines.append(f"- **Tests Passed**: {tests_passed}/{total_tests} ({pass_rate:.1f}%)")
                except (ValueError, ZeroDivisionError):
                    summary_lines.append(f"- **Tests Passed**: {tests_passed}/{total_tests}")
            if avg_degradation != "N/A":
                summary_lines.append(f"- **Average Performance Degradation**: {avg_degradation}%")

            # Adversarial retention if available
            adversarial_retention = self.extract_metric_value(adversarial_data, "adversarial_retention", None)
            if adversarial_retention:
                summary_lines.append(f"- **Adversarial Retention**: {adversarial_retention}")

            summary_lines.append("")

        return "\n".join(summary_lines) if summary_lines else ""

    def generate_phase_sparsity_snapshot(self) -> str:
        """Generate phase/sparsity snapshot section."""
        snapshot_lines = []

        # Look for active node ratio or active phase ratio in various results
        found_metrics = False

        # Check all result sources for phase/sparsity metrics
        for result_name, result_data in self.results.items():
            if isinstance(result_data, dict):
                # Look for active_node_ratio
                active_node_ratio = self.extract_metric_value(result_data, "active_node_ratio", None)
                if active_node_ratio:
                    if not found_metrics:
                        snapshot_lines.append("### Phase / Sparsity Snapshot")
                        snapshot_lines.append("")
                        found_metrics = True
                    snapshot_lines.append(f"- **Active Node Ratio** ({result_name}): {active_node_ratio}")

                # Look for active_phase_ratio
                active_phase_ratio = self.extract_metric_value(result_data, "active_phase_ratio", None)
                if active_phase_ratio:
                    if not found_metrics:
                        snapshot_lines.append("### Phase / Sparsity Snapshot")
                        snapshot_lines.append("")
                        found_metrics = True
                    snapshot_lines.append(f"- **Active Phase Ratio** ({result_name}): {active_phase_ratio}")

                # Check nested structures for phase information
                if "phase_distribution" in result_data:
                    phase_dist = result_data["phase_distribution"]
                    if isinstance(phase_dist, dict):
                        if not found_metrics:
                            snapshot_lines.append("### Phase / Sparsity Snapshot")
                            snapshot_lines.append("")
                            found_metrics = True
                        snapshot_lines.append(f"- **Phase Distribution** ({result_name}):")
                        for phase, ratio in phase_dist.items():
                            snapshot_lines.append(f"  - {phase}: {ratio}")

                # Check for sparsity metrics
                if "sparsity" in result_data:
                    sparsity = result_data["sparsity"]
                    if not found_metrics:
                        snapshot_lines.append("### Phase / Sparsity Snapshot")
                        snapshot_lines.append("")
                        found_metrics = True
                    snapshot_lines.append(f"- **Sparsity** ({result_name}): {sparsity}")

        if found_metrics:
            snapshot_lines.append("")

        return "\n".join(snapshot_lines) if snapshot_lines else ""

    def generate_full_report(self) -> str:
        """Generate the complete Markdown report."""
        report_lines = []

        # Report header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        report_lines.append("# Adaptive Neural Network - Consolidated Report")
        report_lines.append(f"**Generated**: {timestamp}")
        report_lines.append("")

        # Check if we have any data
        if not self.results:
            report_lines.append("⚠️ **No benchmark data found** - Please run benchmarks to generate report content.")
            report_lines.append("")
            report_lines.append("Expected files:")
            report_lines.append("- `benchmark_results.json`")
            report_lines.append("- `enhanced_robustness_results.json`")
            report_lines.append("- `adversarial_results.json`")
            report_lines.append("- `final_validation.json`")
            report_lines.append("- Files under `benchmark_results/` directory")
            report_lines.append("")
            return "\n".join(report_lines)

        # Summary section
        report_lines.append("## Summary")
        report_lines.append("")
        total_files = len(self.results)
        report_lines.append(f"This report consolidates metrics from {total_files} JSON artifact(s):")
        for artifact_name in sorted(self.results.keys()):
            report_lines.append(f"- {artifact_name}")
        report_lines.append("")

        # Benchmarks summary table
        benchmarks_table = self.generate_benchmarks_summary()
        if benchmarks_table.strip():
            report_lines.append("## Benchmarks Summary")
            report_lines.append("")
            report_lines.append(benchmarks_table)
            report_lines.append("")

        # Robustness and adversarial summary
        robustness_section = self.generate_robustness_adversarial_summary()
        if robustness_section.strip():
            report_lines.append("## Robustness & Adversarial Summary")
            report_lines.append("")
            report_lines.append(robustness_section)

        # Phase/sparsity snapshot
        phase_section = self.generate_phase_sparsity_snapshot()
        if phase_section.strip():
            report_lines.append(phase_section)

        # Add footer
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*This report was automatically generated by the `adaptive-report` CLI tool.*")
        report_lines.append("")

        return "\n".join(report_lines)


def main() -> None:
    """Main entry point for the adaptive-report CLI."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive Markdown reports from benchmark and validation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=".",
        help="Directory containing JSON artifacts (default: current directory)"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output file path (default: stdout)"
    )

    args = parser.parse_args()

    # Create report generator
    generator = AdaptiveReportGenerator(args.results_dir)

    # Generate report
    report = generator.generate_full_report()

    # Output to file or stdout
    if args.out:
        try:
            with open(args.out, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report written to: {args.out}")
        except OSError as e:
            print(f"Error writing to {args.out}: {e}")
            sys.exit(1)
    else:
        print(report)

    sys.exit(0)


if __name__ == "__main__":
    main()
