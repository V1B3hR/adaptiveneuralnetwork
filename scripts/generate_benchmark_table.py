#!/usr/bin/env python3
"""
Automated README benchmark table generation script.

This script reads benchmark results from JSON files and generates markdown tables
that can be automatically inserted into README.md files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import re


class BenchmarkTableGenerator:
    """Generates markdown tables from benchmark results."""
    
    def __init__(self, results_dir: Union[str, Path] = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.load_all_results()
    
    def load_all_results(self) -> None:
        """Load all benchmark results from the results directory."""
        if not self.results_dir.exists():
            print(f"Warning: Results directory {self.results_dir} does not exist")
            return
        
        # Load main benchmark results file
        main_results_file = self.results_dir.parent / "benchmark_results.json"
        if main_results_file.exists():
            try:
                with open(main_results_file, 'r') as f:
                    main_results = json.load(f)
                self.results.update(main_results)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {main_results_file}: {e}")
        
        # Load individual result files
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                self.results[result_file.stem] = data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {result_file}: {e}")
        
        # Load enhanced results
        enhanced_file = self.results_dir.parent / "enhanced_robustness_results.json"
        if enhanced_file.exists():
            try:
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f)
                self.results["enhanced_robustness"] = enhanced_data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {enhanced_file}: {e}")
        
        # Load adversarial results
        adversarial_file = self.results_dir.parent / "adversarial_results.json"
        if adversarial_file.exists():
            try:
                with open(adversarial_file, 'r') as f:
                    adversarial_data = json.load(f)
                self.results["adversarial"] = adversarial_data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {adversarial_file}: {e}")
    
    def extract_metric_value(self, data: Dict[str, Any], metric_path: str, default: str = "TBD") -> str:
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
                    return f"{current:.3f}"
            elif isinstance(current, int):
                return str(current)
            elif isinstance(current, str):
                return current
            else:
                return str(current)
                
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    
    def generate_main_benchmark_table(self) -> str:
        """Generate the main benchmark comparison table."""
        table_rows = []
        
        # Table header
        header = "| Model Variant | Dataset | Test Acc | Active Node % | Notes |"
        separator = "|---------------|---------|----------|---------------|-------|"
        table_rows.extend([header, separator])
        
        # Define benchmark configurations to look for
        benchmark_configs = [
            {
                "variant": "Adaptive‑100",
                "dataset": "MNIST",
                "result_key": "mnist_100",
                "acc_path": "test_accuracy",
                "nodes_path": "active_node_ratio",
                "notes": "Basic configuration"
            },
            {
                "variant": "Adaptive‑200", 
                "dataset": "MNIST",
                "result_key": "mnist_200",
                "acc_path": "test_accuracy",
                "nodes_path": "active_node_ratio", 
                "notes": "Enhanced capacity"
            },
            {
                "variant": "Adaptive‑128",
                "dataset": "CIFAR10",
                "result_key": "cifar10_128",
                "acc_path": "test_accuracy",
                "nodes_path": "active_node_ratio",
                "notes": "CIFAR-10 baseline"
            },
            {
                "variant": "Adaptive‑JAX",
                "dataset": "CIFAR10", 
                "result_key": "cifar10_jax",
                "acc_path": "test_accuracy",
                "nodes_path": "active_node_ratio",
                "notes": "JIT accelerated"
            },
            {
                "variant": "Multimodal‑Small",
                "dataset": "Text+Image",
                "result_key": "multimodal_small",
                "acc_path": "accuracy",
                "nodes_path": "active_nodes_percent",
                "notes": "Multimodal fusion"
            }
        ]
        
        # Generate table rows
        for config in benchmark_configs:
            result_key = config["result_key"]
            
            # Look for results in various formats
            test_acc = "TBD"
            active_nodes = "TBD"
            
            # Check direct results
            if result_key in self.results:
                result_data = self.results[result_key]
                test_acc = self.extract_metric_value(result_data, config["acc_path"])
                active_nodes = self.extract_metric_value(result_data, config["nodes_path"])
                
                # Convert to percentage if needed
                if active_nodes != "TBD" and active_nodes.replace('.', '').isdigit():
                    val = float(active_nodes)
                    if val <= 1.0:
                        active_nodes = f"{val * 100:.1f}%"
                    else:
                        active_nodes = f"{val:.1f}%"
            
            # Check in nested results
            for main_key, main_data in self.results.items():
                if isinstance(main_data, dict):
                    # Look for experiment results
                    if "experiments" in main_data:
                        for exp in main_data["experiments"]:
                            if isinstance(exp, dict) and config["dataset"].lower() in str(exp).lower():
                                test_acc = self.extract_metric_value(exp, config["acc_path"], test_acc)
                                active_nodes = self.extract_metric_value(exp, config["nodes_path"], active_nodes)
                    
                    # Look for metrics directly
                    if "metrics" in main_data:
                        metrics = main_data["metrics"]
                        test_acc = self.extract_metric_value(metrics, config["acc_path"], test_acc)
                        active_nodes = self.extract_metric_value(metrics, config["nodes_path"], active_nodes)
            
            # Format accuracy as percentage if it's a decimal
            if test_acc != "TBD" and test_acc.replace('.', '').isdigit():
                val = float(test_acc)
                if val <= 1.0:
                    test_acc = f"~{val:.2f}"
                else:
                    test_acc = f"~{val:.1f}%"
            
            row = f"| {config['variant']} | {config['dataset']} | {test_acc} | {active_nodes} | {config['notes']} |"
            table_rows.append(row)
        
        return "\n".join(table_rows)
    
    def generate_robustness_table(self) -> str:
        """Generate robustness evaluation table."""
        table_rows = []
        
        # Table header
        header = "| Attack Type | Success Rate | Recovery Time | Resilience Score | Notes |"
        separator = "|-------------|--------------|---------------|------------------|-------|"
        table_rows.extend([header, separator])
        
        # Look for robustness/adversarial results
        robustness_configs = [
            {
                "attack": "Energy Depletion",
                "success_path": "energy_depletion.success_rate",
                "recovery_path": "energy_depletion.recovery_time", 
                "resilience_path": "energy_depletion.resilience_score",
                "notes": "Coordinated energy drain"
            },
            {
                "attack": "Signal Jamming",
                "success_path": "signal_jamming.success_rate",
                "recovery_path": "signal_jamming.recovery_time",
                "resilience_path": "signal_jamming.resilience_score", 
                "notes": "Communication disruption"
            },
            {
                "attack": "Trust Manipulation",
                "success_path": "trust_manipulation.success_rate",
                "recovery_path": "trust_manipulation.recovery_time",
                "resilience_path": "trust_manipulation.resilience_score",
                "notes": "Social engineering"
            },
            {
                "attack": "Byzantine Faults",
                "success_path": "byzantine_faults.success_rate", 
                "recovery_path": "byzantine_faults.recovery_time",
                "resilience_path": "byzantine_faults.resilience_score",
                "notes": "False information injection"
            }
        ]
        
        # Check for results in adversarial or enhanced robustness data
        result_sources = [
            self.results.get("adversarial", {}),
            self.results.get("enhanced_robustness", {}),
            self.results.get("robustness", {})
        ]
        
        for config in robustness_configs:
            success_rate = "TBD"
            recovery_time = "TBD" 
            resilience_score = "TBD"
            
            for source in result_sources:
                if isinstance(source, dict):
                    success_rate = self.extract_metric_value(source, config["success_path"], success_rate)
                    recovery_time = self.extract_metric_value(source, config["recovery_path"], recovery_time)
                    resilience_score = self.extract_metric_value(source, config["resilience_path"], resilience_score)
                    
                    # Break if we found good values
                    if all(val != "TBD" for val in [success_rate, recovery_time, resilience_score]):
                        break
            
            # Format values
            if success_rate != "TBD" and success_rate.replace('.', '').isdigit():
                val = float(success_rate)
                success_rate = f"{val * 100:.1f}%" if val <= 1.0 else f"{val:.1f}%"
            
            if recovery_time != "TBD" and recovery_time.replace('.', '').isdigit():
                val = float(recovery_time)
                recovery_time = f"{val:.1f}s"
            
            row = f"| {config['attack']} | {success_rate} | {recovery_time} | {resilience_score} | {config['notes']} |"
            table_rows.append(row)
        
        return "\n".join(table_rows)
    
    def generate_performance_table(self) -> str:
        """Generate performance metrics table."""
        table_rows = []
        
        # Table header  
        header = "| Model | Dataset | Training Time | Memory Usage | Energy Efficiency | FLOPs |"
        separator = "|-------|---------|---------------|--------------|-------------------|-------|"
        table_rows.extend([header, separator])
        
        # Performance configs
        perf_configs = [
            {"model": "Adaptive-Base", "dataset": "MNIST", "key": "mnist_base"},
            {"model": "Adaptive-Enhanced", "dataset": "MNIST", "key": "mnist_enhanced"}, 
            {"model": "Adaptive-JAX", "dataset": "CIFAR10", "key": "cifar10_jax"},
            {"model": "Multimodal", "dataset": "Text+Image", "key": "multimodal"}
        ]
        
        for config in perf_configs:
            # Extract performance metrics
            training_time = "TBD"
            memory_usage = "TBD"
            energy_efficiency = "TBD"
            flops = "TBD"
            
            # Look through all results for performance data
            for result_key, result_data in self.results.items():
                if isinstance(result_data, dict):
                    # Check for performance metrics
                    if "performance" in result_data:
                        perf_data = result_data["performance"]
                        training_time = self.extract_metric_value(perf_data, "training_time", training_time)
                        memory_usage = self.extract_metric_value(perf_data, "memory_usage", memory_usage)
                        energy_efficiency = self.extract_metric_value(perf_data, "energy_efficiency", energy_efficiency)
                        flops = self.extract_metric_value(perf_data, "flops", flops)
                    
                    # Check for timing info
                    if "timing" in result_data:
                        timing_data = result_data["timing"] 
                        training_time = self.extract_metric_value(timing_data, "total_time", training_time)
            
            row = f"| {config['model']} | {config['dataset']} | {training_time} | {memory_usage} | {energy_efficiency} | {flops} |"
            table_rows.append(row)
        
        return "\n".join(table_rows)
    
    def update_readme_tables(self, readme_path: Union[str, Path] = "README.md") -> bool:
        """Update benchmark tables in README.md file."""
        readme_path = Path(readme_path)
        
        if not readme_path.exists():
            print(f"Error: README file {readme_path} does not exist")
            return False
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate tables
            main_table = self.generate_main_benchmark_table()
            robustness_table = self.generate_robustness_table()
            performance_table = self.generate_performance_table()
            
            # Update main benchmark table
            main_pattern = r'(\| Model Variant \| Dataset \| Test Acc \| Active Node % \| Notes \|.*?\n\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|)([^#]*?)(?=\n##|\n\*\*|$)'
            if re.search(main_pattern, content, re.DOTALL):
                content = re.sub(main_pattern, main_table, content, flags=re.DOTALL)
            else:
                print("Warning: Could not find main benchmark table pattern in README")
            
            # Add timestamp comment
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            comment = f"\n<!-- Benchmark tables auto-generated on {timestamp} -->\n"
            
            # Write updated content
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content + comment)
            
            print(f"Successfully updated README tables in {readme_path}")
            return True
            
        except (IOError, re.error) as e:
            print(f"Error updating README: {e}")
            return False
    
    def generate_all_tables(self) -> Dict[str, str]:
        """Generate all benchmark tables and return as dictionary."""
        return {
            "main_benchmarks": self.generate_main_benchmark_table(),
            "robustness": self.generate_robustness_table(), 
            "performance": self.generate_performance_table()
        }
    
    def print_tables(self) -> None:
        """Print all generated tables to stdout."""
        tables = self.generate_all_tables()
        
        print("## Main Benchmark Results")
        print(tables["main_benchmarks"])
        print()
        
        print("## Robustness Evaluation")
        print(tables["robustness"])
        print()
        
        print("## Performance Metrics")
        print(tables["performance"])
        print()


def main():
    """Main entry point for the benchmark table generator."""
    parser = argparse.ArgumentParser(description="Generate benchmark tables from results")
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="benchmark_results",
        help="Directory containing benchmark result files"
    )
    parser.add_argument(
        "--readme-path",
        type=str,
        default="README.md", 
        help="Path to README.md file to update"
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update the README.md file with generated tables"
    )
    parser.add_argument(
        "--print-only",
        action="store_true", 
        help="Only print tables to stdout, don't update files"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = BenchmarkTableGenerator(args.results_dir)
    
    if args.print_only:
        generator.print_tables()
    elif args.update_readme:
        success = generator.update_readme_tables(args.readme_path)
        sys.exit(0 if success else 1)
    else:
        # Default: print tables and show file paths
        generator.print_tables()
        print(f"To update README: {sys.argv[0]} --update-readme")
        print(f"Results loaded from: {generator.results_dir}")


if __name__ == "__main__":
    main()