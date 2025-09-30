#!/usr/bin/env python3
"""
Phase 0 - Dependency Graph Generator
Generates a simplified dependency visualization of the module structure.
"""
import os
import ast
import json
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict


def extract_imports(file_path: Path) -> Set[str]:
    """Extract import statements from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        # Skip files that can't be parsed
        pass
    
    return imports


def analyze_module_dependencies(root_path: Path) -> Dict[str, Set[str]]:
    """
    Analyze dependencies between modules in the package.
    
    Returns:
        Dictionary mapping module names to sets of their dependencies
    """
    package_path = root_path / "adaptiveneuralnetwork"
    
    # Module directories to analyze
    module_dirs = [
        "data", "models", "training", "core", "utils", "api",
        "benchmarks", "scripts", "neuromorphic", "production",
        "applications", "automl", "ecosystem"
    ]
    
    # Track dependencies
    dependencies = defaultdict(set)
    
    for module_name in module_dirs:
        module_path = package_path / module_name
        if not module_path.exists():
            continue
        
        # Collect all imports from this module
        module_imports = set()
        for py_file in module_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            imports = extract_imports(py_file)
            module_imports.update(imports)
        
        # Filter to only internal module dependencies
        internal_deps = set()
        for imp in module_imports:
            # Check if this import refers to another internal module
            if imp == "adaptiveneuralnetwork":
                continue  # Skip self-reference
            # Check if the import matches any of our module names
            for other_module in module_dirs:
                if other_module != module_name and imp == other_module:
                    internal_deps.add(other_module)
        
        dependencies[module_name] = internal_deps
    
    return dict(dependencies)


def generate_mermaid_diagram(dependencies: Dict[str, Set[str]]) -> str:
    """Generate a Mermaid diagram from dependencies."""
    lines = ["```mermaid", "graph TD"]
    
    # Define all nodes
    for module in sorted(dependencies.keys()):
        node_label = module.replace("_", " ").title()
        lines.append(f"    {module}[{node_label}]")
    
    # Define edges
    edges = []
    for module, deps in sorted(dependencies.items()):
        for dep in sorted(deps):
            edges.append(f"    {module} --> {dep}")
    
    lines.extend(edges)
    lines.append("```")
    
    return "\n".join(lines)


def generate_dependency_report(dependencies: Dict[str, Set[str]]) -> str:
    """Generate a markdown report of dependencies."""
    report = []
    report.append("# Phase 0 - Module Dependency Analysis\n")
    report.append("## Internal Module Dependencies\n")
    
    # Calculate metrics
    total_modules = len(dependencies)
    total_deps = sum(len(deps) for deps in dependencies.values())
    
    report.append(f"- **Total Modules**: {total_modules}")
    report.append(f"- **Total Internal Dependencies**: {total_deps}")
    report.append(f"- **Average Dependencies per Module**: {total_deps / total_modules:.1f}\n")
    
    # Most depended-on modules
    dep_count = defaultdict(int)
    for deps in dependencies.values():
        for dep in deps:
            dep_count[dep] += 1
    
    if dep_count:
        report.append("## Most Depended-On Modules\n")
        sorted_deps = sorted(dep_count.items(), key=lambda x: x[1], reverse=True)[:5]
        for module, count in sorted_deps:
            report.append(f"- **{module}**: {count} modules depend on it")
        report.append("")
    
    # Modules with most dependencies
    report.append("## Modules with Most Dependencies\n")
    sorted_modules = sorted(
        dependencies.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:5]
    for module, deps in sorted_modules:
        report.append(f"- **{module}**: depends on {len(deps)} modules")
        if deps:
            dep_list = ", ".join(sorted(deps))
            report.append(f"  - Dependencies: {dep_list}")
    report.append("")
    
    # Dependency graph visualization
    report.append("## Dependency Graph (Simplified)\n")
    report.append(generate_mermaid_diagram(dependencies))
    report.append("")
    
    # Detailed dependencies
    report.append("## Detailed Module Dependencies\n")
    for module in sorted(dependencies.keys()):
        deps = dependencies[module]
        report.append(f"### {module}")
        if deps:
            report.append("**Depends on**:")
            for dep in sorted(deps):
                report.append(f"- {dep}")
        else:
            report.append("*No internal dependencies*")
        report.append("")
    
    # External dependencies note
    report.append("## External Dependencies\n")
    report.append("Key external packages used throughout the system:")
    report.append("- **torch**: Deep learning framework (core functionality)")
    report.append("- **numpy**: Numerical computing (data processing)")
    report.append("- **scipy**: Scientific computing (advanced algorithms)")
    report.append("- **pyyaml**: Configuration management")
    report.append("- **rich**: CLI output formatting\n")
    
    return "\n".join(report)


def main():
    """Main entry point for the dependency analyzer."""
    # Get the repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    
    print("=" * 60)
    print("Phase 0 - Dependency Analysis")
    print("=" * 60)
    print(f"\nAnalyzing repository at: {repo_root}\n")
    
    # Analyze dependencies
    dependencies = analyze_module_dependencies(repo_root)
    
    # Save JSON
    json_path = repo_root / "benchmarks" / "module_dependencies.json"
    with open(json_path, 'w') as f:
        # Convert sets to lists for JSON serialization
        json_deps = {k: sorted(list(v)) for k, v in dependencies.items()}
        json.dump(json_deps, f, indent=2)
    print(f"✓ Saved dependencies to: {json_path}")
    
    # Generate and save report
    report = generate_dependency_report(dependencies)
    md_path = repo_root / "docs" / "phase0" / "dependency_graph.md"
    with open(md_path, 'w') as f:
        f.write(report)
    print(f"✓ Saved report to: {md_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total Modules: {len(dependencies)}")
    total_deps = sum(len(deps) for deps in dependencies.values())
    print(f"Total Internal Dependencies: {total_deps}")
    print("=" * 60)


if __name__ == "__main__":
    main()
