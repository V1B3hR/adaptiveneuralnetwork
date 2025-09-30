#!/usr/bin/env python3
"""
Phase 0 - Module Inventory Script
Lists all modules, their structure, and lines of code.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def count_lines_of_code(file_path: Path) -> Tuple[int, int, int]:
    """Count total lines, code lines, and comment lines in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total = len(lines)
        code = 0
        comments = 0
        blank = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank += 1
            elif stripped.startswith('#'):
                comments += 1
            else:
                code += 1
        
        return total, code, comments
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return 0, 0, 0


def inventory_modules(root_path: Path) -> Dict[str, any]:
    """
    Inventory all modules in the adaptiveneuralnetwork package.
    Returns a structured dictionary of modules and their stats.
    """
    package_path = root_path / "adaptiveneuralnetwork"
    
    if not package_path.exists():
        raise ValueError(f"Package path not found: {package_path}")
    
    inventory = {
        "timestamp": None,
        "root_path": str(root_path),
        "modules": {},
        "summary": {
            "total_files": 0,
            "total_lines": 0,
            "total_code_lines": 0,
            "total_comment_lines": 0
        }
    }
    
    # Define module categories
    module_dirs = [
        "data",
        "models", 
        "training",
        "core",
        "utils",
        "api",
        "benchmarks",
        "scripts",
        "neuromorphic",
        "production",
        "applications",
        "automl",
        "ecosystem"
    ]
    
    for module_name in module_dirs:
        module_path = package_path / module_name
        
        if not module_path.exists():
            continue
        
        module_info = {
            "path": str(module_path.relative_to(root_path)),
            "files": [],
            "stats": {
                "file_count": 0,
                "total_lines": 0,
                "code_lines": 0,
                "comment_lines": 0
            }
        }
        
        # Find all Python files in this module
        for py_file in module_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            total, code, comments = count_lines_of_code(py_file)
            
            file_info = {
                "name": py_file.name,
                "path": str(py_file.relative_to(root_path)),
                "total_lines": total,
                "code_lines": code,
                "comment_lines": comments
            }
            
            module_info["files"].append(file_info)
            module_info["stats"]["file_count"] += 1
            module_info["stats"]["total_lines"] += total
            module_info["stats"]["code_lines"] += code
            module_info["stats"]["comment_lines"] += comments
        
        if module_info["stats"]["file_count"] > 0:
            inventory["modules"][module_name] = module_info
            inventory["summary"]["total_files"] += module_info["stats"]["file_count"]
            inventory["summary"]["total_lines"] += module_info["stats"]["total_lines"]
            inventory["summary"]["total_code_lines"] += module_info["stats"]["code_lines"]
            inventory["summary"]["total_comment_lines"] += module_info["stats"]["comment_lines"]
    
    return inventory


def generate_markdown_report(inventory: Dict) -> str:
    """Generate a markdown report from the inventory."""
    report = []
    report.append("# Phase 0 - Module Inventory Report\n")
    report.append(f"Root Path: `{inventory['root_path']}`\n")
    report.append("## Summary\n")
    report.append(f"- **Total Files**: {inventory['summary']['total_files']}")
    report.append(f"- **Total Lines**: {inventory['summary']['total_lines']:,}")
    report.append(f"- **Code Lines**: {inventory['summary']['total_code_lines']:,}")
    report.append(f"- **Comment Lines**: {inventory['summary']['total_comment_lines']:,}")
    report.append(f"- **Blank Lines**: {inventory['summary']['total_lines'] - inventory['summary']['total_code_lines'] - inventory['summary']['total_comment_lines']:,}\n")
    
    report.append("## Modules\n")
    
    # Sort modules by code lines (descending)
    sorted_modules = sorted(
        inventory["modules"].items(),
        key=lambda x: x[1]["stats"]["code_lines"],
        reverse=True
    )
    
    for module_name, module_info in sorted_modules:
        stats = module_info["stats"]
        report.append(f"### {module_name}")
        report.append(f"- **Path**: `{module_info['path']}`")
        report.append(f"- **Files**: {stats['file_count']}")
        report.append(f"- **Total Lines**: {stats['total_lines']:,}")
        report.append(f"- **Code Lines**: {stats['code_lines']:,}")
        report.append(f"- **Comment Lines**: {stats['comment_lines']:,}\n")
        
        # List top 5 largest files in this module
        if len(module_info["files"]) > 0:
            sorted_files = sorted(
                module_info["files"],
                key=lambda x: x["code_lines"],
                reverse=True
            )[:5]
            
            report.append("**Top Files**:")
            for file_info in sorted_files:
                report.append(f"- `{file_info['name']}`: {file_info['code_lines']} lines")
            report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point for the inventory script."""
    import datetime
    
    # Get the repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    
    print("=" * 60)
    print("Phase 0 - Module Inventory")
    print("=" * 60)
    print(f"\nAnalyzing repository at: {repo_root}\n")
    
    # Run inventory
    inventory = inventory_modules(repo_root)
    inventory["timestamp"] = datetime.datetime.now().isoformat()
    
    # Save JSON
    json_path = repo_root / "benchmarks" / "module_inventory.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(inventory, f, indent=2)
    print(f"✓ Saved inventory to: {json_path}")
    
    # Generate and save markdown report
    report = generate_markdown_report(inventory)
    md_path = repo_root / "docs" / "phase0" / "module_inventory.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write(report)
    print(f"✓ Saved report to: {md_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total Modules: {len(inventory['modules'])}")
    print(f"Total Files: {inventory['summary']['total_files']}")
    print(f"Total Lines: {inventory['summary']['total_lines']:,}")
    print(f"Code Lines: {inventory['summary']['total_code_lines']:,}")
    print(f"Comment Lines: {inventory['summary']['total_comment_lines']:,}")
    print("=" * 60)
    
    # Show top 5 modules by code size
    print("\nTop 5 Modules by Code Lines:")
    sorted_modules = sorted(
        inventory["modules"].items(),
        key=lambda x: x[1]["stats"]["code_lines"],
        reverse=True
    )[:5]
    for i, (name, info) in enumerate(sorted_modules, 1):
        print(f"{i}. {name}: {info['stats']['code_lines']:,} lines")


if __name__ == "__main__":
    main()
