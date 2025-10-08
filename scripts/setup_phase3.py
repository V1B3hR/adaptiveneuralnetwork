#!/usr/bin/env python3
"""
Phase 3 Setup Script - Standardize Tooling

This script helps set up the Phase 3 standardization tooling for the project.
It checks tool availability, runs initial checks, and provides guidance.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
        print(f"Status: {status}")
        return result.returncode == 0
    
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def check_tool_installed(tool_name, check_cmd=None):
    """Check if a tool is installed."""
    if check_cmd is None:
        check_cmd = f"{tool_name} --version"
    
    try:
        result = subprocess.run(
            check_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        version = result.stdout.strip().split('\n')[0]
        print(f"✅ {tool_name}: {version}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {tool_name}: NOT INSTALLED")
        return False


def main():
    """Main setup script."""
    print("=" * 70)
    print("Phase 3: Standardize Tooling - Setup Script")
    print("=" * 70)
    
    # Check project root
    project_root = Path(__file__).parent.parent
    print(f"\nProject root: {project_root}")
    
    # Step 1: Check tool availability
    print("\n" + "=" * 70)
    print("Step 1: Checking Tool Availability")
    print("=" * 70)
    
    tools = {
        "ruff": "ruff --version",
        "black": "black --version",
        "mypy": "mypy --version",
        "pytest": "pytest --version",
        "pre-commit": "pre-commit --version"
    }
    
    all_installed = True
    for tool, cmd in tools.items():
        if not check_tool_installed(tool, cmd):
            all_installed = False
    
    if not all_installed:
        print("\n⚠️  Some tools are not installed!")
        print("Run: pip install -e '.[dev]'")
        return 1
    
    # Step 2: Check configuration files
    print("\n" + "=" * 70)
    print("Step 2: Checking Configuration Files")
    print("=" * 70)
    
    config_files = [
        "pyproject.toml",
        ".editorconfig",
        ".pre-commit-config.yaml",
        "Makefile",
        ".github/workflows/ci.yml"
    ]
    
    all_present = True
    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - NOT FOUND")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Some configuration files are missing!")
        return 1
    
    # Step 3: Run quick checks
    print("\n" + "=" * 70)
    print("Step 3: Running Quick Checks")
    print("=" * 70)
    
    checks_passed = True
    
    # Quick lint check (just count issues, don't fail)
    print("\nRunning quick lint check...")
    result = subprocess.run(
        "ruff check adaptiveneuralnetwork/ --statistics",
        shell=True,
        capture_output=True,
        text=True,
        cwd=project_root
    )
    if "Found" in result.stdout:
        lines = result.stdout.strip().split('\n')
        summary_line = [l for l in lines if "Found" in l]
        if summary_line:
            print(f"  {summary_line[0]}")
    
    # Quick format check
    print("\nRunning quick format check...")
    result = subprocess.run(
        "ruff format --check adaptiveneuralnetwork/ | wc -l",
        shell=True,
        capture_output=True,
        text=True,
        cwd=project_root
    )
    files_to_format = result.stdout.strip()
    print(f"  Files needing formatting: {files_to_format}")
    
    # Step 4: Installation instructions
    print("\n" + "=" * 70)
    print("Step 4: Next Steps")
    print("=" * 70)
    
    print("\n✅ Phase 3 tooling is set up!")
    print("\nTo use the tools:")
    print("  make help              # Show all available commands")
    print("  make lint              # Run linting")
    print("  make format            # Format code")
    print("  make test              # Run tests")
    print("  make all-checks        # Run all checks")
    print("\nTo set up pre-commit hooks:")
    print("  make pre-commit-install")
    print("\nTo format and fix issues automatically:")
    print("  make lint-fix")
    print("  make format")
    
    print("\n" + "=" * 70)
    print("Phase 3 Setup Complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
