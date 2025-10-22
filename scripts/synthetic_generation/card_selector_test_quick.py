"""
Quick test runner for common card distribution testing scenarios.

This wrapper makes it easy to run common tests without remembering all the arguments.
"""

import subprocess
import sys
from pathlib import Path

# Test scenarios
SCENARIOS = {
    'quick': {
        'iterations': 1000,
        'selector': 'weighted',
        'description': 'Quick test with 1,000 iterations (~8 seconds)'
    },
    'standard': {
        'iterations': 25000,
        'selector': 'weighted',
        'description': 'Standard test with 25,000 iterations (~3.5 minutes)'
    },
    'large': {
        'iterations': 50000,
        'selector': 'weighted',
        'description': 'Large test with 50,000 iterations (~7 minutes)'
    },
    'quick-smooth': {
        'iterations': 1000,
        'selector': 'smooth',
        'description': 'Quick test of smooth selector (~8 seconds)'
    },
    'standard-smooth': {
        'iterations': 25000,
        'selector': 'smooth',
        'description': 'Standard test of smooth selector (~3.5 minutes)'
    },
    'compare': {
        'iterations': 25000,
        'selector': 'both',
        'description': 'Compare both selectors with 25k each (~7 minutes)'
    },
    'compare-quick': {
        'iterations': 5000,
        'selector': 'both',
        'description': 'Quick comparison with 5k each (~1.5 minutes)'
    }
}


def print_help():
    """Print available scenarios and usage."""
    print("\nCard Distribution Testing - Quick Test Runner")
    print("=" * 80)
    print("\nUsage:")
    print("  python run_quick_test.py <scenario>")
    print("  python run_quick_test.py <scenario> --output <filename>")
    print("  python run_quick_test.py <scenario> --quiet")
    print("\nAvailable scenarios:")
    print("-" * 80)
    
    for name, config in SCENARIOS.items():
        print(f"\n  {name}")
        print(f"    {config['description']}")
        print(f"    Selector: {config['selector']}, Iterations: {config['iterations']:,}")
    
    print("\n" + "=" * 80)
    print("\nExamples:")
    print("  python run_quick_test.py quick")
    print("  python run_quick_test.py standard --quiet")
    print("  python run_quick_test.py compare --output my_comparison.json")
    print("\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print_help()
        return
    
    scenario_name = sys.argv[1]
    
    if scenario_name not in SCENARIOS:
        print(f"Error: Unknown scenario '{scenario_name}'")
        print("\nRun 'python run_quick_test.py help' to see available scenarios.")
        return
    
    scenario = SCENARIOS[scenario_name]
    
    # Build command
    script_path = Path(__file__).parent / 'card_selector_test_main.py'
    cmd = [
        sys.executable,
        str(script_path),
        '--iterations', str(scenario['iterations']),
        '--selector', scenario['selector']
    ]
    
    # Add optional arguments
    extra_args = sys.argv[2:]
    cmd.extend(extra_args)
    
    # Print scenario info
    print("\n" + "=" * 80)
    print(f"Running scenario: {scenario_name}")
    print("=" * 80)
    print(f"Description: {scenario['description']}")
    print(f"Selector: {scenario['selector']}")
    print(f"Iterations: {scenario['iterations']:,}")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")
    print("=" * 80)
    print()
    
    # Run the test
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
