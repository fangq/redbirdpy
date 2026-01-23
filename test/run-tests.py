#!/usr/bin/env python
"""
Redbird Test Runner

Runs all unit tests for the Redbird toolbox.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py -v        # Verbose output
    python run_tests.py -k index  # Run tests matching 'index'
    python run_tests.py --module forward  # Run only forward tests

Requirements:
    - numpy
    - scipy  
    - redbird (the package being tested)
    - iso2mesh (optional, some tests skipped without it)
"""

import unittest
import sys
import argparse
import os


def discover_tests(pattern="test_*.py", start_dir="."):
    """Discover all test modules."""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern=pattern)
    return suite


def run_specific_module(module_name):
    """Run tests for a specific module."""
    loader = unittest.TestLoader()

    module_map = {
        "utility": "test_utility",
        "forward": "test_forward",
        "property": "test_property",
        "recon": "test_recon",
        "integration": "test_integration",
    }

    if module_name in module_map:
        test_module = module_map[module_name]
    else:
        test_module = f"test_{module_name}"

    try:
        suite = loader.loadTestsFromName(test_module)
        return suite
    except ModuleNotFoundError:
        print(f"Test module '{test_module}' not found")
        return None


def run_tests_with_keyword(keyword, verbosity=2):
    """Run tests matching a keyword."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover all tests
    all_tests = discover_tests()

    # Filter by keyword
    def add_matching(test_suite):
        for test in test_suite:
            if isinstance(test, unittest.TestSuite):
                add_matching(test)
            else:
                if keyword.lower() in str(test).lower():
                    suite.addTest(test)

    add_matching(all_tests)

    if suite.countTestCases() == 0:
        print(f"No tests found matching '{keyword}'")
        return False

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def print_test_summary():
    """Print summary of available tests."""
    print("\nRedbird Test Suite")
    print("=" * 50)
    print("\nAvailable test modules:")
    print("  - test_utility.py     : Utility functions (meshprep, sdmap, etc.)")
    print("  - test_forward.py     : Forward modeling (FEM solver, Jacobian)")
    print("  - test_property.py    : Optical properties (extinction, updateprop)")
    print("  - test_recon.py       : Reconstruction (Gauss-Newton, regularization)")
    print("  - test_integration.py : End-to-end and index convention tests")
    print("\nTest categories:")
    print("  - Index convention: Verify 1-based mesh indices")
    print("  - Unit tests: Individual function testing")
    print("  - Integration: Full workflow testing")
    print()


def check_dependencies():
    """Check if required dependencies are available."""
    deps = {
        "numpy": False,
        "scipy": False,
        "redbird": False,
        "iso2mesh": False,
    }

    try:
        import numpy

        deps["numpy"] = True
    except ImportError:
        pass

    try:
        import scipy

        deps["scipy"] = True
    except ImportError:
        pass

    try:
        import redbird

        deps["redbird"] = True
    except ImportError:
        pass

    try:
        import iso2mesh

        deps["iso2mesh"] = True
    except ImportError:
        pass

    print("\nDependency Status:")
    print("-" * 30)
    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Not found"
        print(f"  {dep:15s}: {status}")

    if not deps["redbird"]:
        print("\n⚠ Warning: redbird not found. Most tests will be skipped.")
        print("  Install with: pip install -e .")

    if not deps["iso2mesh"]:
        print("\n⚠ Warning: iso2mesh not found. Some tests will be skipped.")
        print("  Install with: pip install iso2mesh")

    print()
    return deps


def main():
    parser = argparse.ArgumentParser(
        description="Run Redbird unit tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py -v                 # Verbose output
  python run_tests.py --module forward   # Test forward module only
  python run_tests.py -k index           # Run index-related tests
  python run_tests.py --check            # Check dependencies only
  python run_tests.py --list             # List available tests
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (use -vv for more)",
    )
    parser.add_argument(
        "--module", "-m", type=str, help="Run tests for specific module"
    )
    parser.add_argument("-k", "--keyword", type=str, help="Run tests matching keyword")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument(
        "--failfast", "-f", action="store_true", help="Stop on first failure"
    )

    args = parser.parse_args()

    # Check dependencies
    deps = check_dependencies()

    if args.check:
        return 0 if deps["redbird"] else 1

    if args.list:
        print_test_summary()

        # Also list discovered tests
        print("Discovered tests:")
        print("-" * 30)
        suite = discover_tests()

        def print_tests(test_suite, indent=0):
            for test in test_suite:
                if isinstance(test, unittest.TestSuite):
                    print_tests(test, indent)
                else:
                    print(f"{'  ' * indent}{test}")

        print_tests(suite)
        return 0

    # Set verbosity
    verbosity = args.verbose + 1

    # Run tests
    if args.keyword:
        success = run_tests_with_keyword(args.keyword, verbosity)
    elif args.module:
        suite = run_specific_module(args.module)
        if suite:
            runner = unittest.TextTestRunner(
                verbosity=verbosity, failfast=args.failfast
            )
            result = runner.run(suite)
            success = result.wasSuccessful()
        else:
            success = False
    else:
        # Run all tests
        suite = discover_tests()
        runner = unittest.TextTestRunner(verbosity=verbosity, failfast=args.failfast)
        result = runner.run(suite)
        success = result.wasSuccessful()

    # Print summary
    print("\n" + "=" * 50)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
