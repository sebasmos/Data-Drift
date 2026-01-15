"""
Quick sanity check - tests that the main analysis modules can be imported.
"""

import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

def test_batch_analysis_import():
    """Test that batch_analysis.py can be parsed (not executed)."""
    print("Testing batch_analysis.py can be parsed...", end=" ", flush=True)
    try:
        import ast
        with open(os.path.join(os.path.dirname(__file__), '..', 'code', 'batch_analysis.py'), 'r') as f:
            source = f.read()
        ast.parse(source)
        print("OK")
        return True
    except SyntaxError as e:
        print(f"FAILED: Syntax error - {e}")
        return False
    except FileNotFoundError:
        print("FAILED: File not found")
        return False

def test_generate_figures_import():
    """Test that generate_all_figures.py can be parsed."""
    print("Testing generate_all_figures.py can be parsed...", end=" ", flush=True)
    try:
        import ast
        with open(os.path.join(os.path.dirname(__file__), '..', 'code', 'generate_all_figures.py'), 'r') as f:
            source = f.read()
        ast.parse(source)
        print("OK")
        return True
    except SyntaxError as e:
        print(f"FAILED: Syntax error - {e}")
        return False
    except FileNotFoundError:
        print("FAILED: File not found")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("QUICK SANITY CHECK")
    print("=" * 50)

    results = [
        test_batch_analysis_import(),
        test_generate_figures_import(),
    ]

    print("=" * 50)
    if all(results):
        print("All sanity checks PASSED!")
        sys.exit(0)
    else:
        print("Some checks FAILED!")
        sys.exit(1)
