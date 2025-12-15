"""
Library Import Tests for Data-Drift Project
Tests that all required libraries can be imported and basic functionality works.
"""

import sys
import time

def test_import(module_name, timeout_warning=5):
    """Test importing a module and measure time."""
    print(f"Testing import: {module_name}...", end=" ", flush=True)
    start = time.time()
    try:
        __import__(module_name)
        elapsed = time.time() - start
        status = "OK" if elapsed < timeout_warning else f"SLOW ({elapsed:.1f}s)"
        print(f"{status} ({elapsed:.2f}s)")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED ({elapsed:.2f}s)")
        print(f"  Error: {e}")
        return False, elapsed

def test_numpy():
    """Test numpy basic operations."""
    print("\nTesting numpy operations...", end=" ", flush=True)
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        assert arr.sum() == 15
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_pandas():
    """Test pandas basic operations."""
    print("Testing pandas operations...", end=" ", flush=True)
    try:
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        assert df['a'].sum() == 6
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_scipy():
    """Test scipy basic operations."""
    print("Testing scipy operations...", end=" ", flush=True)
    try:
        from scipy import stats
        result = stats.ttest_ind([1, 2, 3], [4, 5, 6])
        assert result.pvalue is not None
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_matplotlib():
    """Test matplotlib basic operations."""
    print("Testing matplotlib operations...", end=" ", flush=True)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_seaborn():
    """Test seaborn basic operations."""
    print("Testing seaborn operations...", end=" ", flush=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        tips = sns.load_dataset("tips") if False else None  # Skip loading dataset
        print("OK (import only)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def main():
    print("=" * 60)
    print("DATA-DRIFT LIBRARY TEST SUITE")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("=" * 60)

    # Test imports
    print("\n[1] IMPORT TESTS")
    print("-" * 40)

    libraries = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "tabulate",
    ]

    results = {}
    total_time = 0
    for lib in libraries:
        success, elapsed = test_import(lib)
        results[lib] = success
        total_time += elapsed

    print(f"\nTotal import time: {total_time:.2f}s")

    # Test functionality
    print("\n[2] FUNCTIONALITY TESTS")
    print("-" * 40)

    func_results = {
        'numpy': test_numpy(),
        'pandas': test_pandas(),
        'scipy': test_scipy(),
        'matplotlib': test_matplotlib(),
        'seaborn': test_seaborn(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results.values()) and all(func_results.values())
    failed_imports = [k for k, v in results.items() if not v]
    failed_funcs = [k for k, v in func_results.items() if not v]

    if all_passed:
        print("All tests PASSED!")
    else:
        if failed_imports:
            print(f"Failed imports: {', '.join(failed_imports)}")
        if failed_funcs:
            print(f"Failed functionality: {', '.join(failed_funcs)}")

    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
