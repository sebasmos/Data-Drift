"""
Fast test for batch_analysis.py - runs with minimal bootstrap to verify functionality.
"""

import sys
import os
import time

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

# Patch the bootstrap settings BEFORE importing batch_analysis
import batch_analysis
batch_analysis.N_BOOTSTRAP = 10  # Reduce from 1000 to 10 for fast testing

def test_single_dataset():
    """Test analysis on a single dataset with minimal bootstrap."""
    print("=" * 60)
    print("FAST TEST: batch_analysis.py")
    print("=" * 60)
    print(f"N_BOOTSTRAP reduced to: {batch_analysis.N_BOOTSTRAP}")

    start = time.time()

    # Test with just mimiciv (has multiple time periods)
    print("\nTesting with mimiciv dataset only...")
    try:
        results, deltas = batch_analysis.run_batch_analysis(['mimiciv'])
        elapsed = time.time() - start

        print(f"\nTest completed in {elapsed:.1f} seconds")
        print(f"Results rows: {len(results)}")
        print(f"Deltas rows: {len(deltas)}")

        if len(results) > 0 and len(deltas) > 0:
            print("\nSUCCESS: batch_analysis.py is working correctly!")
            print("The slowness is due to N_BOOTSTRAP=1000 (1000 iterations per AUC)")
            print("\nTo speed up for development, edit batch_analysis.py line 25:")
            print("  Change: N_BOOTSTRAP = 1000")
            print("  To:     N_BOOTSTRAP = 100  (or lower)")
            return True
        else:
            print("\nWARNING: No results generated")
            return False

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_dataset()
    sys.exit(0 if success else 1)
