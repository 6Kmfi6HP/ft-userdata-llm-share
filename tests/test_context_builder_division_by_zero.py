"""
Test for division by zero bug fix in ContextBuilder

This test verifies that the build_market_context function handles
the edge case where previous close price is zero, which would
cause a ZeroDivisionError without the fix.
"""

import unittest
import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from user_data.strategies.llm_modules.utils.context_builder import ContextBuilder


class TestContextBuilderDivisionByZero(unittest.TestCase):
    """Test case for division by zero bug in ContextBuilder"""

    def setUp(self):
        """Set up test fixtures"""
        self.context_config = {
            "max_context_tokens": 6000
        }
        self.context_builder = ContextBuilder(self.context_config)

    def test_build_market_context_with_zero_previous_close(self):
        """
        Test that build_market_context handles zero previous close price gracefully.

        This test verifies the fix for the bug at context_builder.py:68 where
        division by zero would crash the strategy when prev['close'] is 0.

        Expected behavior:
        - Before fix: ZeroDivisionError is raised
        - After fix: Returns context string with "N/A (invalid previous price)"
        """
        # Create test dataframe with zero previous close price
        test_data = {
            'date': [datetime(2025, 1, 1, 10, 0), datetime(2025, 1, 1, 10, 15)],
            'open': [100.0, 105.0],
            'high': [110.0, 115.0],
            'low': [95.0, 100.0],
            'close': [0.0, 105.0],  # Previous close is 0!
            'volume': [1000.0, 1500.0]
        }

        dataframe = pd.DataFrame(test_data)
        metadata = {'pair': 'BTC/USDT'}

        # This should NOT raise ZeroDivisionError after the fix
        try:
            context = self.context_builder.build_market_context(
                dataframe=dataframe,
                metadata=metadata,
                wallets=None,
                current_trades=None,
                exchange=None
            )

            # Verify the context was built successfully
            self.assertIsInstance(context, str)
            self.assertIn('BTC/USDT', context)

            # Verify it contains the safe error message instead of a calculated percentage
            self.assertIn('N/A (invalid previous price)', context)

            print("✅ Test PASSED: Division by zero is handled gracefully")
            print(f"Context contains safe message: {context.find('N/A (invalid previous price)') != -1}")

        except ZeroDivisionError as e:
            self.fail(f"❌ Test FAILED: ZeroDivisionError was raised despite fix: {e}")

    def test_build_market_context_with_valid_previous_close(self):
        """
        Test that build_market_context calculates price change correctly with valid data.

        This test ensures the fix doesn't break normal operation when prices are valid.
        """
        # Create test dataframe with valid price data
        test_data = {
            'date': [datetime(2025, 1, 1, 10, 0), datetime(2025, 1, 1, 10, 15)],
            'open': [100.0, 105.0],
            'high': [110.0, 115.0],
            'low': [95.0, 100.0],
            'close': [100.0, 105.0],  # Valid previous close
            'volume': [1000.0, 1500.0]
        }

        dataframe = pd.DataFrame(test_data)
        metadata = {'pair': 'BTC/USDT'}

        context = self.context_builder.build_market_context(
            dataframe=dataframe,
            metadata=metadata,
            wallets=None,
            current_trades=None,
            exchange=None
        )

        # Verify the context was built successfully
        self.assertIsInstance(context, str)
        self.assertIn('BTC/USDT', context)

        # Verify it contains a calculated percentage (5% increase from 100 to 105)
        self.assertIn('价格变化:', context)
        self.assertIn('5.00%', context)

        # Verify it does NOT contain the error message
        self.assertNotIn('N/A (invalid previous price)', context)

        print("✅ Test PASSED: Normal price change calculation works correctly")
        print(f"Context contains price change: {context.find('5.00%') != -1}")

    def test_build_market_context_with_single_row(self):
        """
        Test that build_market_context handles single row dataframe.

        When there's only one row, prev equals latest, so if close is 0,
        it would trigger division by zero in the unfixed version.
        """
        # Create test dataframe with single row and zero close
        test_data = {
            'date': [datetime(2025, 1, 1, 10, 0)],
            'open': [100.0],
            'high': [110.0],
            'low': [95.0],
            'close': [0.0],  # Single row with zero close
            'volume': [1000.0]
        }

        dataframe = pd.DataFrame(test_data)
        metadata = {'pair': 'ETH/USDT'}

        # This should NOT raise ZeroDivisionError
        try:
            context = self.context_builder.build_market_context(
                dataframe=dataframe,
                metadata=metadata,
                wallets=None,
                current_trades=None,
                exchange=None
            )

            self.assertIsInstance(context, str)
            self.assertIn('ETH/USDT', context)

            # With single row, prev = latest, both are 0, so change is 0/0
            # Our fix handles this case
            self.assertIn('N/A (invalid previous price)', context)

            print("✅ Test PASSED: Single row with zero close is handled")

        except ZeroDivisionError as e:
            self.fail(f"❌ Test FAILED: ZeroDivisionError with single row: {e}")


def run_tests():
    """Run the test suite and return results"""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContextBuilderDivisionByZero)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
