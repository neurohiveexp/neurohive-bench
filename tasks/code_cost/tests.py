"""Hidden tests for code_cost. The checker places the model's code in solution.py next to this file."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import request_cost  # noqa: E402

PLAIN = {"input": 2.0, "cached": 0.1, "output": 10.0}
LONG = {"input": 2.0, "cached": 0.2, "output": 10.0,
        "long_threshold": 272_000, "long_input_mult": 2.0, "long_output_mult": 1.5}


class TestRequestCost(unittest.TestCase):
    def test_plain(self):
        # 80k uncached * $2 + 20k cached * $0.1 + 5k out * $10, per 1M
        self.assertAlmostEqual(request_cost(100_000, 5_000, 20_000, PLAIN), 0.212, places=6)

    def test_zero(self):
        self.assertEqual(request_cost(0, 0, 0, PLAIN), 0)

    def test_all_cached(self):
        self.assertAlmostEqual(request_cost(50_000, 0, 50_000, PLAIN), 0.005, places=6)

    def test_rounding(self):
        self.assertEqual(request_cost(1, 1, 0, {"input": 1.234567, "cached": 0, "output": 0}), 0.000001)

    def test_long_request(self):
        # 300k uncached * $4 + 10k out * $15, per 1M
        self.assertAlmostEqual(request_cost(300_000, 10_000, 0, LONG), 1.35, places=6)

    def test_long_request_cache_not_multiplied(self):
        # 200k uncached * $4 + 100k cached * $0.2 + 10k out * $15, per 1M
        self.assertAlmostEqual(request_cost(300_000, 10_000, 100_000, LONG), 0.97, places=6)

    def test_threshold_is_strict(self):
        # exactly at the threshold: no surcharge
        self.assertAlmostEqual(request_cost(272_000, 10_000, 0, LONG), 0.644, places=6)

    def test_threshold_without_long_keys_in_plain(self):
        self.assertAlmostEqual(request_cost(1_000_000, 0, 0, PLAIN), 2.0, places=6)

    def test_invalid(self):
        for args in [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (10, 0, 11)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    request_cost(*args, PLAIN)


if __name__ == "__main__":
    unittest.main()
