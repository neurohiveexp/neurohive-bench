"""Hidden tests for code_bugfix. The checker places the model's code in solution.py next to this file."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import paginate, total_pages  # noqa: E402

ITEMS = list(range(1, 26))  # 25 items


class TestPaginate(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(list(paginate(ITEMS, 1)), list(range(1, 11)))

    def test_middle_and_last_page(self):
        self.assertEqual(list(paginate(ITEMS, 2)), list(range(11, 21)))
        self.assertEqual(list(paginate(ITEMS, 3)), [21, 22, 23, 24, 25])

    def test_custom_per_page(self):
        self.assertEqual(list(paginate(ITEMS, 2, per_page=4)), [5, 6, 7, 8])
        self.assertEqual(list(paginate(ITEMS, 25, per_page=1)), [25])

    def test_past_the_end(self):
        self.assertEqual(list(paginate(ITEMS, 4)), [])
        self.assertEqual(list(paginate([], 1)), [])

    def test_invalid(self):
        for page, per_page in [(0, 10), (-1, 10), (1, 0), (1, -5)]:
            with self.subTest(page=page, per_page=per_page):
                with self.assertRaises(ValueError):
                    paginate(ITEMS, page, per_page)


class TestTotalPages(unittest.TestCase):
    def test_values(self):
        self.assertEqual(total_pages(0), 0)
        self.assertEqual(total_pages(1), 1)
        self.assertEqual(total_pages(10), 1)
        self.assertEqual(total_pages(11), 2)
        self.assertEqual(total_pages(25), 3)
        self.assertEqual(total_pages(25, per_page=5), 5)
        self.assertEqual(total_pages(26, per_page=5), 6)

    def test_invalid(self):
        for n, per_page in [(-1, 10), (10, 0), (10, -2)]:
            with self.subTest(n=n, per_page=per_page):
                with self.assertRaises(ValueError):
                    total_pages(n, per_page)


if __name__ == "__main__":
    unittest.main()
