"""Hidden tests for code_parse_date. The checker places the model's code in solution.py next to this file."""
import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import parse_ru_date  # noqa: E402

D = datetime.date


class TestParseRuDate(unittest.TestCase):
    def test_words(self):
        self.assertEqual(parse_ru_date("1 ноября 2026"), D(2026, 11, 1))
        self.assertEqual(parse_ru_date("01 ноября 2026 г."), D(2026, 11, 1))
        self.assertEqual(parse_ru_date("1 ноября 2026 года"), D(2026, 11, 1))
        self.assertEqual(parse_ru_date("5 мая 2027"), D(2027, 5, 5))
        self.assertEqual(parse_ru_date("31 декабря 2026"), D(2026, 12, 31))

    def test_words_case_and_spaces(self):
        self.assertEqual(parse_ru_date("  5   марта  2027  года "), D(2027, 3, 5))
        self.assertEqual(parse_ru_date("1 НОЯБРЯ 2026"), D(2026, 11, 1))
        self.assertEqual(parse_ru_date("12 Января 2026"), D(2026, 1, 12))

    def test_dots(self):
        self.assertEqual(parse_ru_date("01.11.2026"), D(2026, 11, 1))
        self.assertEqual(parse_ru_date("1.11.2026"), D(2026, 11, 1))
        self.assertEqual(parse_ru_date(" 29.02.2028 "), D(2028, 2, 29))

    def test_iso(self):
        self.assertEqual(parse_ru_date("2026-11-01"), D(2026, 11, 1))
        self.assertEqual(parse_ru_date("2026-02-28"), D(2026, 2, 28))

    def test_invalid_dates(self):
        for s in ["31 февраля 2026", "29 февраля 2027", "31.04.2026", "2026-13-01", "0 мая 2026"]:
            with self.subTest(s=s):
                with self.assertRaises(ValueError):
                    parse_ru_date(s)

    def test_invalid_formats(self):
        for s in ["", "   ", "1 ноябрь 2026", "1 nov 2026", "2026/11/01", "01.11.26",
                  "1 ноября 26", "ноября 2026", "завтра", "1 ноября 2026 года и ещё"]:
            with self.subTest(s=s):
                with self.assertRaises(ValueError):
                    parse_ru_date(s)


if __name__ == "__main__":
    unittest.main()
