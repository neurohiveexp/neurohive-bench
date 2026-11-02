"""Hidden acceptance tests for greenfield_meter_bills (named so that pytest of the bench repo skips it). Copied as test_hidden_meter.py into the workspace after the agent finishes."""
import os
import subprocess
import sys
import tempfile
import unittest

APP = os.environ.get("METER_APP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "meter.py"))

DATA = """apartment,date,cold,hot
12,2026-03-25,105.500,40.250
3,2026-02-20,10.000,5.000
12,2026-02-25,100.000,38.000
3,2026-03-05,11.000,5.500
3,2026-03-28,12.345,6.001
12А,2026-03-20,1.000,1.000
7,2026-01-31,50.000,20.000
7,2026-03-01,52.000,21.000
12,2026-03-10,103.000,39.000
"""


def bill(month, cold="45.50", hot="210.00", data=DATA):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "readings.csv")
        if data is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        return subprocess.run(
            [sys.executable, APP, "bill", path, "--month", month, "--cold-rate", cold, "--hot-rate", hot],
            capture_output=True, text=True, encoding="utf-8", timeout=60)


class MeterBillTests(unittest.TestCase):
    def test_march(self):
        p = bill("2026-03")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip().splitlines(), [
            "12\t5.500\t2.250\t722.75",
            "3\t2.345\t1.001\t316.91",
            "7\t2.000\t1.000\t301.00",
            "ИТОГО\t1340.66",
        ])

    def test_half_up_rounding(self):
        data = "apartment,date,cold,hot\n1,2026-01-01,0,0\n1,2026-02-01,0.001,0\n"
        p = bill("2026-02", cold="5", hot="1", data=data)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip().splitlines(), ["1\t0.001\t0.000\t0.01", "ИТОГО\t0.01"])

    def test_decimal_not_float(self):
        data = "apartment,date,cold,hot\n1,2026-01-01,0,0\n1,2026-02-01,1.115,0\n"
        p = bill("2026-02", cold="1", hot="1", data=data)
        self.assertEqual(p.stdout.strip().splitlines()[0], "1\t1.115\t0.000\t1.12")

    def test_no_apartments(self):
        p = bill("2025-12")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "ИТОГО\t0.00")

    def test_needs_previous_month(self):
        p = bill("2026-02")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip().splitlines(), ["ИТОГО\t0.00"])

    def test_decreasing_reading(self):
        data = DATA + "3,2026-03-30,12.000,7.000\n"
        p = bill("2026-03", data=data)
        self.assertEqual(p.returncode, 2)
        self.assertEqual(p.stdout.strip(), "")
        self.assertIn("ошибка", p.stderr.lower())
        self.assertIn("3", p.stderr)

    def test_bad_row_reports_line(self):
        data = "apartment,date,cold,hot\n1,2026-01-01,0,0\n1,2026-13-01,1,1\n"
        p = bill("2026-02", data=data)
        self.assertEqual(p.returncode, 2)
        self.assertIn("3", p.stderr)

    def test_missing_field(self):
        data = "apartment,date,cold,hot\n1,2026-01-01,0\n"
        p = bill("2026-02", data=data)
        self.assertEqual(p.returncode, 2)
        self.assertIn("2", p.stderr)

    def test_missing_file(self):
        p = bill("2026-03", data=None)
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
