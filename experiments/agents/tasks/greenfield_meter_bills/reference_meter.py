"""Reference solution for greenfield_meter_bills. Used only to validate the hidden tests."""
import argparse
import csv
import re
import sys
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

NUM = re.compile(r"^\d+(\.\d{1,3})?$")


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


def read(path):
    try:
        f = open(path, encoding="utf-8", newline="")
    except OSError as e:
        fail(f"ошибка: не удалось открыть файл: {e}")
    rows = []
    with f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["apartment", "date", "cold", "hot"]:
            fail("ошибка: строка 1: неверный заголовок")
        for lineno, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                apt, d, cold, hot = row
                if not apt.strip() or not NUM.match(cold) or not NUM.match(hot):
                    raise ValueError
                rows.append((apt.strip(), date.fromisoformat(d), Decimal(cold), Decimal(hot)))
            except (ValueError, InvalidOperation):
                fail(f"ошибка: строка {lineno}: неверные данные")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="meter.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bill")
    b.add_argument("file")
    b.add_argument("--month", required=True)
    b.add_argument("--cold-rate", required=True, type=Decimal)
    b.add_argument("--hot-rate", required=True, type=Decimal)
    a = p.parse_args(argv)
    y, m = map(int, a.month.split("-"))
    start = date(y, m, 1)
    end = date(y + (m == 12), m % 12 + 1, 1)
    by_apt = {}
    for apt, d, cold, hot in read(a.file):
        by_apt.setdefault(apt, []).append((d, cold, hot))
    lines, total = [], Decimal("0")
    for apt in sorted(by_apt):
        rs = sorted(by_apt[apt])
        for prev, cur in zip(rs, rs[1:]):
            if cur[1] < prev[1] or cur[2] < prev[2]:
                fail(f"ошибка: квартира {apt}: показания уменьшились {cur[0]}")
        before = [r for r in rs if r[0] < start]
        inside = [r for r in rs if start <= r[0] < end]
        if not before or not inside:
            continue
        dc = inside[-1][1] - before[-1][1]
        dh = inside[-1][2] - before[-1][2]
        s = (dc * a.cold_rate + dh * a.hot_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
        total += s
        lines.append(f"{apt}\t{dc:.3f}\t{dh:.3f}\t{s:.2f}")
    lines.append(f"ИТОГО\t{total:.2f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
