import datetime
import re

_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

_WORDS = re.compile(r"^(\d{1,2})\s+([а-яё]+)\s+(\d{4})(?:\s*(?:г\.|года))?$")
_DOTS = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def parse_ru_date(s: str) -> datetime.date:
    if not isinstance(s, str):
        raise ValueError("expected a string")
    text = " ".join(s.strip().lower().split())
    m = _WORDS.match(text)
    if m:
        month = _MONTHS.get(m.group(2).replace("ё", "е"))
        if month is None:
            raise ValueError(f"unknown month: {m.group(2)}")
        return datetime.date(int(m.group(3)), month, int(m.group(1)))
    m = _DOTS.match(text)
    if m:
        return datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = _ISO.match(text)
    if m:
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    raise ValueError(f"unsupported date format: {s!r}")
