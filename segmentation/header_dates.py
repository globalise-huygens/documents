"""
Parse dates from transcribed running headers, and compare them noise-aware.

Running headers typically look like
    "Van Bengale onder dato 14:' Febr: 1743."
    "Van Iavas Oost Cust den 15„e 9ber 1714."
    "A„o 1693, Junij banda int Cast:s nass=r"
    "141. van Cabo de goede Hoop ult:o meij 1720."
with HTR noise in separators, month spellings ("Navember", "IJunij", "desemb")
and digits. The parser works on tokens: it finds a month word (exact prefix,
the 7ber/8ber/9ber/Xber abbreviations, or a fuzzy match), then a year within a
few tokens after (or before) it, and a day number or ultimo/primo right before
it.
"""

import calendar
import datetime
import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

YEAR_MIN, YEAR_MAX = 1595, 1815

# Canonical month spellings (Dutch, Latin, French, Portuguese variants seen in
# the headers); fuzzy matching runs against these.
MONTH_WORDS = {
    1: ["januarij", "januari", "january", "janvier", "januario", "ianuarij"],
    2: ["februarij", "februari", "february", "fevrier", "februario"],
    3: ["maart", "martij", "martio", "mars", "marco"],
    4: ["april", "aprilis", "avril", "abril"],
    5: ["meij", "maij", "mei", "maio", "maius", "mai"],
    6: ["junij", "juni", "junio", "junius", "juin", "junho"],
    7: ["julij", "juli", "julio", "julius", "juillet", "julho"],
    8: ["augustus", "augusto", "august", "aoust", "agosto"],
    9: ["september", "septembris", "septembre", "setembro"],
    10: ["october", "october", "octobris", "octobre", "outubro"],
    11: ["november", "novembris", "novembre", "novembro"],
    12: ["december", "decembris", "decembre", "dezembro"],
}
# Abbreviation prefixes that are unambiguous (checked before fuzzy matching)
MONTH_PREFIXES = [
    (1, ("jan", "ian")), (2, ("feb", "febr")), (3, ("maa", "mart", "mrt")),
    (4, ("apr", "abr")), (5, ("mei", "maij", "meij", "may")), (6, ("jun", "iun")),
    (7, ("jul", "iul")), (8, ("aug",)), (9, ("sep", "7b")), (10, ("oct", "okt", "8b")),
    (11, ("nov", "nav", "9b")), (12, ("dec", "desem", "dezem", "xb", "10b")),
]
_FUZZY_CHOICES = {w: m for m, ws in MONTH_WORDS.items() for w in ws}
FUZZY_MIN_SCORE = 80

TOKEN_RE = re.compile(r"[a-z]+|\d+", re.IGNORECASE)
NUMBER_MONTH_RE = re.compile(r"^(7|8|9|10|x)b(e?r\w*|re)?$", re.IGNORECASE)


@dataclass(frozen=True)
class HeaderDate:
    date: datetime.date  # first day of the month/year when less precise
    precision: str  # "day" | "month" | "year"
    day_text: str | None = None  # the day as transcribed, for noise-aware comparison


def _month_of(token: str) -> int | None:
    t = token.lower()
    if NUMBER_MONTH_RE.match(t):
        return {"7": 9, "8": 10, "9": 11, "10": 12, "x": 12}[re.match(r"(10|7|8|9|x)", t)[1]]
    if not t.isalpha() or len(t) < 3:
        return None
    # "ij"/"y" confusion and a leading OCR "i"/"l" ("IJunij", "lJulij")
    candidates = [t, t[1:]] if t[0] in "il" and len(t) > 4 else [t]
    for c in candidates:
        for month, prefixes in MONTH_PREFIXES:
            if any(c.startswith(p) for p in prefixes if len(p) >= 3):
                return month
    if len(t) >= 5:
        hit = process.extractOne(t, _FUZZY_CHOICES.keys(), scorer=fuzz.ratio)
        if hit and hit[1] >= FUZZY_MIN_SCORE:
            return _FUZZY_CHOICES[hit[0]]
    return None


def _tokens(header: str) -> list[str]:
    text = header.replace("7ber", " 7ber ").replace("8ber", " 8ber ")
    text = text.replace("9ber", " 9ber ").replace("10ber", " 10ber ")
    toks = TOKEN_RE.findall(text)
    # re-glue "7" + "ber"/"bre" split by punctuation ("7:ber")
    out: list[str] = []
    for t in toks:
        if out and t.lower() in ("ber", "bre", "br", "bris") and out[-1] in ("7", "8", "9", "10"):
            out[-1] = out[-1] + "ber"
        else:
            out.append(t)
    return out


def _year(tok: str) -> int | None:
    if tok.isdigit() and len(tok) == 4 and YEAR_MIN <= int(tok) <= YEAR_MAX:
        return int(tok)
    return None


def parse_header_date(header: str | None) -> HeaderDate | None:
    """Return the most precise date found in a header string, or None."""
    if not header:
        return None
    toks = _tokens(header)
    best: HeaderDate | None = None
    rank = {"day": 3, "month": 2, "year": 1}
    for i, tok in enumerate(toks):
        month = _month_of(tok)
        if month is None:
            continue
        # year: first 4-digit year within 8 tokens after (dates can run over
        # two header lines), else within 6 before ("A:o 1693, Junij")
        year = next((y for t in toks[i + 1 : i + 9] if (y := _year(t))), None)
        if year is None:
            year = next((y for t in reversed(toks[max(0, i - 6) : i]) if (y := _year(t))), None)
        if year is None:
            continue
        day, day_text = None, None
        for t in reversed(toks[max(0, i - 3) : i]):
            tl = t.lower()
            if tl.startswith(("ult", "uld", "laatst")):
                day, day_text = calendar.monthrange(year, month)[1], "ult"
                break
            if tl.startswith("prim"):
                day, day_text = 1, "primo"
                break
            if t.isdigit() and len(t) <= 2 and 1 <= int(t) <= 31:
                day, day_text = int(t), t
                break
            if t.isdigit():
                break
        try:
            if day is not None:
                cand = HeaderDate(datetime.date(year, month, day), "day", day_text)
            else:
                cand = HeaderDate(datetime.date(year, month, 1), "month")
        except ValueError:  # e.g. 31 February from a misread day
            cand = HeaderDate(datetime.date(year, month, 1), "month")
        if best is None or rank[cand.precision] > rank[best.precision]:
            best = cand
        if best.precision == "day":
            break
    if best is None:
        year = next((y for t in toks if (y := _year(t))), None)
        if year is not None:
            best = HeaderDate(datetime.date(year, 1, 1), "year")
    return best


def _one_char_apart(a: str, b: str) -> bool:
    """True when two short digit strings differ by one edit (HTR misreads)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:k] + long_[k + 1 :] == short for k in range(len(long_)))


def compare_dates(a: HeaderDate | None, b: HeaderDate | None) -> str:
    """
    Compare two header dates, allowing for transcription noise:
      'unknown'   – either is missing
      'same'      – equal at the shared precision
      'uncertain' – same month and year, days one character apart (e.g. 14 vs 11)
      'changed'   – clearly different
    """
    if a is None or b is None:
        return "unknown"
    if a.date.year != b.date.year:
        return "changed"
    if "year" in (a.precision, b.precision):
        return "same"
    if a.date.month != b.date.month:
        return "changed"
    if "month" in (a.precision, b.precision) or a.date.day == b.date.day:
        return "same"
    if a.day_text and b.day_text and a.day_text.isdigit() and b.day_text.isdigit():
        if _one_char_apart(a.day_text, b.day_text):
            return "uncertain"
    return "changed"
