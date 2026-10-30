"""
Dates from the descriptions of the digitized indexes (typoscripts, OBP NT).

The descriptions are clean typescript, e.g.
    "Copie missive ... in dato 28 Januarij 1608."
    "Dagregister ... sedert den 10 Januarij tot den 15 October 1739."
    "Drie circulaire brieven ... op 8, 24 November en 18 December 1738."
    "Copie register ... zedert primo Januarij tot ultimo September 1791."
    "Missive ... in dato 26 Julij 1669 met een apendix van 28 ditto."
`title_dates` returns every date of a description as a (first day, last day)
interval: a day, a month ("Maart 1783") or a year ("anno 1738"). Days and
months wait for the next year ("13, 16 October, 11 November 1734"), or take
the year of the date before them when none follows; a day or month without
any year is dropped. Parentheticals about receipt or shipping
("(ontfangen den 15 Julij 1713 per de fluijt ...)", "(den 11 Augustus 1732
per 't schip Jacoba)") are not the document's dates: `receipt_dates` returns
them; a document without other dates was written before. These are the
conventions of the dates of the enriched index (begin_of_begin ... end_of_end,
step 7), which gives such documents only an end.
"""

import calendar
import datetime
import re

from segmentation.header_dates import _month_of as header_month

YEAR_MIN, YEAR_MAX = 1595, 1815

MONTH_RE = re.compile(
    r"(?:(januar\w*|ianuar\w*|janvar\w*|janry)|(febr\w*|febrero|hebrero)|(maart|maert|maerd|meert|mart|martij|martius|martio|maertio)|(april\w*|aprill|aprijl)|"
    r"(mei|meij|maij|maj|may|meije)|(junij|juni|junius|junio|junijo|juny|iunij)|(julij|juli|julius|julio|julijo|july|iulij)|"
    r"(august\w*|augusti\w*)|(septemb\w*|sept|7ber|7bre)|(octob\w*|oktob\w*|8ber|8bre)|"
    r"(novemb\w*|9ber|9bre)|(decemb\w*|desemb\w*|decbr|xber|10ber|xbre))\.?$",
    re.IGNORECASE,
)
# abbreviations that are also names or words ("Jan", "Mart"): only next to a day or a year
SHORT_MONTH_RE = re.compile(r"(jan|feb|mrt|apr|jun|jul|aug|sep|oct|okt|nov|dec)\.?$", re.IGNORECASE)
SHORT_MONTHS = {"jan": 1, "feb": 2, "mrt": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "okt": 10, "nov": 11, "dec": 12}
TOKEN_RE = re.compile(r"\d+(?:e|en|ste|sten|de|den)?\b|[A-Za-zÀ-ÿ]+\.?|[,&\[\]/-]")
# words that may stand between days, months and their year
DAY_GLUE = {",", "en", "tot", "met", "&", "a", "à", "den", "de", "t", "m", "/", "-", "of", "als", "ook", "op"}
YEAR_GLUE = {",", "anno", "ao", "a", "des", "der", "jaars", "jaers", "jaar", "jaer", "deses", "dezes", "loopende", "lopende", "[", "]"}
# a bare year counts only after one of these (quantities like "1700 ponden" do not); in a description
# without other dates also after a comma or at its end ("Amonitie van oorloge op Amboijna, 1620.")
YEAR_CUE = {"anno", "ao", "a", "jaar", "jaer", "jaare", "jaere", "jare", "jaars", "jaers", "jaren", "jaaren", "jaeren",
            "gedurende", "geduurende", "tegen", "tegens", "in", "van", "tot", "en", "sedert",
            "zedert", "seedert", "zeedert", "sint", "sints", "sinds", "over", "na", "voor", "uit", "uijt", "uyt", "dato", "datis", "data",
            "gedateerd", "gedateert", "gedagtekend", "gedagteekend", "gedagteekent", "ontfangen", "ontvangen", "onfangen",
            "[", "-", "/", "t", "m", "den", "het", "'t", "ultimo", "primo", "medio"}
ORDINALS = {"eersten": 1, "eerste": 1, "tweeden": 2, "tweede": 2, "derden": 3, "derde": 3, "vierden": 4, "vierde": 4,
            "vijfden": 5, "vijfde": 5, "sesden": 6, "zesden": 6, "sevenden": 7, "zevenden": 7, "achsten": 8, "agsten": 8,
            "achtsten": 8, "negenden": 9, "tienden": 10, "laatsten": "ult", "laetsten": "ult", "laatste": "ult", "laetste": "ult"}
# ... nor before a unit ("van 1700 ponden")
UNITS = {"ponden", "pond", "lb", "lbs", "stux", "stuks", "stucks", "stuk", "man", "rds", "rijksdaalders", "rijxdaelders", "guldens",
         "picols", "pikols", "peculs", "balen", "lasten", "coppen", "koppen", "pees", "packen", "pakken", "kisten", "vaten", "ellen",
         "realen", "ropijen", "pagoden", "catties", "kannen", "leggers", "zakken", "sacken", "persoonen", "personen", "zielen"}
RECEIPT_RE = re.compile(r"\((?=[^()]*(?:ontf|ontv|onfang|ontang|per\s|verzonden|versonden|gesonden|gezonden|ingekomen|aangekomen))[^()]*\)",
                        re.IGNORECASE)


def _month(tok: str, prev: str | None, nxt: str | None) -> int | None:
    if (m := MONTH_RE.match(tok)):
        return next(i for i, g in enumerate(m.groups(), 1) if g)
    if (m := SHORT_MONTH_RE.match(tok)) and ((prev or "").isdigit() or _year(nxt or "")):
        return SHORT_MONTHS[m.group(1).lower()]
    if len(tok) >= 5 and _day(prev or "") and _year(nxt or ""):  # "5 Decemher 1715", "11 Vanuarij 1706"
        return header_month(tok)
    return None


def _day(tok: str) -> int | None:
    m = re.fullmatch(r"(\d{1,2})(?:e|en|ste|sten|de|den)?", tok)
    return int(m.group(1)) if m and 1 <= int(m.group(1)) <= 31 else None


def _year(tok: str) -> int | None:
    return int(tok) if re.fullmatch(r"\d{4}", tok) and YEAR_MIN <= int(tok) <= YEAR_MAX else None


def _interval(year: int, month: int | None, day: int | str | None) -> tuple[datetime.date, datetime.date] | None:
    if month is None:
        return datetime.date(year, 1, 1), datetime.date(year, 12, 31)
    last = calendar.monthrange(year, month)[1]
    if day is None:
        return datetime.date(year, month, 1), datetime.date(year, month, last)
    d = last if day == "ult" else (1 if day == "primo" else day)
    if d > last:
        return None  # 31 September: a typo
    return datetime.date(year, month, d), datetime.date(year, month, d)


def title_dates(text: str | None) -> list[tuple[datetime.date, datetime.date]]:
    """Every date of a description outside its receipt parentheticals, in text order, as (first day, last day)."""
    if not isinstance(text, str):
        return []
    body = RECEIPT_RE.sub(" ", text)
    return _dates(body) or _dates(body, loose=True)


def receipt_dates(text: str | None) -> list[tuple[datetime.date, datetime.date]]:
    """The dates in the receipt and shipping parentheticals of a description."""
    if not isinstance(text, str):
        return []
    return _dates(" ".join(RECEIPT_RE.findall(text)))


def yearless_dates(text: str | None) -> list[tuple[int, object]]:
    """(month, day) of the dates of a description that has no year at all ("dato 14 Junij"),
    outside its receipt parentheticals; day is None, a number, "ult" or "primo"."""
    if not isinstance(text, str):
        return []
    yearless: list[tuple[int, object]] = []
    return yearless if not title_dates(text) and not _dates(RECEIPT_RE.sub(" ", text), yearless) else []


def date_in_year(year: int, month: int, day) -> tuple[datetime.date, datetime.date] | None:
    """A yearless date (from yearless_dates) in the given year."""
    return _interval(year, month, day)


def _dates(text: str, yearless: list | None = None, loose: bool = False) -> list[tuple[datetime.date, datetime.date]]:
    toks = TOKEN_RE.findall(text)
    out: list[tuple[datetime.date, datetime.date]] = []
    days: list = []  # days (or "ult"/"primo") waiting for their month
    waiting: list[tuple[int, object]] = []  # (month, day) waiting for their year
    last_month = None  # for "28 ditto"
    idle = 0  # words since the last date token
    for i, tok in enumerate(toks):
        low = tok.lower().rstrip(".")
        prev = toks[i - 1] if i else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if (y := _year(tok)) is not None:
            if waiting:
                out += [iv for m, d in waiting if (iv := _interval(y, m, d))]
                waiting, days, idle = [], [], 0
            elif (prev is not None and prev.lower().rstrip(".") in YEAR_CUE or nxt in ("en", "tot", "-", "/")
                  or loose and (prev == "," or nxt in (None, "["))) and (nxt or "").lower() not in UNITS:
                out.append(_interval(y, None, None))
            continue
        if (m := _month(tok, prev, nxt)) is not None:
            waiting += [(m, d) for d in days] or [(m, None)]
            days, last_month, idle = [], m, 0
            continue
        if low in ("ditto", "dito") and days and last_month:
            if out:
                ref = out[-1][0]
                out += [iv for d in days if isinstance(d, int) and (iv := _interval(ref.year, ref.month, d))]
            else:  # "sedert primo April tot ultimo dito", no year yet
                waiting += [(last_month, d) for d in days]
            days = []
            continue
        if (d := _day(tok)) is not None:
            days.append(d)
            continue
        if low in ORDINALS:
            days.append(ORDINALS[low])
            continue
        if low.startswith(("ult", "uld")):
            days.append("ult")
            continue
        if low.startswith("prim"):
            days.append("primo")
            continue
        if low in DAY_GLUE or low in YEAR_GLUE:
            continue
        days = []  # other words: a number before them was no day
        idle += 1
        if idle > 6 and waiting:  # no year follows
            out += _year_before(waiting, out)
            if not out and yearless is not None:
                yearless += waiting
            waiting = []
    out += _year_before(waiting, out)
    if not out and yearless is not None:
        yearless += waiting
    return out


def _year_before(waiting, out):
    """Days and months without a year of their own take the year of the date before them
    ("op den 6 Februarij 1731 ..., van twee in datis 17 en 26 Januarij")."""
    if not out:
        return []
    return [iv for m, d in waiting if (iv := _interval(out[-1][0].year, m, d))]


def date_range(ds: list[tuple[datetime.date, datetime.date]]) -> tuple[datetime.date, datetime.date, datetime.date, datetime.date] | None:
    """(earliest begin, latest begin, earliest end, latest end) of dates from
    title_dates or nt_dates: the first and the last date, each from its first to
    its last day."""
    if not ds:
        return None
    first, last = min(ds), max(ds, key=lambda d: (d[1], d[0]))
    return first[0], first[1], last[0], last[1]


def nt_dates(jaar: str | None) -> list[tuple[datetime.date, datetime.date]]:
    """The dates of the JAAR column of OBP NT_gecorrigeerd ("20-10-1736 | 29-10-1736",
    "??-03-1620", "03-1620", "1638"), which also has receipt dates."""
    out = []
    for part in str(jaar or "").split("|"):
        m = re.fullmatch(r"\s*(?:(\d\d|\?\?)-)?(?:(\d\d)-)?(\d{4})\s*", part)
        if not m or not YEAR_MIN <= int(m.group(3)) <= YEAR_MAX:
            continue
        day, month = (m.group(1), m.group(2)) if m.group(2) else (None, m.group(1))
        month = int(month) if month and month.isdigit() and 1 <= int(month) <= 12 else None
        day = int(day) if day and day.isdigit() and month else None
        if (iv := _interval(int(m.group(3)), month, day)):
            out.append(iv)
    return out


# Year of a yearless date ("dato 14 Junij"): the typoscript's year (YEAR (EARLIEST))
# or the year before. With a receipt date, the typoscript year unless the date would
# fall after receipt; else the year before when it fits the dated neighbouring entries
# better by this many days. Hiding the years of dated descriptions (step 23): right for
# 99.3% with a receipt date, 92.1% without (the typoscript year alone: 83.2%; the true
# year is one of the two for 94.9%).
YEAR_MARGIN_DAYS = 270


def yearless_range(items: list[tuple[int, object]], year: int) -> tuple[datetime.date, ...] | None:
    """(earliest begin, latest begin, earliest end, latest end) of yearless dates in a
    year; an end before the begin is in the next year ("12 November tot 7 Januarij")."""
    first, last = _interval(year, *items[0]), _interval(year, *items[-1])
    if first is None or last is None:
        return None
    if last[1] < first[0]:
        last = _interval(year + 1, *items[-1]) or last
    return first[0], first[1], last[0], last[1]


def guess_year(items, typoscript_year: int, receipt=None, neighbours=()) -> int:
    """The year of yearless dates: typoscript_year or the one before (see YEAR_MARGIN_DAYS).
    receipt: the (first day, last day) of the receipt date; neighbours: those of the first
    dates of the nearest dated entries before and after it."""
    y = typoscript_year
    if receipt is not None:
        r = yearless_range(items, y)
        return y - 1 if r is not None and r[0] > receipt[1] else y

    def cost(year):
        r = yearless_range(items, year)
        return float("inf") if r is None else sum(abs((r[0] - a[0]).days) for a in neighbours)

    return y - 1 if neighbours and cost(y - 1) + YEAR_MARGIN_DAYS < cost(y) else y
