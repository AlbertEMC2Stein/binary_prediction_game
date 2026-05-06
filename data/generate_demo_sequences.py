#!/usr/bin/env python3
"""Generate demo 0-1 sequence YAML files.

The generated YAML files have the schema

    description: <short description>
    origin: system
    created_at: <date>
    sequence_length: <length>
    sequence: '<bit sequence>'

The first three sequences are purely deterministic. The last two fetch public
historical data:

* Open-Meteo Historical Weather API, hourly cloud_cover for Kaiserslautern.
* Stooq daily OHLCV CSV data for AAPL.US.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ORIGIN = "system"
CREATED_AT = datetime.now(timezone.utc).date().isoformat()

# Fixed, historically complete windows make the demo reproducible.
WEATHER_START = "2020-01-01"
WEATHER_END = "2025-12-31"  # inclusive
AAPL_FETCH_START = "2010-01-01"
AAPL_FETCH_END = "2025-12-31"  # inclusive

# Kaiserslautern, Germany, WGS84 coordinates.
KAISERSLAUTERN_LATITUDE = 49.4401
KAISERSLAUTERN_LONGITUDE = 7.7491
WEATHER_CLOUDY_THRESHOLD_PERCENT = 75.0

AAPL_STOOQ_SYMBOL = "aapl.us"
STOOQ_API_KEY = "AsFbTQMOB3EVj8JgnGltq4SrUHmoDZf7"

OUTPUT_DIR = Path("data/built-in-sequences")


@dataclass(frozen=True)
class SequenceRecord:
    """In-memory representation of one YAML sequence file."""

    filename: str
    description: str
    sequence: str

    @property
    def sequence_length(self) -> int:
        return len(self.sequence)


def main(argv: Sequence[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Generate YAML files containing pre-defined 0-1 sequences."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory in which the .yml files are written.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Generate only the first three deterministic files and skip data downloads.",
    )
    args = parser.parse_args(argv)

    if not args.output_dir.is_dir():
        print(
            f"Error: output directory {args.output_dir} does not exist", file=sys.stderr
        )
        return 1

    records = [
        alternating_bits_record(length=1000),
        lfsr_bits_record(length=1000),
        pi_even_odd_record(length=1000),
    ]

    if not args.offline_only:
        records.extend(
            [
                kaiserslautern_cloudiness_record(length=-1),
                aapl_close_direction_record(length=-1),
            ]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        validate_binary_sequence(
            record.sequence, expected_length=record.sequence_length
        )
        path = args.output_dir / record.filename
        write_sequence_yaml(path, record)
        print(f"wrote {path} ({record.sequence_length} bits)")

    return 0


def alternating_bits_record(length: int) -> SequenceRecord:
    sequence = "".join("0" if i % 2 == 0 else "1" for i in range(length))
    return SequenceRecord(
        filename=f"alternating_01_{length}.yml",
        description=(
            "Deterministic alternating 0/1 sequence starting with 0; "
            "expected next bit is the period-2 continuation."
        ),
        sequence=sequence,
    )


def lfsr_bits_record(length: int) -> SequenceRecord:
    sequence = generate_lfsr_bits(length=length)
    return SequenceRecord(
        filename=f"simple_lfsr_10bit_{length}.yml",
        description=(
            "10-bit Fibonacci LFSR seeded with all ones; feedback is "
            "state[-1] XOR state[-4], so the sequence is deterministic and linear."
        ),
        sequence=sequence,
    )


def generate_lfsr_bits(length: int) -> str:
    """Generate bits from a simple 10-bit Fibonacci LFSR.

    State convention: state[-1] is emitted. Then feedback = state[-1] XOR
    state[-4] is shifted into state[0]. With the all-ones seed this tap choice
    has period 1023, so 1000 bits are non-repeating within the file.
    """
    state = [1] * 10
    bits: list[str] = []

    for _ in range(length):
        output_bit = state[-1]
        feedback_bit = state[-1] ^ state[-4]
        bits.append(str(output_bit))
        state = [feedback_bit] + state[:-1]

    return "".join(bits)


def pi_even_odd_record(length: int) -> SequenceRecord:
    digits = pi_digits_without_decimal_point(length)
    sequence = "".join("0" if int(digit) % 2 == 0 else "1" for digit in digits)
    return SequenceRecord(
        filename=f"pi_decimal_digit_parity_{length}.yml",
        description=(
            f"First {length} digits of pi in decimal notation with the decimal point removed "
            f"(starts with 3); even digits mapped to 0 and odd digits mapped to 1."
        ),
        sequence=sequence,
    )


def pi_digits_without_decimal_point(count: int) -> str:
    """Return the first `count` digits of pi with the decimal point removed."""
    if count < 1:
        raise ValueError("count must be positive")

    guard_digits = 20
    precision = count + guard_digits
    getcontext().prec = precision

    pi_value = compute_pi_chudnovsky(decimal_digits=precision)
    decimal_places = count - 1
    text = f"{pi_value:.{decimal_places + 2}f}"
    integer_part, after_point = text.split(".", maxsplit=1)
    digits = integer_part + after_point[:decimal_places]

    if len(digits) != count or not digits.isdigit():
        raise RuntimeError("failed to compute enough decimal digits of pi")

    return digits


def compute_pi_chudnovsky(decimal_digits: int) -> Decimal:
    """Compute pi using the Chudnovsky series."""
    getcontext().prec = decimal_digits
    terms = decimal_digits // 14 + 2

    constant = Decimal(426880) * Decimal(10005).sqrt()
    summation = Decimal(13591409)

    m = 1
    l = 13591409
    x = 1
    k = 6

    for i in range(1, terms):
        # Integer recurrence for the Chudnovsky M term.
        m = (m * (k**3 - 16 * k)) // (i**3)
        l += 545140134
        x *= -262537412640768000
        summation += Decimal(m * l) / Decimal(x)
        k += 12

    pi_apx = constant / summation

    print(
        f"computed pi to {decimal_digits} decimal digits using {terms} Chudnovsky terms: {pi_apx}"
    )

    return pi_apx


def kaiserslautern_cloudiness_record(length: int) -> SequenceRecord:
    daily_cloud_cover = fetch_open_meteo_hourly_cloud_cover(
        latitude=KAISERSLAUTERN_LATITUDE,
        longitude=KAISERSLAUTERN_LONGITUDE,
        start_date=WEATHER_START,
        end_date=WEATHER_END,
        timezone_name="Europe/Berlin",
    )

    year_start = date.fromisoformat(WEATHER_START).year
    year_end = date.fromisoformat(WEATHER_END).year

    dates = sorted(daily_cloud_cover)
    if length != -1 and len(dates) < length:
        raise RuntimeError(
            f"not enough daily weather observations: got {len(dates)}, need at least {length} "
            f"for {WEATHER_START}..{WEATHER_END}"
        )

    missing_days = [day for day in dates if math.isnan(daily_cloud_cover[day])]
    if missing_days:
        raise RuntimeError(f"missing daily cloud-cover values: {missing_days[:10]}")

    sequence = "".join(
        "1" if daily_cloud_cover[day] >= WEATHER_CLOUDY_THRESHOLD_PERCENT else "0"
        for day in dates
    )

    actual_length = len(sequence)

    return SequenceRecord(
        filename=f"kaiserslautern_cloudy_open_meteo__{year_start}_{year_end}_{actual_length}.yml",
        description=(
            "Kaiserslautern daily cloudiness from Open-Meteo hourly cloud_cover, "
            f"{WEATHER_START} to {WEATHER_END}; bit=1 if daily mean cloud cover "
            f">= {WEATHER_CLOUDY_THRESHOLD_PERCENT:g}%, else 0; missing days=0."
        ),
        sequence=sequence,
    )


def fetch_open_meteo_hourly_cloud_cover(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone_name: str,
) -> dict[str, float]:
    """Fetch hourly cloud cover and aggregate it to daily means."""
    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "cloud_cover",
        "timezone": timezone_name,
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    payload = fetch_json(url)

    try:
        times: list[str] = payload["hourly"]["time"]
        values: list[float | None] = payload["hourly"]["cloud_cover"]
    except KeyError as exc:
        raise RuntimeError(
            f"Open-Meteo response is missing expected key: {exc}"
        ) from exc

    by_day: dict[str, list[float]] = defaultdict(list)
    for timestamp, value in zip(times, values, strict=True):
        if value is None:
            continue
        day = timestamp[:10]
        by_day[day].append(float(value))

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    expected_dates = [
        (start.toordinal() + offset)
        for offset in range(end.toordinal() - start.toordinal() + 1)
    ]

    result: dict[str, float] = {}
    for ordinal in expected_dates:
        day = date.fromordinal(ordinal).isoformat()
        hourly_values = by_day.get(day, [])
        if len(hourly_values) < 12:
            result[day] = math.nan
        else:
            result[day] = sum(hourly_values) / len(hourly_values)

    return result


def aapl_close_direction_record(length: int) -> SequenceRecord:
    rows = fetch_stooq_daily_prices(
        symbol=AAPL_STOOQ_SYMBOL,
        start_date=AAPL_FETCH_START,
        end_date=AAPL_FETCH_END,
    )

    start_year = date.fromisoformat(AAPL_FETCH_START).year
    end_year = date.fromisoformat(AAPL_FETCH_END).year

    if length != -1 and len(rows) < length + 1:
        raise RuntimeError(
            f"not enough daily price observations: got {len(rows)}, need at least {length} "
            f"for {AAPL_FETCH_START}..{AAPL_FETCH_END}"
        )

    bits_with_dates: list[tuple[str, str]] = []
    tie_count = 0
    missing_close_rows = 0

    # Work backwards so the sequence uses the latest available 'length' non-tie moves
    # ending no later than AAPL_FETCH_END, then reverse back to chronological order.
    for previous, current in zip(reversed(rows[:-1]), reversed(rows[1:]), strict=True):
        previous_date, previous_close = previous
        current_date, current_close = current

        if previous_close is None or current_close is None:
            missing_close_rows += 1
            print(
                f"skipping {current_date} due to missing close value; "
                f"total missing close rows so far: {missing_close_rows}"
            )
            continue
        if current_close == previous_close:
            tie_count += 1
            print(
                f"skipping {current_date} due to tie with previous close; "
                f"total tie rows so far: {tie_count}"
            )
            continue

        bit = "1" if current_close > previous_close else "0"
        bits_with_dates.append((current_date, bit))

        if len(bits_with_dates) == length:
            break

    if length != -1 and len(bits_with_dates) != length:
        raise RuntimeError(
            f"only found {len(bits_with_dates)} non-tie AAPL moves, need {length}"
        )

    bits_with_dates.reverse()
    first_move_date = bits_with_dates[0][0]
    last_move_date = bits_with_dates[-1][0]
    sequence = "".join(bit for _, bit in bits_with_dates)

    actual_length = len(sequence)

    return SequenceRecord(
        filename=f"aapl_close_direction_stooq_{start_year}_{end_year}_{actual_length}.yml",
        description=(
            "Apple AAPL.US daily closes from Stooq; chronological close-to-close "
            f"moves {first_move_date} to {last_move_date}; bit=1 if close is higher "
            "than previous trading day and 0 if lower; "
            f"equal-close rows skipped={tie_count}, missing close rows={missing_close_rows}."
        ),
        sequence=sequence,
    )


def fetch_stooq_daily_prices(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[tuple[str, float | None]]:
    """Fetch daily Stooq CSV rows as sorted (date, close) tuples."""
    params = {
        "s": symbol,
        "i": "d",
        "d1": start_date.replace("-", ""),
        "d2": end_date.replace("-", ""),
        "apikey": STOOQ_API_KEY,
    }
    url = "https://stooq.com/q/d/l/?" + urlencode(params)
    text = fetch_text(url)

    reader = csv.DictReader(text.splitlines())
    required_columns = {"Date", "Close"}
    if not required_columns.issubset(reader.fieldnames or set()):
        raise RuntimeError(
            f"Stooq CSV has unexpected columns: {reader.fieldnames}; response starts with {text[:80]!r}"
        )

    rows: list[tuple[str, float | None]] = []
    for row in reader:
        raw_date = row["Date"]
        raw_close = row["Close"]
        if not raw_date:
            continue
        close_value: float | None
        try:
            close_value = float(raw_close)
        except (TypeError, ValueError):
            close_value = None
        rows.append((raw_date, close_value))

    rows.sort(key=lambda item: item[0])
    return rows


def fetch_json(url: str) -> dict:
    text = fetch_text(url)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to decode JSON from {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def fetch_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "demo-sequence-generator/1.0 (+https://example.invalid)"
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset)
    except HTTPError as exc:
        raise RuntimeError(
            f"HTTP error while fetching {url}: {exc.code} {exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"network error while fetching {url}: {exc.reason}") from exc


def validate_binary_sequence(sequence: str, *, expected_length: int) -> None:
    if len(sequence) != expected_length:
        raise ValueError(f"expected length {expected_length}, got {len(sequence)}")
    invalid_chars = set(sequence) - {"0", "1"}
    if invalid_chars:
        raise ValueError(
            f"sequence contains non-binary characters: {sorted(invalid_chars)}"
        )


def write_sequence_yaml(path: Path, record: SequenceRecord) -> None:
    content = "\n".join(
        [
            f"description: {yaml_double_quote(record.description)}",
            f"origin: {ORIGIN}",
            f"created_at: {CREATED_AT}",
            f"sequence_length: {record.sequence_length}",
            f"sequence: '{record.sequence}'",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def yaml_double_quote(value: str) -> str:
    """Return a conservative YAML double-quoted scalar."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


if __name__ == "__main__":
    raise SystemExit(main())
