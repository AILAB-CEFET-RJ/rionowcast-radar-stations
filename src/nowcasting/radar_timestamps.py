"""Timestamp conventions for raw Radar Sumaré image filenames."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def local_filename_timestamp_to_utc(timestamp: datetime, source_timezone: str) -> datetime:
    """Convert a naive filename timestamp in ``source_timezone`` to naive UTC.

    The historical archive stores wall-clock timestamps in filenames.  Using an
    IANA timezone instead of a fixed offset preserves Brazil's historical DST
    transitions, which affected part of the 2012--2024 archive.
    """
    if timestamp.tzinfo is not None:
        raise ValueError("O timestamp de arquivo deve ser naive/local.")
    return timestamp.replace(tzinfo=ZoneInfo(source_timezone)).astimezone(timezone.utc).replace(tzinfo=None)
