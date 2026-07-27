"""SSOT-backed metric defaults for distribution bucketing and axis labelling.

This module is the single source of truth for deriving distribution parameters
and histogram axis metadata from SSOT metric records.  It is called directly by
the metric-request factory so that orchestration never needs to inject axis/bucket
knowledge as callbacks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, cast

from src.domain.dto.charts.types import ChartAxis
from src.shared.ssot_loader import get_metric_display_name, get_metric_metadata

logger = logging.getLogger(__name__)

_METRIC_METADATA: Dict[str, Any] = get_metric_metadata()

_AXIS_LABEL_OVERRIDES: Dict[str, str] = {
    "DTN": "Door-to-Needle Time",
    "ONSET_TO_DOOR": "Onset-to-Door Time",
    "DOOR_TO_REPERFUSION": "Door-to-Reperfusion Time",
}

_AXIS_UNIT_OVERRIDES: Dict[str, str] = {
    "DTN": "minutes",
    "ONSET_TO_DOOR": "minutes",
    "DOOR_TO_REPERFUSION": "minutes",
}

_AXIS_ACRONYMS = {"NIHSS", "DTN", "IVT", "EVT", "TIA", "LVO", "ICH", "SAH", "CT", "MRI"}


def _mapping_to_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return cast(Dict[str, Any], value)


def _normalize_axis_display_label(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    def _word_case(word: str) -> str:
        token = word.strip()
        if not token:
            return token
        if token.upper() in _AXIS_ACRONYMS:
            return token.upper()
        if token.isupper() and len(token) <= 4:
            return token
        if token[:1].isdigit():
            return token
        return token[:1].upper() + token[1:].lower()

    out_words = []
    for word in text.split():
        if "-" in word:
            out_words.append("-".join(_word_case(p) for p in word.split("-")))
        else:
            out_words.append(_word_case(word))
    return " ".join(out_words)


def get_enum_labels(metric_code: str) -> Optional[list[str]]:
    """Return SSOT display labels for an Enum metric's categories, in property order.

    Used as a fallback when the backend's own `labels` field is null — some
    boolean-shaped Enum metrics (e.g. WAKEUP_STROKE) return a single caseCount
    with no labels array, rather than one entry per category.
    """
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))
    labels = meta.get("labels")
    if isinstance(labels, list) and labels:
        return [str(label) for label in cast(list, labels)]
    return None


def is_enum_metric(metric_code: str) -> bool:
    """Return True if the metric's SSOT data_type is Enum (categorical, not numeric).

    Enumeration-type metrics (e.g. SEX, HOSPITALIZED_IN) reject the backend's
    numeric kpi(kpiOptions/distribution) query shape; they must be requested
    via the bare kpi + labels shape instead (see MetricRequest.with_categorical).
    """
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))
    return str(meta.get("data_type") or "").strip().lower() == "enum"


def get_distribution_defaults(metric_code: str) -> tuple[int, int, int]:
    """Return (bins, min_value, max_value) from SSOT metadata.

    Falls back to safe per-metric ranges, then to a general default of (20, 0, 200).
    All parameters are derived from data; no runtime inference is performed.
    """
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))

    bins_any: Any = meta.get("distribution_default_buckets")
    numeric_block = _mapping_to_dict(meta.get("numeric"))
    bins = bins_any or numeric_block.get("default_buckets") or 20

    rmin: Any = meta.get("range_min")
    rmax: Any = meta.get("range_max")
    if rmin is None or rmax is None:
        rmin = rmin if rmin is not None else numeric_block.get("range_min")
        rmax = rmax if rmax is not None else numeric_block.get("range_max")

    if rmin is None or rmax is None:
        known_ranges: dict[str, tuple[int, int]] = {
            "AGE": (18, 95),
            "ADMISSION_NIHSS": (0, 42),
            "DTN": (0, 120),
        }
        if code in known_ranges:
            rmin, rmax = known_ranges[code]
        else:
            rmin = rmin if rmin is not None else 0
            rmax = rmax if rmax is not None else 200

    try:
        bins = int(bins)
    except Exception:
        logger.debug("[ssot_metric_defaults] Could not parse bin count for %s; using 20", code)
        bins = 20
    try:
        rmin = int(rmin)
        rmax = int(rmax)
    except Exception:
        logger.debug("[ssot_metric_defaults] Could not parse range for %s; using 0-200", code)
        rmin, rmax = 0, 200

    if rmin > rmax:
        rmin, rmax = rmax, rmin

    return bins, rmin, rmax


def get_histogram_axes(
    metric_code: str, x_min: int, x_max: int
) -> tuple[ChartAxis, ChartAxis]:
    """Return (x_axis, y_axis) for a histogram chart from SSOT metadata."""
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))

    display = _AXIS_LABEL_OVERRIDES.get(code) or _normalize_axis_display_label(
        get_metric_display_name(code)
    )

    unit_any: Any = meta.get("unit") or _mapping_to_dict(meta.get("numeric")).get("unit")
    unit: Optional[str] = cast(Optional[str], unit_any) or _AXIS_UNIT_OVERRIDES.get(code)

    x_label = f"{display} ({unit})" if unit else display
    return ChartAxis(label=x_label, min_value=x_min, max_value=x_max), ChartAxis(label="Cases")
