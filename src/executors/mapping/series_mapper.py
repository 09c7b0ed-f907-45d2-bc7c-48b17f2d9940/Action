from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.graphql.request import TimePeriod
from src.shared.ssot_loader import (
    get_enum_option_label,
    get_metric_display_name,
    get_metric_metadata,
)

_METRIC_METADATA = get_metric_metadata()


def metric_label_from_alias(metric_alias: str) -> str:
    code = metric_alias
    if code.lower().startswith("metric_"):
        code = code[len("metric_") :]
    return get_metric_display_name(code.upper())


def _metric_code_from_alias(metric_alias: str) -> str:
    code = metric_alias
    if code.lower().startswith("metric_"):
        code = code[len("metric_") :]
    return code.upper()


def _metric_is_enum(metric_name: str) -> bool:
    metric_code = _metric_code_from_alias(metric_name)
    meta = _METRIC_METADATA.get(metric_code) or {}
    return str(meta.get("data_type") or "").strip().lower() == "enum"


def _enum_series_values(kpi1: Any, expected_count: int) -> List[float]:
    percents_any = getattr(kpi1, "percents", None)
    if isinstance(percents_any, list):
        percents_list = cast(List[Any], percents_any)
        values: List[float] = []
        for raw in percents_list[:expected_count]:
            if isinstance(raw, (int, float)):
                values.append(float(raw))
            else:
                values.append(0.0)
        if len(values) == expected_count:
            return values

    raw_counts = getattr(kpi1, "case_count", None)
    if isinstance(raw_counts, list):
        fallback: List[float] = []
        for raw in cast(List[Any], raw_counts)[:expected_count]:
            if isinstance(raw, (int, float)):
                fallback.append(float(raw))
            else:
                fallback.append(0.0)
        return fallback
    return []


def _enum_category_labels(metric_code: str, count: int) -> List[str]:
    meta = _METRIC_METADATA.get((metric_code or "").upper()) or {}
    labels_any = meta.get("labels")
    props_any = meta.get("properties")

    labels: List[str] = []
    if isinstance(labels_any, list):
        labels = [
            str(v)
            for v in cast(List[Any], labels_any)
            if isinstance(v, str) and v.strip()
        ]
    if len(labels) >= count:
        return labels[:count]

    if isinstance(props_any, list):
        props = [
            str(v)
            for v in cast(List[Any], props_any)
            if isinstance(v, str) and v.strip()
        ]
        mapped: List[str] = []
        for key in props[:count]:
            mapped.append(get_enum_option_label(metric_code, key) or key)
        if len(mapped) >= count:
            return mapped[:count]

    return [f"Category {idx + 1}" for idx in range(count)]


def period_to_label(tp: TimePeriod) -> str:
    start = tp.start_date
    end = tp.end_date
    if isinstance(start, str) and start:
        try:
            dt = datetime.fromisoformat(start)
            return dt.strftime("%Y-%m")
        except Exception:
            pass
    if isinstance(start, str) and isinstance(end, str) and start and end:
        return f"{start} to {end}"
    if isinstance(start, str) and start:
        return start
    if isinstance(end, str) and end:
        return end
    return "period"


def merge_series_by_name(series: List[ChartSeries]) -> List[ChartSeries]:
    merged: Dict[str, ChartSeries] = {}
    ordered_names: List[str] = []

    for item in series:
        existing = merged.get(item.name)
        if existing is None:
            merged[item.name] = ChartSeries(
                name=item.name, data=list(item.data), color=item.color, style=item.style
            )
            ordered_names.append(item.name)
        else:
            existing.data.extend(item.data)

    return [merged[name] for name in ordered_names]


def _origin_label_from_kpi_group(kpi_group: Any) -> Optional[str]:
    origin = getattr(kpi_group, "data_origin", None)
    if origin is None:
        return None

    provider_id = getattr(origin, "provider_id", None)
    if isinstance(provider_id, int):
        return f"Provider {provider_id}"

    provider_group_id = getattr(origin, "provider_group_id", None)
    if isinstance(provider_group_id, int):
        return f"Group {provider_group_id}"

    custom_group_name = getattr(origin, "custom_group_name", None)
    if isinstance(custom_group_name, str) and custom_group_name.strip():
        return custom_group_name.strip()

    return None


def _base_series_name_parts(
    metric_name: str,
    include_metric_alias: bool,
    label_parts: List[str],
    origin_label: Optional[str],
    scope_label: Optional[str],
) -> List[str]:
    name_parts: List[str] = []
    if include_metric_alias:
        name_parts.append(metric_label_from_alias(metric_name))
    name_parts.extend([part for part in label_parts if part])
    if origin_label:
        name_parts.append(origin_label)
    if scope_label and scope_label not in name_parts:
        name_parts.append(scope_label)
    if not name_parts:
        name_parts.append(metric_label_from_alias(metric_name))
    return name_parts


def map_metrics_payload_to_series(
    metrics_payload: Dict[str, Any],
    label_parts: List[str],
    include_metric_alias: bool,
    group_by_field: Optional[str],
    add_time_period_labels: bool,
    scope_label: Optional[str] = None,
    batched_time_periods: Optional[List[Any]] = None,  # ADD THIS
) -> List[ChartSeries]:
    series: List[ChartSeries] = []

    for metric_name, metric in metrics_payload.items():
        for kpi_index, kpi in enumerate(metric.kpi_group):
            if getattr(kpi, "kpi1", None) is None:
                continue

            server_label = kpi.grouped_by.group_item_name if kpi.grouped_by else None
            mapped_server_label = (
                get_enum_option_label(group_by_field, server_label)
                if (group_by_field and server_label)
                else None
            )
            origin_label = _origin_label_from_kpi_group(kpi)

            # Use aggregate-style mapping only for native backend groupBy output
            # or for explicit time-batched requests.
            # Label-only filter splits should fall through to the default d1 path.
            is_grouped_or_time = bool(group_by_field) or add_time_period_labels
            if is_grouped_or_time:
                x_value: str
                tp_start: Optional[str] = None
                tp_end: Optional[str] = None
                if kpi.time_period is not None:
                    tp_start = kpi.time_period.start_date
                    tp_end = kpi.time_period.end_date
                elif add_time_period_labels and batched_time_periods:
                    # kpi_index = list(metric.kpi_group).index(kpi)
                    if kpi_index < len(batched_time_periods):
                        tp = batched_time_periods[kpi_index]
                        tp_start = getattr(tp, "startDate", None) or getattr(
                            tp, "start_date", None
                        )
                        tp_end = getattr(tp, "endDate", None) or getattr(
                            tp, "end_date", None
                        )

                if add_time_period_labels and tp_start:
                    try:
                        dt = datetime.fromisoformat(str(tp_start))
                        x_value = dt.strftime("%Y-%m")
                    except Exception:
                        x_value = f"{tp_start} to {tp_end}" if tp_end else str(tp_start)
                elif add_time_period_labels and batched_time_periods:
                    # Dual-dimension fallback: if backend omits per-kpi timePeriod,
                    # cycle through requested periods to keep the x-axis temporal.
                    period_count = len(batched_time_periods)
                    if period_count > 0:
                        tp = batched_time_periods[kpi_index % period_count]
                        start_value = getattr(tp, "startDate", None) or getattr(
                            tp, "start_date", None
                        )
                        end_value = getattr(tp, "endDate", None) or getattr(
                            tp, "end_date", None
                        )
                        if start_value:
                            try:
                                dt = datetime.fromisoformat(str(start_value))
                                x_value = dt.strftime("%Y-%m")
                            except Exception:
                                x_value = (
                                    f"{start_value} to {end_value}"
                                    if end_value
                                    else str(start_value)
                                )
                        elif server_label:
                            x_value = mapped_server_label or server_label
                        elif label_parts:
                            x_value = label_parts[-1]
                        else:
                            x_value = "value"
                    elif server_label:
                        x_value = mapped_server_label or server_label
                    elif label_parts:
                        x_value = label_parts[-1]
                    else:
                        x_value = "value"
                elif server_label:
                    x_value = mapped_server_label or server_label
                elif label_parts:
                    x_value = label_parts[-1]
                else:
                    x_value = "value"

                raw_counts = getattr(kpi.kpi1, "case_count", None)
                raw_counts_list = (
                    cast(List[Any], raw_counts) if isinstance(raw_counts, list) else []
                )
                metric_is_enum = _metric_is_enum(metric_name)
                if (
                    add_time_period_labels
                    and metric_is_enum
                    and len(raw_counts_list) > 1
                ):
                    metric_code = _metric_code_from_alias(metric_name)
                    labels = _enum_category_labels(metric_code, len(raw_counts_list))
                    values = _enum_series_values(kpi.kpi1, len(raw_counts_list))
                    base_parts = _base_series_name_parts(
                        metric_name=metric_name,
                        include_metric_alias=include_metric_alias,
                        label_parts=label_parts,
                        origin_label=origin_label,
                        scope_label=scope_label,
                    )
                    if add_time_period_labels and server_label:
                        base_parts.append(mapped_server_label or server_label)
                    for category_label, raw_value in zip(labels, values):
                        category_parts = list(base_parts)
                        category_parts.append(category_label)
                        series.append(
                            ChartSeries(
                                name=" — ".join(category_parts),
                                data=[
                                    ChartPoint(
                                        x=x_value, y=float(raw_value), label=x_value
                                    )
                                ],
                            )
                        )
                    continue

                y_value: Optional[float] = None
                if isinstance(kpi.kpi1.median, (int, float)):
                    y_value = float(kpi.kpi1.median)
                elif isinstance(kpi.kpi1.mean, (int, float)):
                    y_value = float(kpi.kpi1.mean)
                elif metric_is_enum and isinstance(
                    getattr(kpi.kpi1, "percents", None), list
                ):
                    percents_list = cast(List[Any], getattr(kpi.kpi1, "percents", None))
                    if percents_list:
                        first_percent = percents_list[0]
                        if isinstance(first_percent, (int, float)):
                            y_value = float(first_percent)
                elif kpi.kpi1.case_count:
                    try:
                        y_value = float(kpi.kpi1.case_count[0])
                    except Exception:
                        y_value = None

                # Preserve explicit time buckets as missing points (y=None)
                # so frontend can render gaps instead of implied zeros.
                if y_value is None and not (add_time_period_labels and tp_start):
                    continue

                name_parts = _base_series_name_parts(
                    metric_name=metric_name,
                    include_metric_alias=include_metric_alias,
                    label_parts=label_parts,
                    origin_label=origin_label,
                    scope_label=scope_label,
                )
                if add_time_period_labels and server_label:
                    name_parts.append(mapped_server_label or server_label)
                series_name = " — ".join(name_parts)
                series.append(
                    ChartSeries(
                        name=series_name,
                        data=[ChartPoint(x=x_value, y=y_value, label=x_value)],
                    )
                )
                continue

            if not kpi.kpi1.d1:
                raw_counts = getattr(kpi.kpi1, "case_count", None)
                if isinstance(raw_counts, list):
                    raw_counts_list = cast(List[Any], raw_counts)
                else:
                    raw_counts_list = []

                metric_is_enum = _metric_is_enum(metric_name)
                if metric_is_enum and len(raw_counts_list) > 1:
                    metric_code = _metric_code_from_alias(metric_name)
                    labels = _enum_category_labels(metric_code, len(raw_counts_list))
                    values = _enum_series_values(kpi.kpi1, len(raw_counts_list))

                    parts: List[str] = []
                    if include_metric_alias:
                        parts.append(metric_label_from_alias(metric_name))
                    parts.extend([part for part in label_parts if part])
                    if origin_label:
                        parts.append(origin_label)
                    if scope_label and scope_label not in parts:
                        parts.append(scope_label)

                    series_name = (
                        " — ".join(parts)
                        if parts
                        else metric_label_from_alias(metric_name)
                    )
                    points: List[ChartPoint] = []
                    for x, y in zip(labels, values):
                        points.append(ChartPoint(x=x, y=float(y)))
                    if points:
                        series.append(ChartSeries(name=series_name, data=points))
                continue

            parts: List[str] = []
            if include_metric_alias:
                parts.append(metric_label_from_alias(metric_name))
            parts.extend([part for part in label_parts if part])
            if origin_label:
                parts.append(origin_label)
            if server_label:
                mapped = (
                    get_enum_option_label(group_by_field, server_label)
                    if group_by_field
                    else None
                )
                parts.append(mapped or server_label)
            if scope_label and scope_label not in parts:
                parts.append(scope_label)

            if add_time_period_labels and kpi.time_period is not None:
                start = kpi.time_period.start_date
                end = kpi.time_period.end_date
                if isinstance(start, str) and isinstance(end, str):
                    parts.append(f"{start} to {end}")
                elif isinstance(start, str):
                    parts.append(start)
                elif isinstance(end, str):
                    parts.append(end)

            series_name = (
                " — ".join(parts) if parts else metric_label_from_alias(metric_name)
            )
            series.append(
                ChartSeries(
                    name=series_name,
                    data=[
                        ChartPoint(x=x, y=y)
                        for x, y in zip(kpi.kpi1.d1.edges, kpi.kpi1.d1.case_count)
                    ],
                )
            )

    return series
