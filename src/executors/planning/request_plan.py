from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, cast

from src.domain.graphql.request import (
    DataOrigin,
    GraphQLQueryRequest,
    LogicalFilter,
    MetricRequest,
    TimePeriod,
)
from src.domain.graphql.request import DateFilter as GQLDateFilter
from src.domain.graphql.request import LogicalFilter as GQLLogicalFilter
from src.domain.graphql.ssot_enums import GroupByType
from src.executors.planning.query_compiler import Dimension
from src.util import env as env_util


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


_DEFAULT_PROVIDER_GROUP_IDS = _parse_int_csv(env_util.get_env("EXECUTOR_DEFAULT_PROVIDER_GROUP_IDS", default="1") or "1")
_DEFAULT_PROVIDER_IDS = _parse_int_csv(env_util.get_env("EXECUTOR_DEFAULT_PROVIDER_IDS", default="") or "")
_INCLUDE_GENERAL_STATS = env_util.env_flag("EXECUTOR_INCLUDE_GENERAL_STATS", default=True)


def _build_data_origin(override: Optional[DataOrigin] = None) -> DataOrigin:
    if override is not None:
        return override

    provider_group_ids = list(_DEFAULT_PROVIDER_GROUP_IDS)
    provider_ids = list(_DEFAULT_PROVIDER_IDS)
    if not provider_group_ids and not provider_ids:
        provider_group_ids = [1]

    kwargs: dict[str, Any] = {}
    if provider_group_ids:
        kwargs["providerGroupId"] = provider_group_ids
    if provider_ids:
        kwargs["providerId"] = provider_ids
    return DataOrigin(**kwargs)


@dataclass
class RequestSpec:
    req: GraphQLQueryRequest
    label_parts: List[str]
    include_metric_alias: bool
    group_by_field: Optional[str]
    add_time_period_labels: bool
    scope_label: Optional[str] = None
    batched_time_periods: List[TimePeriod] = field(default_factory=lambda: cast(List[TimePeriod], []))


def _collect_date_bounds(
    filter_obj: Optional[Any],
) -> tuple[Optional[str], Optional[str]]:
    if filter_obj is None:
        return None, None

    min_start: Optional[str] = None
    max_end: Optional[str] = None

    def visit(node: Any) -> None:
        nonlocal min_start, max_end
        if isinstance(node, GQLLogicalFilter):
            for child in node.children:
                visit(child)
            return
        if isinstance(node, GQLDateFilter) and node.property == "DISCHARGE_DATE":
            op = node.operator.value
            val = node.value
            if op in ("GE", "GT"):
                if min_start is None or val < min_start:
                    min_start = val
            if op in ("LE", "LT"):
                if max_end is None or val > max_end:
                    max_end = val

    visit(filter_obj)
    return min_start, max_end


def _build_case_filter(chart_filter: Optional[Any], filter_dims: List[Dimension], combo: tuple[Any, ...]) -> tuple[Optional[Any], List[str]]:
    combo_filters: List[Any] = []
    label_parts: List[str] = []

    for dim, cat in zip(filter_dims, combo):
        gql_filter = dim.filter_for(cat)
        if gql_filter is not None:
            combo_filters.append(gql_filter)
        label_parts.append(dim.label_for(cat))

    if len(combo_filters) == 0:
        return chart_filter, label_parts

    if len(combo_filters) == 1 and chart_filter is None:
        return combo_filters[0], label_parts

    merged_children: List[Any] = []
    if chart_filter is not None:
        merged_children.append(chart_filter)
    merged_children.extend(combo_filters)
    return LogicalFilter(operator="AND", children=merged_children), label_parts  # type: ignore[arg-type]


def build_primary_request_specs(
    metric_requests: List[MetricRequest],
    metric_data_origins: Optional[Sequence[Optional[DataOrigin]]],
    chart_filter: Optional[Any],
    filter_dims: List[Dimension],
    combos_list: List[tuple[Any, ...]],
    batched_time_enabled: bool,
    batched_time_periods: List[TimePeriod],
    include_metric_alias: bool,
    group_by_field: Optional[str],
    metric_scope_labels: Optional[Sequence[Optional[str]]] = None,
    data_origin: Optional[DataOrigin] = None,
) -> List[RequestSpec]:
    specs: List[RequestSpec] = []
    per_metric_data_origin = any(origin is not None for origin in (metric_data_origins or []))

    for combo in combos_list:
        case_filter, label_parts = _build_case_filter(chart_filter, filter_dims, combo)

        req_time_period: TimePeriod | List[TimePeriod]
        if batched_time_enabled and batched_time_periods:
            req_time_period = batched_time_periods
        else:
            start_bound, end_bound = _collect_date_bounds(case_filter)
            req_time_period = TimePeriod(startDate=start_bound, endDate=end_bound)

        if per_metric_data_origin:
            for idx, metric_request in enumerate(metric_requests):
                metric_origin = (metric_data_origins[idx] if metric_data_origins and idx < len(metric_data_origins) else None) or data_origin
                scope_label = metric_scope_labels[idx] if metric_scope_labels and idx < len(metric_scope_labels) else None
                effective_scope_label = scope_label.strip() if isinstance(scope_label, str) and scope_label.strip() else None

                req = GraphQLQueryRequest(
                    metrics=[metric_request],
                    timePeriod=req_time_period,
                    dataOrigin=_build_data_origin(metric_origin),
                    includeGeneralStats=_INCLUDE_GENERAL_STATS,
                    caseFilter=case_filter,
                    groupBy=(GroupByType(group_by_field) if group_by_field else None),
                )

                specs.append(
                    RequestSpec(
                        req=req,
                        label_parts=label_parts,
                        include_metric_alias=include_metric_alias,
                        group_by_field=group_by_field,
                        add_time_period_labels=batched_time_enabled,
                        scope_label=effective_scope_label,
                        batched_time_periods=batched_time_periods,
                    )
                )
        else:
            req = GraphQLQueryRequest(
                metrics=metric_requests,
                timePeriod=req_time_period,
                dataOrigin=_build_data_origin(data_origin),
                includeGeneralStats=_INCLUDE_GENERAL_STATS,
                caseFilter=case_filter,
                groupBy=(GroupByType(group_by_field) if group_by_field else None),
            )

            specs.append(
                RequestSpec(
                    req=req,
                    label_parts=label_parts,
                    include_metric_alias=include_metric_alias,
                    group_by_field=group_by_field,
                    add_time_period_labels=batched_time_enabled,
                    batched_time_periods=batched_time_periods,
                )
            )

    return specs
