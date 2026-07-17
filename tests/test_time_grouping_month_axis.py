from types import SimpleNamespace

from src.domain.langchain import schema as S
from src.executors.mapping.series_mapper import map_metrics_payload_to_series
from src.executors.planning.query_compiler import (
    compile_chart_grouping,
    is_native_typed_groupby_enabled,
    is_server_groupby_supported,
)


def test_compile_chart_grouping_derives_month_periods_from_date_filters() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        group_by=[S.GroupByTime(grain="MONTH")],
        filters=S.AndFilter(
            and_=[
                S.DateFilter(type="DateFilter", operator="GE", value="2022-10-01"),
                S.DateFilter(type="DateFilter", operator="LE", value="2023-09-30"),
            ]
        ),
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 12
    assert batch.batched_time_periods[0].start_date == "2022-10-01"
    assert batch.batched_time_periods[-1].end_date == "2023-09-30"


def test_map_metrics_payload_to_series_uses_month_label_for_time_periods() -> None:
    kpi = SimpleNamespace(
        kpi1=SimpleNamespace(
            median=None,
            mean=14.0,
            case_count=[],
            d1=None,
        ),
        grouped_by=None,
        time_period=SimpleNamespace(start_date="2023-02-01", end_date="2023-02-28"),
        data_origin=None,
    )
    metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

    series = map_metrics_payload_to_series(
        metrics_payload=metric_payload,
        label_parts=[],
        include_metric_alias=True,
        group_by_field=None,
        add_time_period_labels=True,
    )

    assert len(series) == 1
    assert len(series[0].data) == 1
    assert series[0].data[0].x == "2023-02"


def test_map_metrics_payload_to_series_emits_missing_time_point_as_null() -> None:
    kpi = SimpleNamespace(
        kpi1=SimpleNamespace(
            median=None,
            mean=None,
            case_count=[],
            d1=None,
        ),
        grouped_by=None,
        time_period=SimpleNamespace(start_date="2023-03-01", end_date="2023-03-31"),
        data_origin=None,
    )
    metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

    series = map_metrics_payload_to_series(
        metrics_payload=metric_payload,
        label_parts=[],
        include_metric_alias=True,
        group_by_field=None,
        add_time_period_labels=True,
    )

    assert len(series) == 1
    assert len(series[0].data) == 1
    assert series[0].data[0].x == "2023-03"
    assert series[0].data[0].y is None


def test_compile_chart_grouping_month_and_sex_defaults_to_recent_12_months() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        group_by=[
            S.GroupByTime(grain="MONTH"),
            S.GroupBySex(categories=["MALE", "FEMALE"]),
        ],
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 12
    # If SEX_TYPE is supported natively, sex is server_groupby and no filter split
    # combos are needed. Otherwise, sex remains filter-driven with two combos.
    if is_server_groupby_supported("SEX_TYPE") and is_native_typed_groupby_enabled():
        assert batch.server_groupby == "SEX_TYPE"
        assert len(batch.combos_list) == 1
    else:
        assert len(batch.combos_list) == 2


def test_compile_chart_grouping_sex_filter_split_excludes_unknown_category() -> None:
    chart = S.ChartSpec(
        chart_type="BAR",
        metrics=[S.MetricSpec(metric="HOSPITALIZED_IN")],
        group_by=[S.GroupBySex(categories=None)],
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    if is_server_groupby_supported("SEX_TYPE") and is_native_typed_groupby_enabled():
        assert batch.server_groupby == "SEX_TYPE"
    else:
        rendered = [str(value) for combo in batch.combos_list for value in combo]
        assert "UNKNOWN" not in rendered
        assert set(rendered) == {"MALE", "FEMALE", "OTHER"}


def test_compile_chart_grouping_defaults_to_recent_quarters_without_group_by() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        group_by=None,
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 8
    assert batch.combos_list == [tuple()]


def test_compile_chart_grouping_sex_only_adds_default_quarter_grain() -> None:
    chart = S.ChartSpec(
        chart_type="BAR",
        metrics=[S.MetricSpec(metric="HOSPITALIZED_IN")],
        group_by=[S.GroupBySex(categories=["MALE", "FEMALE"])],
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 8

    if is_server_groupby_supported("SEX_TYPE") and is_native_typed_groupby_enabled():
        assert batch.server_groupby == "SEX_TYPE"
        assert len(batch.combos_list) == 1
    else:
        assert len(batch.combos_list) == 2


def test_compile_chart_grouping_canonical_only_does_not_add_default_quarter() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        group_by=[S.GroupByCanonicalField(field="INR_MODE")],
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.server_groupby == "INR_MODE"
    assert batch.batched_time_enabled is False
    assert batch.batched_time_periods == []
    assert batch.combos_list == [tuple()]


def test_compile_chart_grouping_canonical_with_explicit_quarter_keeps_both() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        group_by=[
            S.GroupByCanonicalField(field="INR_MODE"),
            S.GroupByTime(grain="QUARTER"),
        ],
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.server_groupby == "INR_MODE"
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 8
    assert batch.combos_list == [tuple()]


def test_map_metrics_payload_filter_split_uses_default_plotting_path() -> None:
    kpi = SimpleNamespace(
        kpi1=SimpleNamespace(
            median=22.0,
            mean=21.0,
            case_count=[3, 7],
            d1=SimpleNamespace(edges=["0-10", "10-20"], case_count=[3, 7]),
        ),
        grouped_by=None,
        time_period=None,
        data_origin=None,
    )
    metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

    series = map_metrics_payload_to_series(
        metrics_payload=metric_payload,
        label_parts=["MALE"],
        include_metric_alias=False,
        group_by_field=None,
        add_time_period_labels=False,
    )

    assert len(series) == 1
    assert series[0].name == "MALE"
    assert [point.x for point in series[0].data] == ["0-10", "10-20"]
    assert [point.y for point in series[0].data] == [3, 7]


def test_map_metrics_payload_enum_case_counts_maps_to_categories() -> None:
    kpi = SimpleNamespace(
        kpi1=SimpleNamespace(
            median=None,
            mean=None,
            case_count=[12, 8, 3],
            percents=[52.2, 34.8, 13.0],
            d1=None,
        ),
        grouped_by=None,
        time_period=None,
        data_origin=None,
    )
    metric_payload = {"metric_SEX": SimpleNamespace(kpi_group=[kpi])}

    series = map_metrics_payload_to_series(
        metrics_payload=metric_payload,
        label_parts=[],
        include_metric_alias=True,
        group_by_field=None,
        add_time_period_labels=False,
    )

    assert len(series) == 1
    assert [point.x for point in series[0].data] == ["Male", "Female", "Unknown Other"]
    assert [point.y for point in series[0].data] == [52.2, 34.8, 13.0]

    def test_map_metrics_payload_groupby_and_time_keeps_time_x_axis() -> None:
        kpis = [
            SimpleNamespace(
                kpi1=SimpleNamespace(median=66.0, mean=None, case_count=[], d1=None),
                grouped_by=SimpleNamespace(group_item_name="NOT_DONE"),
                time_period=None,
                data_origin=None,
            ),
            SimpleNamespace(
                kpi1=SimpleNamespace(median=56.0, mean=None, case_count=[], d1=None),
                grouped_by=SimpleNamespace(group_item_name="LAB"),
                time_period=None,
                data_origin=None,
            ),
            SimpleNamespace(
                kpi1=SimpleNamespace(median=95.0, mean=None, case_count=[], d1=None),
                grouped_by=SimpleNamespace(group_item_name="NOT_DONE"),
                time_period=None,
                data_origin=None,
            ),
            SimpleNamespace(
                kpi1=SimpleNamespace(median=60.0, mean=None, case_count=[], d1=None),
                grouped_by=SimpleNamespace(group_item_name="LAB"),
                time_period=None,
                data_origin=None,
            ),
        ]
        metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=kpis)}
        batched_time_periods = [
            SimpleNamespace(start_date="2025-01-01", end_date="2025-03-31"),
            SimpleNamespace(start_date="2025-04-01", end_date="2025-06-30"),
        ]

        series = map_metrics_payload_to_series(
            metrics_payload=metric_payload,
            label_parts=["My Hospital"],
            include_metric_alias=False,
            group_by_field="INR_MODE",
            add_time_period_labels=True,
            batched_time_periods=batched_time_periods,
        )

        assert len(series) == 4
        x_values = [point.x for s in series for point in s.data]
        assert set(x_values) == {"2025-01", "2025-04"}
        assert all(x not in {"NOT_DONE", "LAB", "unknown"} for x in x_values)
        assert len(set(s.name for s in series)) >= 2


def test_map_metrics_payload_enum_case_counts_over_time_maps_to_category_series() -> (
    None
):
    kpi = SimpleNamespace(
        kpi1=SimpleNamespace(
            median=None,
            mean=None,
            case_count=[12, 8, 3],
            percents=[52.2, 34.8, 13.0],
            d1=None,
        ),
        grouped_by=None,
        time_period=SimpleNamespace(start_date="2023-02-01", end_date="2023-02-28"),
        data_origin=None,
    )
    metric_payload = {"metric_SEX": SimpleNamespace(kpi_group=[kpi])}

    series = map_metrics_payload_to_series(
        metrics_payload=metric_payload,
        label_parts=[],
        include_metric_alias=True,
        group_by_field=None,
        add_time_period_labels=True,
    )

    assert len(series) == 3
    assert {item.name for item in series} == {
        "biological sex — Male",
        "biological sex — Female",
        "biological sex — Unknown Other",
    }
    for item in series:
        assert len(item.data) == 1
        assert item.data[0].x == "2023-02"
    assert {item.data[0].y for item in series} == {52.2, 34.8, 13.0}


def test_map_metrics_payload_enum_grouped_prefers_percentages() -> None:
    kpi = SimpleNamespace(
        kpi1=SimpleNamespace(
            median=None,
            mean=None,
            case_count=[12],
            percents=[52.2],
            d1=None,
        ),
        grouped_by=SimpleNamespace(group_item_name="MALE"),
        time_period=None,
        data_origin=None,
    )
    metric_payload = {"metric_HOSPITALIZED_IN": SimpleNamespace(kpi_group=[kpi])}

    series = map_metrics_payload_to_series(
        metrics_payload=metric_payload,
        label_parts=[],
        include_metric_alias=True,
        group_by_field="SEX_TYPE",
        add_time_period_labels=False,
    )

    assert len(series) == 1
    assert series[0].data[0].x == "MALE"
    assert series[0].data[0].y == 52.2
