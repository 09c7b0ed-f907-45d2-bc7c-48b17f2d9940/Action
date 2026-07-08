from src.domain.langchain import schema as S
from src.executors.planning.query_compiler import compile_chart_grouping


def test_compile_chart_grouping_prefers_semantics_splits_and_time() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=S.MeasureSemanticsSpec(type="MEAN"),
            time=S.TimeSemanticsSpec(grain="MONTH"),
            splits=[S.SplitSpec(kind="SEX", categories=["MALE", "FEMALE"])],
        ),
        filters=S.AndFilter(
            and_=[
                S.DateFilter(type="DateFilter", operator="GE", value="2023-01-01"),
                S.DateFilter(type="DateFilter", operator="LE", value="2023-12-31"),
            ]
        ),
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 12
    # Compiled from semantic split SEX, not fallback group_by stroke type.
    assert len(batch.combos_list) == 2


def test_compile_chart_grouping_rejects_time_grain_without_explicit_bounds() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=S.MeasureSemanticsSpec(type="MEAN"),
            time=S.TimeSemanticsSpec(grain="MONTH"),
        ),
    )

    try:
        compile_chart_grouping(chart)
        assert False, "Expected ValueError when semantic time grain lacks explicit bounds"
    except ValueError as exc:
        assert "explicit time window/range" in str(exc)


def test_compile_chart_grouping_rejects_unsupported_custom_split() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="COMPARISON",
            measure=S.MeasureSemanticsSpec(type="MEDIAN"),
            splits=[S.SplitSpec(kind="CUSTOM")],
        ),
    )

    try:
        compile_chart_grouping(chart)
        assert False, "Expected ValueError for unsupported CUSTOM split"
    except ValueError as exc:
        assert "CUSTOM" in str(exc)


def test_compile_chart_grouping_requires_semantics() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
    )

    try:
        compile_chart_grouping(chart)
        assert False, "Expected ValueError when semantics are missing"
    except ValueError as exc:
        assert "semantics" in str(exc)


def test_compile_chart_grouping_requires_semantic_measure() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            time=S.TimeSemanticsSpec(grain="MONTH"),
        ),
    )

    try:
        compile_chart_grouping(chart)
        assert False, "Expected ValueError when semantics.measure is missing"
    except ValueError as exc:
        assert "semantics.measure" in str(exc)
