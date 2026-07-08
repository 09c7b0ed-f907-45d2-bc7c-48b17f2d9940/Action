import unittest

from src.domain.dto.charts.types import ChartSeries
from src.domain.langchain import schema as S
from src.executors.mapping.chart_builder import build_chart_dto
from src.executors.planning.query_compiler import compile_chart_grouping


class ChartBuilderSemanticsTests(unittest.TestCase):
    def test_line_chart_with_semantic_time_is_not_smoothed(self) -> None:
        chart = S.ChartSpec(
            chart_type="LINE",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="TREND",
                measure=S.MeasureSemanticsSpec(type="MEAN"),
                time=S.TimeSemanticsSpec(grain="MONTH"),
            ),
            filters=S.AndFilter(
                and_=[
                    S.DateFilter(type="DateFilter", operator="GE", value="2023-01-01"),
                    S.DateFilter(type="DateFilter", operator="LE", value="2023-12-31"),
                ]
            ),
        )

        compiled = compile_chart_grouping(chart)
        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=compiled.dimensions,
            series=[ChartSeries(name="DTN", data=[])],
            derived_axes=None,
        )

        self.assertFalse(dto.smooth)

    def test_line_chart_without_time_dimension_is_smoothed(self) -> None:
        chart = S.ChartSpec(
            chart_type="LINE",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="COMPARISON",
                measure=S.MeasureSemanticsSpec(type="MEAN"),
                splits=[S.SplitSpec(kind="SEX", categories=["MALE", "FEMALE"])],
            ),
        )

        compiled = compile_chart_grouping(chart)
        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=compiled.dimensions,
            series=[ChartSeries(name="DTN", data=[])],
            derived_axes=None,
        )

        self.assertTrue(dto.smooth)

    def test_unsupported_chart_type_is_rejected_by_schema(self) -> None:
        with self.assertRaises(ValueError):
            S.ChartSpec(
                chart_type="UNSUPPORTED_SHAPE",
                metrics=[S.MetricSpec(metric="DTN")],
                semantics=S.AnalysisSemanticsSpec(
                    intent="COMPARISON",
                    measure=S.MeasureSemanticsSpec(type="MEAN"),
                ),
            )


if __name__ == "__main__":
    unittest.main()