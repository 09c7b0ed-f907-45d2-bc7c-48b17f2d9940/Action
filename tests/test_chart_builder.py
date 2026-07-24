import unittest

from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.langchain.schema import ChartSpec, MetricSpec
from src.executors.mapping.chart_builder import build_chart_dto


class ChartBuilderTests(unittest.TestCase):
    def test_bar_distribution_without_dimensions_uses_metric_and_cases_axes(
        self,
    ) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            metrics=[MetricSpec(metric="SYSTOLIC_PRESSURE")],
        )
        series = [
            ChartSeries(
                name="SYSTOLIC_PRESSURE",
                data=[
                    ChartPoint(x=0, y=0),
                    ChartPoint(x=30, y=15),
                    ChartPoint(x=60, y=18),
                ],
            )
        ]

        chart = build_chart_dto(
            plan_chart=plan_chart,
            dimensions=[],
            series=series,
            derived_axes=None,
        )

        self.assertIsNotNone(chart.metadata.x_axis)
        self.assertIsNotNone(chart.metadata.y_axis)
        x_axis = chart.metadata.x_axis
        y_axis = chart.metadata.y_axis
        if x_axis is None or y_axis is None:
            self.fail("Expected both axes to be present")
        self.assertEqual(x_axis.label, "initial systolic blood pressure (mmHg)")
        self.assertEqual(x_axis.type.value, "linear")
        self.assertEqual(y_axis.label, "Cases")
        self.assertEqual(y_axis.type.value, "linear")


if __name__ == "__main__":
    unittest.main()
