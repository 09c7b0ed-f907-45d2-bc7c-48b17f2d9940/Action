from types import SimpleNamespace
import unittest

from src.domain.langchain import schema as S
from src.executors.mapping.series_mapper import map_metrics_payload_to_series
from src.executors.planning.query_compiler import compile_chart_grouping


class TimeGroupingMonthAxisTests(unittest.TestCase):
    def test_compile_chart_grouping_derives_month_periods_from_date_filters(self) -> None:
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
                    S.DateFilter(type="DateFilter", operator="GE", value="2022-10-01"),
                    S.DateFilter(type="DateFilter", operator="LE", value="2023-09-30"),
                ]
            ),
        )

        compiled = compile_chart_grouping(chart)
        self.assertEqual(len(compiled.batches), 1)

        batch = compiled.batches[0]
        self.assertTrue(batch.batched_time_enabled)
        self.assertEqual(len(batch.batched_time_periods), 12)
        self.assertEqual(batch.batched_time_periods[0].start_date, "2022-10-01")
        self.assertEqual(batch.batched_time_periods[-1].end_date, "2023-09-30")

    def test_map_metrics_payload_to_series_uses_month_label_for_time_periods(self) -> None:
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

        self.assertEqual(len(series), 1)
        self.assertEqual(len(series[0].data), 1)
        self.assertEqual(series[0].data[0].x, "2023-02")

    def test_map_metrics_payload_to_series_emits_missing_time_point_as_null(self) -> None:
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

        self.assertEqual(len(series), 1)
        self.assertEqual(len(series[0].data), 1)
        self.assertEqual(series[0].data[0].x, "2023-03")
        self.assertIsNone(series[0].data[0].y)

    def test_map_metrics_payload_to_series_fails_when_distribution_payload_missing(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(
                median=14.0,
                mean=None,
                case_count=[],
                d1=None,
            ),
            grouped_by=None,
            time_period=None,
            data_origin=None,
        )
        metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

        with self.assertRaises(ValueError) as err:
            map_metrics_payload_to_series(
                metrics_payload=metric_payload,
                label_parts=["MALE"],
                include_metric_alias=True,
                group_by_field=None,
                add_time_period_labels=False,
            )

        self.assertIn("missing d1", str(err.exception))

    def test_compile_chart_grouping_month_and_sex_falls_back_to_default_temporal_bounds(self) -> None:
        chart = S.ChartSpec(
            chart_type="LINE",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="TREND",
                measure=S.MeasureSemanticsSpec(type="MEAN"),
                time=S.TimeSemanticsSpec(grain="MONTH"),
                splits=[S.SplitSpec(kind="SEX", categories=["MALE", "FEMALE"])],
            ),
        )

        result = compile_chart_grouping(chart)
        self.assertGreater(result.total_requests, 0)

    def test_map_metrics_payload_to_series_raises_on_non_numeric_grouped_case_count(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(
                median=None,
                mean=None,
                case_count=["not-a-number"],
                d1=None,
            ),
            grouped_by=SimpleNamespace(group_item_name="MALE"),
            time_period=SimpleNamespace(start_date="2023-03-01", end_date="2023-03-31"),
            data_origin=None,
        )
        metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

        with self.assertRaises(ValueError) as err:
            map_metrics_payload_to_series(
                metrics_payload=metric_payload,
                label_parts=[],
                include_metric_alias=True,
                group_by_field="SEX",
                add_time_period_labels=True,
            )

        self.assertIn("case_count", str(err.exception))

    def test_map_metrics_payload_to_series_raises_when_distribution_d1_missing(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(
                median=14.0,
                mean=None,
                case_count=[],
                d1=None,
            ),
            grouped_by=None,
            time_period=None,
            data_origin=None,
        )
        metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

        with self.assertRaises(ValueError) as err:
            map_metrics_payload_to_series(
                metrics_payload=metric_payload,
                label_parts=[],
                include_metric_alias=True,
                group_by_field=None,
                add_time_period_labels=False,
            )

        self.assertIn("missing d1", str(err.exception))


if __name__ == "__main__":
    unittest.main()
