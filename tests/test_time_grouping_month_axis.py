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

    def test_map_metrics_payload_to_series_uses_day_labels_for_daily_buckets(self) -> None:
        """Regression test: a 'last 30 days, daily' request previously
        rendered as if it had ~2 data points, because every single-day
        bucket in the same month collapsed onto the same "%Y-%m" label.
        Distinct day buckets must get distinct, day-precision labels.
        """
        kpis = [
            SimpleNamespace(
                kpi1=SimpleNamespace(median=None, mean=float(i), case_count=[], d1=None),
                grouped_by=None,
                time_period=SimpleNamespace(start_date=f"2026-07-{6 + i:02d}", end_date=f"2026-07-{6 + i:02d}"),
                data_origin=None,
            )
            for i in range(5)
        ]
        metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=kpis)}

        series = map_metrics_payload_to_series(
            metrics_payload=metric_payload,
            label_parts=[],
            include_metric_alias=True,
            group_by_field=None,
            add_time_period_labels=True,
        )

        x_values = [point.x for s in series for point in s.data]
        self.assertEqual(x_values, ["2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10"])
        self.assertEqual(len(set(x_values)), 5, "distinct day buckets must not share a label")

    def test_map_metrics_payload_to_series_uses_quarter_label_for_calendar_quarter(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(median=None, mean=14.0, case_count=[], d1=None),
            grouped_by=None,
            time_period=SimpleNamespace(start_date="2023-04-01", end_date="2023-06-30"),
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

        self.assertEqual(series[0].data[0].x, "2023-Q2")

    def test_map_metrics_payload_to_series_uses_year_label_for_calendar_year(self) -> None:
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(median=None, mean=14.0, case_count=[], d1=None),
            grouped_by=None,
            time_period=SimpleNamespace(start_date="2023-01-01", end_date="2023-12-31"),
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

        self.assertEqual(series[0].data[0].x, "2023")

    def test_map_metrics_payload_to_series_uses_full_date_for_non_calendar_aligned_span(self) -> None:
        """A week-ish bucket that doesn't line up with a calendar month/
        quarter/year boundary must fall back to full date precision, not be
        mistaken for one of those and mislabeled."""
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(median=None, mean=14.0, case_count=[], d1=None),
            grouped_by=None,
            time_period=SimpleNamespace(start_date="2023-02-06", end_date="2023-02-12"),
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

        self.assertEqual(series[0].data[0].x, "2023-02-06")

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

    def test_map_metrics_payload_to_series_uses_single_aggregate_point_for_filter_grouped_bucket(self) -> None:
        """Regression test: an age/NIHSS bucket split (or any boolean split) has
        no server-side groupBy -- GraphQL has no native groupBy for arbitrary
        buckets, so each bucket is a separate, case-filtered request instead
        (group_by_field=None). Each such request's kpi1 carries BOTH the
        bucket-scoped aggregate (median) AND a full d1 distribution
        (build_metric_requests always requests both for numeric metrics).
        Previously, group_by_field=None alone decided which branch to take, so
        every bucket's series silently rendered the 52-point d1 histogram
        instead of the one point that actually represents that bucket --
        is_filter_grouped is the signal that this batch, despite having no
        server_groupby, still represents exactly one category.
        """
        kpi = SimpleNamespace(
            kpi1=SimpleNamespace(
                median=42.0,
                mean=99.0,
                case_count=[7],
                d1=SimpleNamespace(edges=[0, 10, 20], case_count=[1, 2, 3]),
            ),
            grouped_by=None,
            time_period=None,
            data_origin=None,
        )
        metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

        series = map_metrics_payload_to_series(
            metrics_payload=metric_payload,
            label_parts=["10-20"],
            include_metric_alias=True,
            group_by_field=None,
            add_time_period_labels=False,
            is_filter_grouped=True,
        )

        self.assertEqual(len(series), 1)
        self.assertEqual(len(series[0].data), 1, "must be one aggregate point, not the full d1 distribution")
        self.assertEqual(series[0].data[0].y, 42.0, "must use kpi1.median, not d1's histogram values")
        self.assertIn("10-20", series[0].name)

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
