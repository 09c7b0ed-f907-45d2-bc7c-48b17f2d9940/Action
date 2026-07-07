import unittest
from unittest.mock import patch

from src.domain.langchain import schema as S
from src.executors.planning import origin_scope_resolver


class OriginScopeResolverSemanticsTests(unittest.TestCase):
    def test_resolve_plan_metric_origins_preserves_chart_semantics(self) -> None:
        plan = S.AnalysisPlan(
            charts=[
                S.ChartSpec(
                    chart_type="LINE",
                    metrics=[S.MetricSpec(metric="DTN")],
                    semantics=S.AnalysisSemanticsSpec(
                        intent="TREND",
                        measure=S.MeasureSemanticsSpec(type="MEAN"),
                        time=S.TimeSemanticsSpec(grain="MONTH"),
                    ),
                )
            ]
        )

        with patch.object(origin_scope_resolver, "_resolve_metric_origin", side_effect=lambda metric, **_: metric):
            resolved = origin_scope_resolver.resolve_plan_metric_origins(
                plan=plan,
                user_sub="user-1",
                trace_id="trace-1",
            )

        self.assertIsNotNone(resolved.charts)
        resolved_chart = resolved.charts[0]
        self.assertIsNotNone(resolved_chart.semantics)
        self.assertEqual(resolved_chart.semantics.intent, "TREND")
        self.assertIsNotNone(resolved_chart.semantics.measure)
        self.assertEqual(resolved_chart.semantics.measure.type, "MEAN")
        self.assertIsNotNone(resolved_chart.semantics.time)
        self.assertEqual(resolved_chart.semantics.time.grain, "MONTH")


if __name__ == "__main__":
    unittest.main()