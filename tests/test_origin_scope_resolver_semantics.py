import unittest
from unittest.mock import patch

from src.domain.langchain import schema as S
from src.executors.planning import origin_scope_resolver


class OriginScopeResolverSemanticsTests(unittest.TestCase):
    def test_list_accessible_providers_does_not_apply_user_filter(self) -> None:
        calls = []

        class FakeClient:
            def list_providers(self, **kwargs):
                calls.append(kwargs)
                return {
                    "results": [
                        {
                            "id": 1853,
                            "nameEnglish": "Army Alhama de Murcia Hospital",
                        }
                    ],
                    "count": 1,
                }

        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=FakeClient()):
            providers = origin_scope_resolver._list_accessible_providers(
                user_sub="0a709c3b-2c71-4c5b-85d6-66454da5c9d7:thread:4",
                trace_id="trace-1",
            )

        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["id"], 1853)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("user", calls[0])

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

    def test_search_accessible_providers_by_name_stops_after_exact_match(self) -> None:
        calls = []

        class FakeClient:
            def list_providers(self, **kwargs):
                calls.append(kwargs)
                return {
                    "results": [
                        {
                            "id": 1853,
                            "nameEnglish": "Army Alhama de Murcia Hospital",
                        }
                    ],
                    "count": 1,
                }

        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=FakeClient()):
            providers = origin_scope_resolver._search_accessible_providers_by_name(
                requested_names=["Army Alhama de Murcia Hospital"],
                user_sub="user-1",
                trace_id="trace-2",
            )

        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["id"], 1853)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].get("user"), "user-1")

    def test_resolve_metric_origin_enriches_mine_scope_with_provider_label(self) -> None:
        metric = S.MetricSpec(
            metric="DTN",
            originScope=S.OriginScopeSpec(scopeType="mine"),
        )

        with patch.object(
            origin_scope_resolver,
            "_resolve_scope",
            return_value=(S.DataOriginSpec(providerId=[1853]), "Army Alhama de Murcia Hospital"),
        ):
            resolved = origin_scope_resolver._resolve_metric_origin(
                metric,
                default_scope=None,
                user_sub="user-1",
                trace_id="trace-3",
                fail_open_for_default_scope=False,
                inferred_country_code=None,
            )

        self.assertIsNotNone(resolved.origin_scope)
        self.assertEqual(resolved.origin_scope.label, "Army Alhama de Murcia Hospital")
        self.assertEqual(resolved.data_origin.provider_id, [1853])

    def test_resolve_metric_origin_overwrites_generic_mine_label(self) -> None:
        for placeholder in ["My hospital", "mine", "My Hospital", "our hospital"]:
            with self.subTest(placeholder=placeholder):
                metric = S.MetricSpec(
                    metric="DTN",
                    originScope=S.OriginScopeSpec(scopeType="mine", label=placeholder),
                )

                with patch.object(
                    origin_scope_resolver,
                    "_resolve_scope",
                    return_value=(S.DataOriginSpec(providerId=[279]), "Aalborg University - Hospital"),
                ):
                    resolved = origin_scope_resolver._resolve_metric_origin(
                        metric,
                        default_scope=None,
                        user_sub="user-1",
                        trace_id="trace-4",
                        fail_open_for_default_scope=False,
                        inferred_country_code=None,
                    )

                self.assertEqual(resolved.origin_scope.label, "Aalborg University - Hospital")

    def test_resolve_metric_origin_keeps_scope_label_when_resolver_returns_none(self) -> None:
        # When _resolve_scope returns no label (non-mine scopes), the original
        # scope label is preserved unchanged.
        metric = S.MetricSpec(
            metric="DTN",
            originScope=S.OriginScopeSpec(scopeType="provider_name", value="My Clinic", label="My Clinic"),
        )

        with patch.object(
            origin_scope_resolver,
            "_resolve_scope",
            return_value=(S.DataOriginSpec(providerId=[279]), None),
        ):
            resolved = origin_scope_resolver._resolve_metric_origin(
                metric,
                default_scope=None,
                user_sub="user-1",
                trace_id="trace-5",
                fail_open_for_default_scope=False,
                inferred_country_code=None,
            )

        self.assertEqual(resolved.origin_scope.label, "My Clinic")


if __name__ == "__main__":
    unittest.main()
