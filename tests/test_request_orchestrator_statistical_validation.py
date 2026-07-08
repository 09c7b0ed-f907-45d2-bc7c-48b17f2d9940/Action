import unittest
from unittest.mock import patch

from src.domain.langchain.schema import (
    AnalysisPlan,
    DataOriginSpec,
    MetricSpec,
    OriginScopeSpec,
    StatisticalTestSpec,
)
from src.planners.langchain.request_orchestrator import orchestrate_visualization_request
from src.planners.langchain.request_orchestrator import VisualizationRequestOutcome


class RequestOrchestratorStatisticalValidationTests(unittest.TestCase):
    def test_clarifies_when_mann_whitney_has_fewer_than_two_metrics(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[MetricSpec(metric="DTN")],
                )
            ]
        )

        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ), patch(
            "src.planners.langchain.request_orchestrator.generate_analysis_plan",
            return_value=plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test on DTN",
                entities={"statistical_test_type": "MANN_WHITNEY_U_TEST"},
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")
        self.assertIn("two explicit cohorts", outcome.message or "")

    def test_clarifies_when_mann_whitney_metrics_are_not_distinct_cohorts(self) -> None:
        shared_origin = DataOriginSpec(providerId=[1])
        shared_scope = OriginScopeSpec(scopeType="mine", label="same")
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(metric="DTN", data_origin=shared_origin, origin_scope=shared_scope),
                        MetricSpec(metric="DTN", data_origin=shared_origin, origin_scope=shared_scope),
                    ],
                )
            ]
        )

        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ), patch(
            "src.planners.langchain.request_orchestrator.generate_analysis_plan",
            return_value=plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN",
                entities={"statistical_test_type": "MANN_WHITNEY_U_TEST"},
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")
        self.assertIn("two distinct cohorts", outcome.message or "")

    def test_proceeds_when_distinct_cohorts_are_present(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(
                            metric="DTN",
                            data_origin=DataOriginSpec(providerId=[1]),
                            origin_scope=OriginScopeSpec(scopeType="mine", label="male cohort"),
                        ),
                        MetricSpec(
                            metric="DTN",
                            data_origin=DataOriginSpec(providerId=[1]),
                            origin_scope=OriginScopeSpec(scopeType="country_code", countryCode="CZ", label="female cohort"),
                        ),
                    ],
                )
            ]
        )

        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ), patch(
            "src.planners.langchain.request_orchestrator.generate_analysis_plan",
            return_value=plan,
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN",
                entities={"statistical_test_type": "MANN_WHITNEY_U_TEST"},
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)


if __name__ == "__main__":
    unittest.main()
