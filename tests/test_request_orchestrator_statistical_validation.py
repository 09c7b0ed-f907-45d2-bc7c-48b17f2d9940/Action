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
    def test_clarifies_when_statistical_entities_do_not_define_two_cohorts(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between Aalborg University - Hospital and Kronborg Castle Hospital",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "metric": "DTN",
                    "date": ["2024-01-01", "2026-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")
        self.assertIn("two explicit cohorts", outcome.message or "")

    def test_proceeds_to_planning_when_entities_define_two_cohorts(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        MetricSpec(
                            metric="DTN",
                            data_origin=DataOriginSpec(providerGroupId=[279]),
                            origin_scope=OriginScopeSpec(scopeType="provider_group_name", value="Aalborg University - Hospital", label="Aalborg University - Hospital"),
                        ),
                        MetricSpec(
                            metric="DTN",
                            data_origin=DataOriginSpec(providerGroupId=[280]),
                            origin_scope=OriginScopeSpec(scopeType="provider_group_name", value="Kronborg Castle Hospital", label="Kronborg Castle Hospital"),
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
                question="Run a Mann-Whitney U test for DTN between provider group 279 and provider group 280 from 2024-01-01 to 2026-12-31",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "metric": "DTN",
                    "provider_group_id": ["provider group 279", "provider group 280"],
                    "date": ["2024-01-01", "2026-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)

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
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "provider_group_id": ["provider group 111", "provider group 112"],
                },
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
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "provider_group_id": ["provider group 1", "provider group 2"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)

    def test_clarifies_when_question_mentions_provider_group_but_entities_only_have_provider_ids(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between provider group 289 and provider group 252",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "metric": "DTN",
                    "provider_id": ["provider 289", "provider 252"],
                    "date": ["2023-01-01", "2023-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_provider_group_cohorts")
        self.assertIn("provider groups", outcome.message or "")

    def test_clarifies_when_question_mentions_provider_but_entities_only_have_provider_group_ids(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(decision="proceed", reason="ok"),
        ):
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between provider 289 and provider 252",
                entities={
                    "statistical_test_type": "MANN_WHITNEY_U_TEST",
                    "metric": "DTN",
                    "provider_group_id": ["provider group 289", "provider group 252"],
                    "date": ["2023-01-01", "2023-12-31"],
                },
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_provider_cohorts")
        self.assertIn("providers", outcome.message or "")


if __name__ == "__main__":
    unittest.main()
