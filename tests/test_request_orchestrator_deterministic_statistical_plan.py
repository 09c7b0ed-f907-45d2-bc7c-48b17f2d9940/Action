import unittest
from unittest.mock import patch

from src.planners.langchain.request_orchestrator import orchestrate_visualization_request
from src.planners.langchain.request_orchestrator import VisualizationRequestOutcome


class RequestOrchestratorDeterministicStatPlanTests(unittest.TestCase):
    def test_builds_deterministic_statistical_plan_for_semantic_scopes(self) -> None:
        entities = {
            "metric": ["DTN"],
            "provider_name": ["Aalborg Hospital", "Copenhagen Hospital"],
            "date": ["2023-01-01", "2023-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
        }

        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(
                decision="proceed",
                reason="all_required_fields_present",
            ),
        ), patch("src.planners.langchain.request_orchestrator.generate_analysis_plan") as llm_plan:
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN between Aalborg Hospital and Copenhagen Hospital",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertFalse(llm_plan.called)

        test = outcome.plan.statistical_tests[0]
        self.assertEqual(test.metrics[0].origin_scope.scope_type, "provider_name")
        self.assertEqual(test.metrics[0].origin_scope.value, "Aalborg Hospital")
        self.assertEqual(test.metrics[1].origin_scope.scope_type, "provider_name")
        self.assertEqual(test.metrics[1].origin_scope.value, "Copenhagen Hospital")

    def test_builds_deterministic_statistical_plan_for_provider_group_comparison(self) -> None:
        entities = {
            "metric": ["DTN"],
            "provider_group_id": ["provider group 2825", "provider group 3001"],
            "date": ["2023-01-01", "2023-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
        }

        with patch(
            "src.planners.langchain.request_orchestrator._decision_stage",
            return_value=VisualizationRequestOutcome(
                decision="proceed",
                reason="all_required_fields_present",
            ),
        ), patch("src.planners.langchain.request_orchestrator.generate_analysis_plan") as llm_plan:
            outcome = orchestrate_visualization_request(
                question="Run a Mann-Whitney U test for DTN, cohort A is provider group 2825 from 2023-01-01 to 2023-12-31, cohort B is provider group 3001 from 2023-01-01 to 2023-12-31",
                entities=entities,
                include_plan=True,
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertFalse(llm_plan.called)

        test = outcome.plan.statistical_tests[0]
        self.assertEqual(test.test_type, "MANN_WHITNEY_U_TEST")
        self.assertEqual(test.metrics[0].data_origin.provider_group_id, [2825])
        self.assertEqual(test.metrics[1].data_origin.provider_group_id, [3001])

        and_filters = test.filters.and_
        self.assertEqual(len(and_filters), 2)
        self.assertEqual(and_filters[0].operator, "GE")
        self.assertEqual(and_filters[0].value, "2023-01-01")
        self.assertEqual(and_filters[1].operator, "LE")
        self.assertEqual(and_filters[1].value, "2023-12-31")


if __name__ == "__main__":
    unittest.main()
