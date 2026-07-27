import unittest

from src.domain.langchain.schema import AnalysisPlan, AndFilter, DataOriginSpec, DateFilter, MetricSpec, StatisticalTestSpec

from tests.prompt_contract_harness import load_scenarios, run_orchestrator


def _make_mw_plan() -> AnalysisPlan:
    return AnalysisPlan(
        statistical_tests=[
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                metrics=[
                    MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerGroupId=[289])),
                    MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerGroupId=[252])),
                ],
                filters=AndFilter(
                    and_=[
                        DateFilter(operator="GE", value="2023-01-01"),
                        DateFilter(operator="LE", value="2023-12-31"),
                    ]
                ),
            )
        ]
    )


class PromptContractStatisticalScenarioTests(unittest.TestCase):
    def test_missing_cohort_clarification_scenario(self) -> None:
        scenarios = load_scenarios("prompt_contract_statistical_scenarios.json")
        missing = next(item for item in scenarios if item["id"] == "mw_missing_cohorts")

        outcome = run_orchestrator(
            prompt=str(missing["prompt"]),
            entities=dict(missing["entities"]),
        )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_statistical_cohorts")
        self.assertEqual(outcome.clarification_type, "analysis_plan")
        self.assertIn("two explicit cohorts", outcome.message or "")

    def test_statistical_plan_scenario(self) -> None:
        scenarios = load_scenarios("prompt_contract_statistical_scenarios.json")
        empty = next(item for item in scenarios if item["id"] == "mw_empty_cohort")

        outcome = run_orchestrator(
            prompt=str(empty["prompt"]),
            entities=dict(empty["entities"]),
            plan=_make_mw_plan(),
        )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(outcome.plan)
        self.assertTrue(outcome.plan.statistical_tests)

    def test_multi_turn_clarify_to_proceed_scenario(self) -> None:
        scenarios = load_scenarios("prompt_contract_statistical_scenarios.json")
        multi_turn = next(item for item in scenarios if item["id"] == "mw_missing_then_fix")
        turns = list(multi_turn["turns"])

        first_outcome = run_orchestrator(
            prompt=str(turns[0]["prompt"]),
            entities=dict(turns[0]["entities"]),
        )
        self.assertEqual(first_outcome.decision, "clarify")
        self.assertEqual(first_outcome.reason, "missing_statistical_cohorts")

        second_outcome = run_orchestrator(
            prompt=str(turns[1]["prompt"]),
            entities=dict(turns[1]["entities"]),
            plan=_make_mw_plan(),
        )
        self.assertEqual(second_outcome.decision, "proceed")
        self.assertEqual(second_outcome.reason, "deterministic_statistical_plan")
        self.assertIsNotNone(second_outcome.plan)

    def test_provider_group_phrase_missing_second_group_clarifies(self) -> None:
        scenarios = load_scenarios("prompt_contract_statistical_scenarios.json")
        scenario = next(item for item in scenarios if item["id"] == "mw_provider_group_phrase_missing_second_group")

        outcome = run_orchestrator(
            prompt=str(scenario["prompt"]),
            entities=dict(scenario["entities"]),
        )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_provider_group_cohorts")
        self.assertEqual(outcome.clarification_type, "analysis_plan")
        self.assertIn("provider-group cohorts", outcome.message or "")

    def test_provider_phrase_missing_second_provider_clarifies(self) -> None:
        scenarios = load_scenarios("prompt_contract_statistical_scenarios.json")
        scenario = next(item for item in scenarios if item["id"] == "mw_provider_phrase_missing_second_provider")

        outcome = run_orchestrator(
            prompt=str(scenario["prompt"]),
            entities=dict(scenario["entities"]),
        )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_provider_cohorts")
        self.assertEqual(outcome.clarification_type, "analysis_plan")
        self.assertIn("provider cohorts", outcome.message or "")


if __name__ == "__main__":
    unittest.main()