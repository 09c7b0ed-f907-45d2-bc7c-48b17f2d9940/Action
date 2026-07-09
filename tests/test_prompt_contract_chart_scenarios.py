import unittest

from src.domain.langchain.schema import AnalysisPlan, ChartSpec, MetricSpec

from tests.prompt_contract_harness import load_scenarios, run_orchestrator


class PromptContractChartScenarioTests(unittest.TestCase):
    def test_chart_prompt_contract_scenario(self) -> None:
        scenarios = load_scenarios("prompt_contract_chart_scenarios.json")
        scenario = scenarios[0]
        chart_plan = AnalysisPlan(
            charts=[
                ChartSpec(
                    chart_type="LINE",
                    metrics=[MetricSpec(metric="DTN")],
                )
            ]
        )

        outcome = run_orchestrator(
            prompt=str(scenario["prompt"]),
            entities=dict(scenario["entities"]),
            plan=chart_plan,
        )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)
        self.assertTrue(outcome.plan.charts)
        self.assertEqual(outcome.plan.charts[0].chart_type, "LINE")
        self.assertEqual(outcome.plan.charts[0].metrics[0].metric, "DTN")

    def test_chart_with_year_entity_scenario(self) -> None:
        scenarios = load_scenarios("prompt_contract_chart_scenarios.json")
        scenario = next(item for item in scenarios if item["id"] == "chart_line_dtn_with_year_entity")
        chart_plan = AnalysisPlan(
            charts=[
                ChartSpec(
                    chart_type="LINE",
                    metrics=[MetricSpec(metric="DTN")],
                )
            ]
        )

        outcome = run_orchestrator(
            prompt=str(scenario["prompt"]),
            entities=dict(scenario["entities"]),
            plan=chart_plan,
        )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)
        self.assertTrue(outcome.plan.charts)
        self.assertEqual(outcome.plan.charts[0].chart_type, "LINE")
        self.assertEqual(outcome.plan.charts[0].metrics[0].metric, "DTN")

    def test_chart_with_optional_demographics_scenario(self) -> None:
        scenarios = load_scenarios("prompt_contract_chart_scenarios.json")
        scenario = next(item for item in scenarios if item["id"] == "chart_bar_dtn_with_optional_demographics")
        chart_plan = AnalysisPlan(
            charts=[
                ChartSpec(
                    chart_type="BAR",
                    metrics=[MetricSpec(metric="DTN")],
                )
            ]
        )

        outcome = run_orchestrator(
            prompt=str(scenario["prompt"]),
            entities=dict(scenario["entities"]),
            plan=chart_plan,
        )

        self.assertEqual(outcome.decision, "proceed")
        self.assertIsNotNone(outcome.plan)
        self.assertTrue(outcome.plan.charts)
        self.assertEqual(outcome.plan.charts[0].chart_type, "BAR")
        self.assertEqual(outcome.plan.charts[0].metrics[0].metric, "DTN")


if __name__ == "__main__":
    unittest.main()