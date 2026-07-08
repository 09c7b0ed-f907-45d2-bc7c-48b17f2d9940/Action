import os
import unittest
import asyncio
from unittest.mock import patch

os.environ.setdefault("RASA_PROXY_URL", "http://localhost")
os.environ.setdefault("ACTION_SERVER_TOKEN", "dummy")
os.environ.setdefault("RASA_PROXY_GRAPHQL_TARGET", "http://localhost/graphql")

from src.domain.langchain.schema import AnalysisPlan, AndFilter, DataOriginSpec, DateFilter, GroupBySex, MetricSpec, StatisticalTestSpec
from src.executors.orchestration import plan_executor


class PlanExecutorStatisticalTestTests(unittest.TestCase):
    def test_mann_whitney_returns_clarification_when_no_cohort_rows_exist(self) -> None:
        test = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[1])),
                MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerId=[2])),
            ],
            filters=AndFilter(
                and_=[
                    DateFilter(operator="GE", value="2023-01-01"),
                    DateFilter(operator="LE", value="2023-12-31"),
                ]
            ),
        )

        with patch.object(
            plan_executor,
            "_translate_mann_whitney_metrics",
            return_value=(["DTN"], None),
        ), patch.object(
            plan_executor,
            "_execute_mann_whitney_query",
            return_value=[],
        ):
            with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
                plan_executor._execute_mann_whitney_test(
                    test=test,
                    user_sub="user-1",
                    trace_id="trace-1",
                )

        self.assertEqual(err.exception.reason, "empty_statistical_cohort")
        self.assertEqual(err.exception.code, "EXEC_STATS_EMPTY_COHORT")
        self.assertIn("different cohorts or a broader scope", err.exception.user_message)

    def test_mann_whitney_group_by_does_not_define_cohorts(self) -> None:
        test = StatisticalTestSpec(
            test_type="MANN_WHITNEY_U_TEST",
            metrics=[MetricSpec(metric="DTN")],
            group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
        )

        with patch.object(
            plan_executor,
            "_translate_mann_whitney_metrics",
            return_value=(["DTN"], None),
        ):
            results = plan_executor._execute_mann_whitney_test(
                test=test,
                user_sub="user-1",
                trace_id="trace-1",
            )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, "skipped")
        self.assertEqual(
            result.reason,
            "MANN_WHITNEY_U_TEST requires two explicit metric cohorts",
        )

    def test_execute_plan_async_fails_fast_when_statistical_cohorts_are_missing(self) -> None:
        plan = AnalysisPlan(
            statistical_tests=[
                StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[MetricSpec(metric="DTN")],
                    group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                )
            ]
        )

        with patch.object(plan_executor, "resolve_plan_metric_origins", return_value=plan):
            with self.assertRaises(plan_executor.VisualizationExecutionError) as err:
                asyncio.run(
                    plan_executor.execute_plan_async(
                        plan=plan,
                        user_sub="user-1",
                        trace_id="trace-1",
                    )
                )

        self.assertEqual(err.exception.reason, "missing_statistical_cohorts")
        self.assertEqual(err.exception.code, "EXEC_STATS_MISSING_COHORTS")


if __name__ == "__main__":
    unittest.main()