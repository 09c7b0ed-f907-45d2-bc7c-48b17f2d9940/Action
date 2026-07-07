import os
import unittest
from unittest.mock import patch

os.environ.setdefault("RASA_PROXY_URL", "http://localhost")
os.environ.setdefault("ACTION_SERVER_TOKEN", "dummy")
os.environ.setdefault("RASA_PROXY_GRAPHQL_TARGET", "http://localhost/graphql")

from src.domain.langchain.schema import GroupBySex, MetricSpec, StatisticalTestSpec
from src.executors.orchestration import plan_executor


class PlanExecutorStatisticalTestTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()