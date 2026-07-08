import unittest

from src.executors.mapping.summary_builder import make_execution_summary


class ExecutionSummaryStatsFieldsTests(unittest.TestCase):
    def test_make_execution_summary_includes_stats_fields(self) -> None:
        summary = make_execution_summary(
            trace_id="trace-1",
            chart_count=0,
            stats_count=2,
            stats_skipped=1,
            stats_errors=0,
            estimated_queries=2,
            actual_queries=2,
            batches=[],
            normalization=None,
        )

        payload = summary.model_dump(exclude_none=True)

        self.assertEqual(payload["stats_count"], 2)
        self.assertEqual(payload["stats_skipped"], 1)
        self.assertEqual(payload["stats_errors"], 0)


if __name__ == "__main__":
    unittest.main()
