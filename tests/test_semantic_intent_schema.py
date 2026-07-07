import unittest

from src.domain.langchain.schema import ChartSpec


class SemanticIntentSchemaTests(unittest.TestCase):
    def test_accepts_semantic_distribution_with_split(self) -> None:
        chart = ChartSpec.model_validate(
            {
                "chart_type": "LINE",
                "metrics": [{"metric": "DTN"}],
                "semantics": {
                    "intent": "distribution",
                    "measure": {"type": "distribution"},
                    "splits": [
                        {
                            "kind": "sex",
                            "categories": ["MALE", "FEMALE"],
                        }
                    ],
                    "xAxis": {
                        "role": "metric_value",
                        "metric": "DTN",
                        "unit": "minutes",
                    },
                    "yAxis": {
                        "role": "count",
                    },
                },
            }
        )

        self.assertIsNotNone(chart.semantics)
        semantics = chart.semantics
        self.assertIsNotNone(semantics)
        self.assertEqual(semantics.intent, "DISTRIBUTION")
        self.assertIsNotNone(semantics.measure)
        self.assertEqual(semantics.measure.type, "DISTRIBUTION")
        self.assertIsNotNone(semantics.splits)
        self.assertEqual(semantics.splits[0].kind, "SEX")
        self.assertIsNotNone(semantics.x_axis)
        self.assertEqual(semantics.x_axis.metric, "DTN")

    def test_rejects_invalid_intent(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "LINE",
                    "metrics": [{"metric": "DTN"}],
                    "semantics": {
                        "intent": "WHATEVER",
                    },
                }
            )

    def test_rejects_percentile_measure_without_percentile(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "LINE",
                    "metrics": [{"metric": "DTN"}],
                    "semantics": {
                        "intent": "TREND",
                        "measure": {"type": "PERCENTILE"},
                    },
                }
            )

    def test_rejects_boolean_split_without_field(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "BAR",
                    "metrics": [{"metric": "DTN"}],
                    "semantics": {
                        "intent": "COMPARISON",
                        "splits": [
                            {
                                "kind": "BOOLEAN",
                            }
                        ],
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
