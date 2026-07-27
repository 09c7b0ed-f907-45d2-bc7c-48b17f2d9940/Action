import ast
import logging
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.domain.langchain import schema as S


def _load_build_guided_plan():
    source_path = Path(__file__).resolve().parents[1] / "src" / "actions" / "guided_visualization_validation.py"
    source = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(source_path))

    required = {
        "_optional_slot_value",
        "_default_semantic_intent",
        "_default_semantic_measure",
        "_semantic_grouping",
        "parse_guided_scope",
        "build_guided_plan",
    }
    selected = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name in required
    ]

    isolated_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated_module)

    namespace = {
        "S": S,
        "logger": logging.getLogger("test-guided-plan"),
        "_guided_scope_log_context": lambda **_: {},
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "SKIP_SENTINEL": "__skip__",
    }
    exec(compile(isolated_module, filename=str(source_path), mode="exec"), namespace)
    return namespace["build_guided_plan"]


class GuidedVisualizationSemanticsTests(unittest.TestCase):
    def test_build_guided_plan_translates_canonical_grouping_to_semantic_split(self) -> None:
        build_guided_plan = _load_build_guided_plan()
        plan = build_guided_plan(
            slots={
                "metric": "DTN",
                "chart_type": "BAR",
                "group_by": "FIRST_CONTACT_PLACE",
                "stroke_type": "ISCHEMIC",
            },
            user_sub="user-1",
            trace_id="trace-1",
        )

        self.assertIsNotNone(plan.charts)
        chart = plan.charts[0]
        self.assertIsNotNone(chart.semantics)
        semantics = chart.semantics
        self.assertIsNotNone(semantics)
        self.assertEqual(semantics.intent, "COMPARISON")
        self.assertIsNotNone(semantics.measure)
        self.assertEqual(semantics.measure.type, "DISTRIBUTION")
        self.assertIsNotNone(semantics.splits)
        self.assertEqual(semantics.splits[0].kind, "CANONICAL")
        self.assertEqual(semantics.splits[0].field, "FIRST_CONTACT_PLACE")
        self.assertIsNone(semantics.time)

    def test_build_guided_plan_translates_time_grain_to_semantic_time(self) -> None:
        build_guided_plan = _load_build_guided_plan()
        plan = build_guided_plan(
            slots={
                "metric": "DTN",
                "chart_type": "LINE",
                "group_by": "month",
            },
            user_sub="user-1",
            trace_id="trace-2",
        )

        self.assertIsNotNone(plan.charts)
        chart = plan.charts[0]
        self.assertIsNotNone(chart.semantics)
        semantics = chart.semantics
        self.assertIsNotNone(semantics)
        self.assertEqual(semantics.intent, "TREND")
        self.assertIsNotNone(semantics.measure)
        self.assertEqual(semantics.measure.type, "MEAN")
        self.assertIsNotNone(semantics.time)
        self.assertEqual(semantics.time.grain, "MONTH")
        self.assertIsNone(semantics.splits)


if __name__ == "__main__":
    unittest.main()
