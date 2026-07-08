import ast
import unittest
from pathlib import Path
from typing import Any, Dict, cast


class _TranslatorStub:
    @staticmethod
    def translate(_key: str, language: str | None = None, params: Dict[str, Any] | None = None, default: str | None = None) -> str:
        if default is None:
            return ""
        if params:
            return default.format(**params)
        return default


def _load_format_execution_summary():
    source_path = Path(__file__).resolve().parents[1] / "src" / "actions" / "helpers" / "visualization.py"
    source = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(source_path))

    required = {
        "SupportsModelDump",
        "_mapping_to_dict",
        "_maybe_model_dump_dict",
        "format_execution_summary",
    }

    selected = [
        node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in required
    ]

    prelude = ast.parse(
        "from typing import Any, Dict, List, Mapping, Optional, Protocol, cast, runtime_checkable"
    )
    isolated_module = ast.Module(body=[*prelude.body, *selected], type_ignores=[])
    ast.fix_missing_locations(isolated_module)

    namespace: Dict[str, Any] = {
        "translate": _TranslatorStub.translate,
    }
    exec(compile(isolated_module, filename=str(source_path), mode="exec"), namespace)
    return cast(Any, namespace["format_execution_summary"])


class VisualizationSummaryStatsTests(unittest.TestCase):
    def test_stats_only_summary_does_not_report_zero_charts(self) -> None:
        formatter = _load_format_execution_summary()
        summary = {
            "trace_id": "abc123",
            "chart_count": 0,
            "stats_count": 1,
            "stats_skipped": 0,
            "stats_errors": 0,
            "estimated_queries": 1,
            "actual_queries": 1,
            "batches": [],
        }

        msg = formatter(summary, show_normalization=False, planner_diagnostics=None, language="en")

        self.assertIn("Plan produced 1 statistical test result.", msg)
        self.assertNotIn("Plan produced 0 charts.", msg)

    def test_stats_summary_includes_skip_and_error_counts(self) -> None:
        formatter = _load_format_execution_summary()
        summary = {
            "chart_count": 1,
            "stats_count": 3,
            "stats_skipped": 1,
            "stats_errors": 1,
            "estimated_queries": 4,
            "actual_queries": 4,
            "batches": [],
        }

        msg = formatter(summary, show_normalization=False, planner_diagnostics=None, language="en")

        self.assertIn("Plan produced 1 chart.", msg)
        self.assertIn("Plan produced 3 statistical test results.", msg)
        self.assertIn("Skipped 1 statistical test result(s).", msg)
        self.assertIn("1 statistical test result(s) returned errors.", msg)


if __name__ == "__main__":
    unittest.main()
