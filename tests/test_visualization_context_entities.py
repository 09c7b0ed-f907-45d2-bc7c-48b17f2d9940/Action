import ast
import json
from pathlib import Path
from typing import Any, Dict, List, cast


def _load_merge_latest_with_thread_entities():
    source_path = Path(__file__).resolve().parents[1] / "src" / "actions" / "actions" / "visualization_action.py"
    source = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(source_path))

    required = {
        "_extract_intent_name_from_user_event",
        "_merge_entities",
        "_extract_entities_from_user_event",
        "_collect_visualization_thread_entities",
        "_dedupe_list_values",
        "merge_latest_with_thread_entities",
        "_extract_bot_custom_payload",
        "_is_visualization_payload",
        "_event_has_visualization_signal",
        "_find_latest_visualization_anchor_user_ordinal",
    }

    selected = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name in required
    ]

    isolated_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {
        "json": json,
        "cast": cast,
        "_VISUALIZATION_THREAD_INTENTS": {
            "generate_visualization",
            "update_visualization",
            "clarify_visualization",
        },
        "_VISUALIZATION_PLAN_TYPE": "visualization_plan",
        "_VISUALIZATION_RESPONSE_SCHEMA_VERSION": 1,
    }
    exec(compile(isolated_module, filename=str(source_path), mode="exec"), namespace)
    return namespace["merge_latest_with_thread_entities"]


def _user_event(text: str, intent: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "event": "user",
        "text": text,
        "parse_data": {
            "intent": {"name": intent},
            "entities": entities,
        },
    }


def test_merge_latest_with_thread_entities_merges_visualization_thread() -> None:
    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    events = [
        _user_event(
            "Show DTN by month for past year split by sex",
            "generate_visualization",
            [
                {"entity": "chart_type", "value": "LINE"},
                {"entity": "metric", "value": "DTN"},
                {"entity": "time_grain", "value": "month"},
                {"entity": "split", "value": "sex"},
                {"entity": "date", "value": "past year"},
            ],
        ),
        _user_event(
            "line chart",
            "clarify_visualization",
            [{"entity": "chart_type", "value": "LINE"}],
        ),
        _user_event(
            "DTN",
            "clarify_visualization",
            [{"entity": "metric", "value": "DTN"}],
        ),
    ]

    extracted = merge_latest_with_thread_entities(
        latest_entities={"metric": "DTN"},
        events=events,
        fallback_limit=12,
    )

    assert extracted["metric"] == "DTN"
    assert extracted["chart_type"] == "LINE"
    assert extracted["date"] == "past year"
    assert extracted["time_grain"] == "month"
    assert extracted["split"] == "sex"


def test_merge_latest_with_thread_entities_prefers_latest_value() -> None:
    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    events = [
        _user_event(
            "Show AGE as line chart",
            "generate_visualization",
            [
                {"entity": "metric", "value": "AGE"},
                {"entity": "chart_type", "value": "LINE"},
            ],
        ),
        _user_event(
            "DTN",
            "clarify_visualization",
            [{"entity": "metric", "value": "DTN"}],
        ),
    ]

    merged = merge_latest_with_thread_entities(
        latest_entities={"metric": "DTN"},
        events=events,
        fallback_limit=12,
    )

    assert merged["metric"] == "DTN"
    assert merged["chart_type"] == "LINE"


def test_merge_latest_with_thread_entities_keeps_latest_provider_scope_keys() -> None:
    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    events = [
        _user_event(
            "Run Mann-Whitney for old provider groups",
            "generate_visualization",
            [
                {"entity": "provider_group_id", "value": "provider group 2825"},
                {"entity": "provider_group_id", "value": "provider group 3001"},
                {"entity": "date", "value": "2023-01-01"},
                {"entity": "date", "value": "2023-12-31"},
            ],
        ),
    ]

    merged = merge_latest_with_thread_entities(
        latest_entities={
            "provider_name": ["Aalborg University - Hospital", "Kronborg Castle Hospital"],
            "date": ["2024-01-01", "2026-12-31"],
            "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
        },
        events=events,
        fallback_limit=12,
    )

    assert merged["provider_name"] == ["Aalborg University - Hospital", "Kronborg Castle Hospital"]
    assert merged["date"] == ["2024-01-01", "2026-12-31"]
    assert merged["statistical_test_type"] == ["MANN_WHITNEY_U_TEST"]


def test_merge_latest_with_thread_entities_keeps_latest_provider_group_ids() -> None:
    merge_latest_with_thread_entities = _load_merge_latest_with_thread_entities()
    events = [
        _user_event(
            "Run Mann-Whitney for old provider groups",
            "generate_visualization",
            [
                {"entity": "provider_group_id", "value": "provider group 2825"},
                {"entity": "provider_group_id", "value": "provider group 3001"},
            ],
        ),
    ]

    merged = merge_latest_with_thread_entities(
        latest_entities={
            "provider_group_id": ["provider group 279", "provider group 280"],
            "date": ["2024-01-01", "2026-12-31"],
        },
        events=events,
        fallback_limit=12,
    )

    assert merged["provider_group_id"] == ["provider group 279", "provider group 280"]
    assert merged["date"] == ["2024-01-01", "2026-12-31"]
