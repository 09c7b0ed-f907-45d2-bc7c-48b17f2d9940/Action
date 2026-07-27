import unittest
from typing import Any, Dict, List, Optional

from src.actions.helpers.metric import extract_metric


class _FakeTracker:
    def __init__(self, entities: Optional[List[Dict[str, Any]]] = None, slots: Optional[Dict[str, Any]] = None) -> None:
        self.latest_message = {"entities": entities or []}
        self._slots = slots or {}

    def get_slot(self, name: str) -> Any:
        return self._slots.get(name)


class ExtractMetricTests(unittest.TestCase):
    def test_reads_metric_entity_from_latest_message(self) -> None:
        tracker = _FakeTracker(entities=[{"entity": "metric", "value": "DTN"}])
        self.assertEqual(extract_metric(tracker), "DTN")

    def test_ignores_legacy_kpi_entity_name(self) -> None:
        # The "metric" entity is what SSOTCanonicalizer and every current NLU
        # example actually produce; a stray "kpi"-tagged entity should not
        # be picked up.
        tracker = _FakeTracker(entities=[{"entity": "kpi", "value": "DTN"}])
        self.assertIsNone(extract_metric(tracker))

    def test_falls_back_to_metric_slot(self) -> None:
        tracker = _FakeTracker(entities=[], slots={"metric": "DIDO"})
        self.assertEqual(extract_metric(tracker), "DIDO")

    def test_returns_none_when_nothing_available(self) -> None:
        tracker = _FakeTracker()
        self.assertIsNone(extract_metric(tracker))


if __name__ == "__main__":
    unittest.main()
