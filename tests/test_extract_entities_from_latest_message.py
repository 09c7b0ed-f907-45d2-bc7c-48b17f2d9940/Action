from src.actions.helpers.visualization import extract_entities_from_latest_message


def _message(entities):
    return {"entities": entities}


def test_repeated_identical_entity_value_stays_a_scalar() -> None:
    """Regression test: "make two line charts, one of male patients DTN,
    the other of female patients DTN" mentions DTN twice -- once per chart,
    not as two distinct metric answers. Turning that into metric=["DTN",
    "DTN"] previously read to the decision-stage LLM as two different
    metric candidates to choose between, producing a spurious "which
    metric?" clarification for an otherwise unambiguous request.
    """
    entities = extract_entities_from_latest_message(
        _message(
            [
                {"entity": "chart_type", "value": "LINE"},
                {"entity": "sex", "value": "MALE"},
                {"entity": "metric", "value": "DTN"},
                {"entity": "sex", "value": "FEMALE"},
                {"entity": "metric", "value": "DTN"},
            ]
        )
    )

    assert entities["metric"] == "DTN"
    assert entities["sex"] == ["MALE", "FEMALE"]


def test_repeated_distinct_entity_values_still_become_a_list() -> None:
    entities = extract_entities_from_latest_message(
        _message(
            [
                {"entity": "group_id", "value": "289"},
                {"entity": "group_id", "value": "252"},
            ]
        )
    )

    assert entities["group_id"] == ["289", "252"]


def test_repeated_distinct_values_in_an_existing_list_are_appended_once() -> None:
    entities = extract_entities_from_latest_message(
        _message(
            [
                {"entity": "group_id", "value": "289"},
                {"entity": "group_id", "value": "252"},
                {"entity": "group_id", "value": "289"},
                {"entity": "group_id", "value": "10"},
            ]
        )
    )

    assert entities["group_id"] == ["289", "252", "10"]


def test_regex_only_entity_dropped_when_diet_claims_the_same_span_differently() -> None:
    """Regression test: "Show me a bar chart of DTN grouped by age in
    10-year buckets" -- DIETClassifier correctly tags "age" as group_by only,
    but RegexEntityExtractor's SSOT lookup table also matches the literal
    word "age" as a metric (it's a legitimate metric name elsewhere), at the
    exact same character span. That regex-only reading of the same tokens
    DIET already assigned to group_by is a false positive, not a second
    metric candidate -- it previously produced metric=["DTN", "AGE"], read by
    the decision-stage LLM as genuine ambiguity for an unambiguous request.
    """
    entities = extract_entities_from_latest_message(
        _message(
            [
                {
                    "entity": "metric",
                    "value": "DTN",
                    "start": 23,
                    "end": 26,
                    "extractors": [{"extractor": "DIETClassifier"}, {"extractor": "RegexEntityExtractor"}],
                },
                {
                    "entity": "group_by",
                    "value": "AGE",
                    "start": 38,
                    "end": 41,
                    "extractors": [{"extractor": "DIETClassifier"}, {"extractor": "RegexEntityExtractor"}],
                },
                {
                    "entity": "metric",
                    "value": "AGE",
                    "start": 38,
                    "end": 41,
                    "extractors": [{"extractor": "RegexEntityExtractor"}],
                },
                {
                    "entity": "group_by",
                    "value": "YEAR",
                    "start": 48,
                    "end": 52,
                    "extractors": [{"extractor": "RegexEntityExtractor"}],
                },
            ]
        )
    )

    assert entities["metric"] == "DTN"
    # "YEAR" has no competing DIET tag at its span at all (DIET didn't tag
    # those tokens as anything), so it isn't a same-span contradiction --
    # only entities that DIET actively assigned to a *different* type at the
    # same span get dropped.
    assert entities["group_by"] == ["AGE", "YEAR"]


def test_regex_only_entity_kept_when_diet_agrees_or_is_silent() -> None:
    entities = extract_entities_from_latest_message(
        _message(
            [
                {
                    "entity": "metric",
                    "value": "DTN",
                    "start": 0,
                    "end": 3,
                    "extractors": [{"extractor": "DIETClassifier"}, {"extractor": "RegexEntityExtractor"}],
                },
                {
                    "entity": "limit",
                    "value": "10",
                    "start": 10,
                    "end": 12,
                    "extractors": [{"extractor": "RegexEntityExtractor"}],
                },
            ]
        )
    )

    assert entities["metric"] == "DTN"
    assert entities["limit"] == "10"
