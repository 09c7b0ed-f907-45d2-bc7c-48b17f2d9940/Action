from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Literal, Optional, cast

from langchain_core.prompts import ChatPromptTemplate

from src.domain.langchain.schema import AnalysisPlan, AndFilter, ChartType, DataOriginSpec, DateFilter, MetricSpec, OriginScopeSpec, StatisticalTestSpec
from src.planners.langchain.llm_factory import create_chat_llm
from src.planners.langchain.pipeline import generate_analysis_plan
from src.shared import ssot_loader
from src.util import env as env_util
from src.util.logging_utils import bind_current_context, log_context

logger = logging.getLogger(__name__)


OutcomeDecision = Literal["proceed", "clarify", "reject"]


@dataclass
class VisualizationRequestOutcome:
    decision: OutcomeDecision
    reason: str
    message: Optional[str] = None
    clarification_type: Optional[str] = None
    clarification_options: List[str] = field(default_factory=lambda: cast(List[str], []))
    missing_fields: List[str] = field(default_factory=lambda: cast(List[str], []))
    plan: Optional[AnalysisPlan] = None


_ORCHESTRATOR_ENABLED = env_util.env_flag("ACTIONS_LLM_REQUEST_ORCHESTRATOR_ENABLED", default=True)
_ORCHESTRATOR_TIMEOUT_RAW = env_util.get_env("ACTIONS_LLM_REQUEST_ORCHESTRATOR_TIMEOUT_SECONDS", default="10") or "10"
_orchestrator_timeout_value = 10.0
try:
    _orchestrator_timeout_value = max(1.0, float(_ORCHESTRATOR_TIMEOUT_RAW))
except Exception:
    _orchestrator_timeout_value = 10.0
_ORCHESTRATOR_TIMEOUT_SECONDS = _orchestrator_timeout_value

_ORCHESTRATOR_TEMPERATURE_RAW = env_util.get_env("ACTIONS_LLM_REQUEST_ORCHESTRATOR_TEMPERATURE", default="0") or "0"
_orchestrator_temperature_value = 0.0
try:
    _orchestrator_temperature_value = float(_ORCHESTRATOR_TEMPERATURE_RAW)
except Exception:
    _orchestrator_temperature_value = 0.0
_ORCHESTRATOR_TEMPERATURE = _orchestrator_temperature_value

_ORCHESTRATOR_FAIL_OPEN = env_util.env_flag("ACTIONS_LLM_REQUEST_ORCHESTRATOR_FAIL_OPEN", default=False)

_TEMPORAL_MISSING_FIELDS = {
    "time",
    "time_scope",
    "time_range",
    "time_window",
    "time_period",
    "date",
    "date_range",
    "date_window",
    "date_period",
    "period",
    "timeframe",
    "time_frame",
    "window",
    "range",
}

_STAT_TEST_ENTITY_KEYS = {
    "statistical_test_type",
    "statistical_tests",
    "test_type",
    "test_types",
}

_STATISTICAL_COHORT_ENTITY_KEYS = {
    "provider_id",
    "provider_group_id",
    "provider_name",
    "provider_group_name",
    "country_code",
    "country_average",
    "sex",
    "stroke_type",
    "age",
    "mine",
}

_STAT_TEST_KEYWORDS = (
    "statistical test",
    "mann-whitney",
    "mann whitney",
    "compare",
    "significant",
    "significance",
    "difference between",
)

_PROVIDER_GROUP_HINTS = (
    "provider group",
    "provider-group",
    "cohort a is group",
    "cohort b is group",
)

_PROVIDER_HINTS = (
    "provider ",
    "cohort a is provider",
    "cohort b is provider",
)

_DECISION_PROMPT = ChatPromptTemplate.from_messages(  # type: ignore[attr-defined]
    [
        (
            "system",
            """
You are the triage stage for a clinical analytics visualization assistant.
Return strict JSON only:
{{
  "decision": "proceed" | "clarify" | "reject",
  "reason": "short_snake_case_reason",
  "missing_fields": string[] | null,
  "clarification_type": string | null,
  "clarification_options": string[] | null,
  "message": string | null
}}

Rules:
- For chart requests, required fields are metric and chart_type.
- For statistical-test-only requests, required fields are metric and statistical_test_type; chart_type is optional and should not trigger clarification.
- If required fields for the detected intent are present, return decision="proceed" and message=null.
- If one required field is missing, return decision="clarify", put the missing field in missing_fields, and write one concise question in message (under 25 words).
- If out of scope, return decision="reject" with a short message.
- Never ask the user to clarify or provide time_scope, time_range, grouping_dimension, sex, or stroke_type — these are optional and should be accepted if present, not rejected.
- Prefer resolving metrics from VALID_METRIC_CANDIDATES_JSON before asking.
- When a date entity contains only a year (e.g. "2026"), expand it to a full-year range: two DateFilters — operator GE with value "{{year}}-01-01" AND operator LE with value "{{year}}-12-31".
- Do not include markdown or prose outside JSON.
        """.strip(),
        ),
        (
            "user",
            "USER_LANGUAGE: {language}\nUSER_QUESTION: {question}\nCONVERSATION_HISTORY_JSON: {conversation_history_json}\nENTITIES_JSON: {entities_json}\nVALID_METRIC_CANDIDATES_JSON: {metric_candidates_json}\nVALID_CHART_TYPES_JSON: {chart_types_json}",
        ),
    ]
)

_llm: Optional[Any] = None
_llm_lock = Lock()


def _get_llm() -> Optional[Any]:
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is not None:
            return _llm
        try:
            _llm = create_chat_llm(temperature=_ORCHESTRATOR_TEMPERATURE)
        except Exception:
            logger.exception(
                "Failed to initialize LLM request orchestrator",
                extra={
                    "log_context": {
                        "event": "orchestrator.llm_init.failed",
                        "operation": "_get_llm",
                        "outcome": "failure",
                    }
                },
            )
            _llm = None
    return _llm


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in cast(List[Any], content):
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                maybe_text = cast(Dict[str, Any], item).get("text")
                if maybe_text is not None:
                    chunks.append(str(maybe_text))
        if chunks:
            return "\n".join(chunks)

    return str(response)


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()

    if not (candidate.startswith("{") and candidate.endswith("}")):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model output does not contain JSON object")
        candidate = candidate[start : end + 1]

    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")
    return cast(Dict[str, Any], parsed)


def _invoke_chain(chain: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(bind_current_context(chain.invoke), payload)
        try:
            response = future.result(timeout=_ORCHESTRATOR_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Orchestrator timed out after {_ORCHESTRATOR_TIMEOUT_SECONDS:.1f}s") from exc

    return _extract_json_object(_extract_text(response))


def _metric_candidates(question: str, limit: int = 8) -> List[str]:
    normalized = ssot_loader.normalize_metric_text_key(question)
    if not normalized:
        return []

    lookup = ssot_loader.get_metric_text_lookup()
    if normalized in lookup:
        entry = lookup[normalized]
        canonical = entry.get("canonical")
        if isinstance(canonical, str) and canonical.strip():
            return [canonical.strip()]
        return [str(entry)]

    out: List[str] = []
    for key, entry in lookup.items():
        if normalized not in key and key not in normalized:
            continue
        canonical = entry.get("canonical")
        if isinstance(canonical, str) and canonical.strip() and canonical not in out:
            out.append(canonical.strip())
        elif str(entry).strip() and str(entry).strip() not in out:
            out.append(str(entry).strip())
        if len(out) >= limit:
            break
    return out


def _chart_types() -> List[str]:
    try:
        return [str(member) for member in ChartType]
    except Exception:
        return ["LINE", "BAR", "AREA", "SCATTER", "HISTOGRAM", "BOX", "VIOLIN"]


def _coerce_missing_fields(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in cast(List[Any], raw):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _coerce_options(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in cast(List[Any], raw):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _is_missing_chart_type_only(missing_fields: List[str]) -> bool:
    normalized = [field.strip().lower() for field in missing_fields if field and field.strip()]
    return bool(normalized) and all(field == "chart_type" for field in normalized)


def _has_statistical_test_signal(question: str, entities: Dict[str, Any]) -> bool:
    for key in _STAT_TEST_ENTITY_KEYS:
        value = entities.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list):
            value_list = cast(List[Any], value)
            if any(isinstance(item, str) and item.strip() for item in value_list):
                return True

    question_norm = (question or "").strip().lower()
    if not question_norm:
        return False
    return any(keyword in question_norm for keyword in _STAT_TEST_KEYWORDS)


def _extract_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, list):
        out: List[str] = []
        for item in cast(List[Any], value):
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _extract_provider_group_ids(entities: Dict[str, Any]) -> List[int]:
    raw_values = _extract_string_list(entities.get("provider_group_id"))
    out: List[int] = []
    for token in raw_values:
        for match in re.findall(r"\d+", token):
            try:
                group_id = int(match)
            except Exception:
                continue
            if group_id > 0 and group_id not in out:
                out.append(group_id)
    return out


def _extract_provider_ids(entities: Dict[str, Any]) -> List[int]:
    raw_values = _extract_string_list(entities.get("provider_id"))
    out: List[int] = []
    for token in raw_values:
        for match in re.findall(r"\d+", token):
            try:
                provider_id = int(match)
            except Exception:
                continue
            if provider_id > 0 and provider_id not in out:
                out.append(provider_id)
    return out


def _extract_metric_code(entities: Dict[str, Any]) -> Optional[str]:
    metrics = _extract_string_list(entities.get("metric"))
    if not metrics:
        return None
    return metrics[0].upper()


def _extract_date_bounds(entities: Dict[str, Any]) -> Optional[tuple[str, str]]:
    date_values = [d for d in _extract_string_list(entities.get("date")) if re.match(r"^\d{4}-\d{2}-\d{2}$", d)]
    if len(date_values) < 2:
        return None
    ordered = sorted(set(date_values))
    if len(ordered) < 2:
        return None
    return ordered[0], ordered[-1]


def _extract_statistical_cohort_tokens(entities: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []

    for key in _STATISTICAL_COHORT_ENTITY_KEYS:
        value = entities.get(key)
        if value is None:
            continue

        if key == "mine" and bool(value):
            tokens.append("mine")
            continue

        if isinstance(value, str):
            token = value.strip()
            if token:
                tokens.append(token)
            continue

        if isinstance(value, list):
            for item in cast(List[Any], value):
                if isinstance(item, str) and item.strip():
                    tokens.append(item.strip())

    normalized: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        marker = token.strip().lower()
        if not marker or marker in seen:
            continue
        seen.add(marker)
        normalized.append(token.strip())
    return normalized


def _question_mentions_provider_group(question: str) -> bool:
    low = (question or "").strip().lower()
    if not low:
        return False
    return any(hint in low for hint in _PROVIDER_GROUP_HINTS)


def _question_mentions_provider(question: str) -> bool:
    low = (question or "").strip().lower()
    if not low:
        return False
    if _question_mentions_provider_group(low):
        return False
    return any(hint in low for hint in _PROVIDER_HINTS)


def _validate_statistical_entity_readiness(question: str, entities: Dict[str, Any]) -> Optional[VisualizationRequestOutcome]:
    if not _has_statistical_test_signal(question, entities):
        return None

    provider_group_ids = _extract_provider_group_ids(entities)
    provider_ids = _extract_provider_ids(entities)
    mentions_provider_group = _question_mentions_provider_group(question)
    mentions_provider = _question_mentions_provider(question)

    if mentions_provider_group and len(provider_group_ids) < 2:
        return VisualizationRequestOutcome(
            decision="clarify",
            reason="missing_provider_group_cohorts",
            message=(
                "Your request mentions provider groups, but I do not have two provider-group cohorts. "
                "Please provide provider group A and provider group B explicitly."
            ),
            clarification_type="analysis_plan",
            clarification_options=[],
            missing_fields=["provider_group_id"],
        )

    if mentions_provider and len(provider_ids) < 2:
        return VisualizationRequestOutcome(
            decision="clarify",
            reason="missing_provider_cohorts",
            message=(
                "Your request mentions providers, but I do not have two provider cohorts. "
                "Please provide provider A and provider B explicitly."
            ),
            clarification_type="analysis_plan",
            clarification_options=[],
            missing_fields=["provider_id"],
        )

    cohort_tokens = _extract_statistical_cohort_tokens(entities)
    if len(cohort_tokens) < 2:
        return VisualizationRequestOutcome(
            decision="clarify",
            reason="missing_statistical_cohorts",
            message=(
                "I can run this statistical test only with two explicit cohorts. "
                "Please provide cohort A and cohort B (for example two providers or provider groups)."
            ),
            clarification_type="analysis_plan",
            clarification_options=[],
            missing_fields=["statistical_cohorts"],
        )

    return None


def _coerce_origin_scope(scope_type: str, value: Any = None, label: Optional[str] = None, country_code: Optional[str] = None) -> OriginScopeSpec:
    payload: Dict[str, Any] = {"scopeType": scope_type}
    if value is not None:
        payload["value"] = value
    if label is not None:
        payload["label"] = label
    if country_code is not None:
        payload["countryCode"] = country_code
    return OriginScopeSpec.model_validate(payload)


def _extract_semantic_scopes(entities: Dict[str, Any]) -> List[OriginScopeSpec]:
    scopes: List[OriginScopeSpec] = []

    if entities.get("mine"):
        scopes.append(_coerce_origin_scope("mine", label="mine"))

    provider_names = _extract_string_list(entities.get("provider_name"))
    for provider_name in provider_names:
        scopes.append(_coerce_origin_scope("provider_name", value=provider_name, label=provider_name))

    provider_group_names = _extract_string_list(entities.get("provider_group_name"))
    for provider_group_name in provider_group_names:
        scopes.append(_coerce_origin_scope("provider_group_name", value=provider_group_name, label=provider_group_name))

    country_codes = _extract_string_list(entities.get("country_code"))
    for country_code in country_codes:
        scopes.append(_coerce_origin_scope("country_code", value=country_code, country_code=country_code, label=country_code))

    if entities.get("country_average"):
        country_average_values = _extract_string_list(entities.get("country_average"))
        if country_average_values:
            for country_code in country_average_values:
                scopes.append(_coerce_origin_scope("country_average", value=country_code, country_code=country_code, label=country_code))
        else:
            scopes.append(_coerce_origin_scope("country_average", label="country average"))

    return scopes


def _build_deterministic_statistical_plan(
    question: str,
    entities: Dict[str, Any],
) -> Optional[AnalysisPlan]:
    if not _has_statistical_test_signal(question, entities):
        return None

    metric = _extract_metric_code(entities)
    provider_group_ids = _extract_provider_group_ids(entities)
    provider_ids = _extract_provider_ids(entities)
    semantic_scopes = _extract_semantic_scopes(entities)
    mentions_provider_group = _question_mentions_provider_group(question)
    mentions_provider = _question_mentions_provider(question)
    bounds = _extract_date_bounds(entities)

    if metric is None or bounds is None:
        return None

    start_date, end_date = bounds
    metrics: List[MetricSpec]

    if len(semantic_scopes) >= 2:
        metrics = [
            MetricSpec(metric=metric, originScope=semantic_scopes[0]),
            MetricSpec(metric=metric, originScope=semantic_scopes[1]),
        ]
    elif mentions_provider_group and len(provider_group_ids) >= 2:
        cohort_a = provider_group_ids[0]
        cohort_b = provider_group_ids[1]
        metrics = [
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerGroupId=[cohort_a])),
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerGroupId=[cohort_b])),
        ]
    elif mentions_provider and len(provider_ids) >= 2:
        cohort_a = provider_ids[0]
        cohort_b = provider_ids[1]
        metrics = [
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerId=[cohort_a])),
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerId=[cohort_b])),
        ]
    elif len(provider_group_ids) >= 2:
        cohort_a = provider_group_ids[0]
        cohort_b = provider_group_ids[1]
        metrics = [
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerGroupId=[cohort_a])),
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerGroupId=[cohort_b])),
        ]
    elif len(provider_ids) >= 2:
        cohort_a = provider_ids[0]
        cohort_b = provider_ids[1]
        metrics = [
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerId=[cohort_a])),
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerId=[cohort_b])),
        ]
    else:
        return None

    return AnalysisPlan(
        charts=None,
        statistical_tests=[
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                metrics=metrics,
                filters=AndFilter(
                    and_=[
                        DateFilter(operator="GE", value=start_date),
                        DateFilter(operator="LE", value=end_date),
                    ]
                ),
            )
        ],
    )


def _has_distinct_metric_cohorts(metric_a: Optional[Any], metric_b: Optional[Any]) -> bool:
    if metric_a is None or metric_b is None:
        return False

    metric_a_origin = getattr(metric_a, "data_origin", None)
    metric_b_origin = getattr(metric_b, "data_origin", None)
    metric_a_scope = getattr(metric_a, "origin_scope", None)
    metric_b_scope = getattr(metric_b, "origin_scope", None)

    if metric_a_origin is not None and metric_b_origin is not None:
        origin_a_payload = metric_a_origin.model_dump(by_alias=True, exclude_none=True)
        origin_b_payload = metric_b_origin.model_dump(by_alias=True, exclude_none=True)
        if origin_a_payload != origin_b_payload:
            return True

    if metric_a_scope is not None and metric_b_scope is not None:
        scope_a_payload = metric_a_scope.model_dump(by_alias=True, exclude_none=True)
        scope_b_payload = metric_b_scope.model_dump(by_alias=True, exclude_none=True)
        if scope_a_payload != scope_b_payload:
            return True

    if metric_a_origin is None or metric_b_origin is None:
        return False

    label_a = getattr(metric_a_scope, "label", None)
    label_b = getattr(metric_b_scope, "label", None)
    if isinstance(label_a, str) and isinstance(label_b, str):
        return bool(label_a.strip() and label_b.strip() and label_a.strip() != label_b.strip())

    return False


def _validate_statistical_plan_readiness(plan: AnalysisPlan) -> Optional[VisualizationRequestOutcome]:
    tests = list(plan.statistical_tests or [])
    if not tests:
        return None

    for test in tests:
        test_type = (test.test_type or "").upper().strip()
        if test_type != "MANN_WHITNEY_U_TEST":
            continue

        metrics = list(test.metrics or [])
        if len(metrics) < 2:
            return VisualizationRequestOutcome(
                decision="clarify",
                reason="missing_statistical_cohorts",
                message=(
                    "I can run Mann-Whitney U only with two explicit cohorts. "
                    "Please provide cohort A and cohort B to compare."
                ),
                clarification_type="analysis_plan",
                clarification_options=[],
                missing_fields=["statistical_cohorts"],
            )

        if not _has_distinct_metric_cohorts(metrics[0], metrics[1]):
            return VisualizationRequestOutcome(
                decision="clarify",
                reason="missing_statistical_cohorts",
                message=(
                    "Mann-Whitney U requires two distinct cohorts. Please provide "
                    "different cohort filters or scopes for each comparison group."
                ),
                clarification_type="analysis_plan",
                clarification_options=[],
                missing_fields=["statistical_cohorts"],
            )

    return None


def _decision_stage(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str],
    conversation_history: Optional[List[str]] = None,
) -> VisualizationRequestOutcome:
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable")

    chain = _DECISION_PROMPT | llm
    payload = {
        "language": (language or "en").strip() or "en",
        "question": question or "",
        "entities_json": json.dumps(entities or {}, ensure_ascii=False),
        "metric_candidates_json": json.dumps(_metric_candidates(question or ""), ensure_ascii=False),
        "chart_types_json": json.dumps(_chart_types(), ensure_ascii=False),
        "conversation_history_json": json.dumps(conversation_history or [], ensure_ascii=False),
    }
    parsed = _invoke_chain(chain, payload)
    logger.info("Decision stage raw response: %s", parsed)

    decision_raw = str(parsed.get("decision") or "").strip().lower()
    if decision_raw not in {"proceed", "clarify", "reject"}:
        raise ValueError("Invalid decision from decision stage")

    reason = str(parsed.get("reason") or "").strip() or "llm_orchestrator"
    missing_fields = _coerce_missing_fields(parsed.get("missing_fields"))
    clarification_type = parsed.get("clarification_type")
    clarification_options = _coerce_options(parsed.get("clarification_options"))
    llm_message_raw = parsed.get("message")
    llm_message = llm_message_raw.strip() if isinstance(llm_message_raw, str) and llm_message_raw.strip() else None

    outcome = VisualizationRequestOutcome(
        decision=cast(OutcomeDecision, decision_raw),
        reason=reason,
        message=llm_message,
        clarification_type=clarification_type.strip() if isinstance(clarification_type, str) and clarification_type.strip() else None,
        clarification_options=clarification_options,
        missing_fields=missing_fields,
    )

    # Deterministic safeguard: do not block statistical-test requests on chart_type.
    if (
        outcome.decision == "clarify"
        and _is_missing_chart_type_only(outcome.missing_fields)
        and _has_statistical_test_signal(question, entities)
    ):
        return VisualizationRequestOutcome(
            decision="proceed",
            reason="statistical_test_without_chart_type",
            message=None,
            clarification_type=None,
            clarification_options=[],
            missing_fields=[],
        )

    return outcome


def orchestrate_visualization_request(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str] = None,
    trace_id: Optional[str] = None,
    max_retries: int = 2,
    include_plan: bool = True,
    conversation_history: Optional[List[str]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> VisualizationRequestOutcome:
    with log_context(trace_id=trace_id or "", orchestrator_include_plan=include_plan):
        if not _ORCHESTRATOR_ENABLED:
            if not include_plan:
                return VisualizationRequestOutcome(decision="proceed", reason="orchestrator_disabled")
            plan = generate_analysis_plan(
                question=question,
                entities=entities,
                language=language,
                max_retries=max_retries,
                debug=False,
                trace_id=trace_id,
                progress_cb=progress_cb,
            )
            return VisualizationRequestOutcome(decision="proceed", reason="orchestrator_disabled", plan=plan)

        def report(message: str) -> None:
            if progress_cb is not None:
                progress_cb(message)

        try:
            report("Analyzing request intent and feasibility")
            logger.info("Orchestrator input - question: %s, entities: %s", question, entities)

            stats_entity_validation = _validate_statistical_entity_readiness(question, entities)
            if stats_entity_validation is not None:
                logger.info(
                    "Orchestrator clarification: %s",
                    stats_entity_validation.reason,
                )
                return stats_entity_validation

            stage1 = _decision_stage(question, entities, language, conversation_history=conversation_history)
            logger.info(
                "Orchestrator decision: %s, message: %s, missing: %s",
                stage1.decision,
                stage1.message,
                stage1.missing_fields,
            )

            if stage1.decision == "reject":
                if stage1.message is None:
                    stage1.message = "This request is outside the visualization flow."
                return stage1

            if stage1.decision == "clarify":
                return stage1

            if not include_plan:
                return VisualizationRequestOutcome(
                    decision="proceed",
                    reason=stage1.reason or "sufficient_information",
                )

            report("Generating visualization plan")
            planner_question = question
            if conversation_history:
                cleaned_history = [item.strip() for item in conversation_history if item.strip()]
                if cleaned_history:
                    joined = "\n".join(f"- {item}" for item in cleaned_history)
                    planner_question = f"Conversation context (oldest to newest user turns):\n{joined}\n\nCurrent request to fulfill:\n{question}"

            deterministic_plan = _build_deterministic_statistical_plan(planner_question, entities)
            if deterministic_plan is not None:
                stats_validation = _validate_statistical_plan_readiness(deterministic_plan)
                if stats_validation is not None:
                    return stats_validation
                return VisualizationRequestOutcome(
                    decision="proceed",
                    reason="deterministic_statistical_plan",
                    plan=deterministic_plan,
                )

            plan = generate_analysis_plan(
                question=planner_question,
                entities=entities,
                language=language,
                max_retries=max_retries,
                debug=False,
                trace_id=trace_id,
                progress_cb=progress_cb,
            )

            stats_validation = _validate_statistical_plan_readiness(plan)
            if stats_validation is not None:
                return stats_validation

            return VisualizationRequestOutcome(
                decision="proceed",
                reason=stage1.reason or "sufficient_information",
                plan=plan,
            )
        except Exception:
            logger.exception(
                "Visualization request orchestration failed",
                extra={
                    "log_context": {
                        "event": "orchestrator.request.failed",
                        "operation": "orchestrate_visualization_request",
                        "outcome": "failure",
                        "include_plan": include_plan,
                        "fail_open_enabled": _ORCHESTRATOR_FAIL_OPEN,
                    }
                },
            )
            if _ORCHESTRATOR_FAIL_OPEN and include_plan:
                logger.warning(
                    "Orchestrator failed; activating direct-plan fallback",
                    extra={
                        "log_context": {
                            "event": "orchestrator.request.fail_open_fallback",
                            "operation": "orchestrate_visualization_request",
                            "outcome": "degraded",
                        }
                    },
                )
                report("Orchestration fallback: generating plan directly")
                plan = generate_analysis_plan(
                    question=question,
                    entities=entities,
                    language=language,
                    max_retries=max_retries,
                    debug=False,
                    trace_id=trace_id,
                    progress_cb=progress_cb,
                )
                return VisualizationRequestOutcome(
                    decision="proceed",
                    reason="orchestrator_fallback_to_plan",
                    plan=plan,
                )

            return VisualizationRequestOutcome(
                decision="clarify",
                reason="orchestrator_failed",
                message=(
                    "I could not produce a valid statistical plan from that request. "
                    "Please provide a metric, two explicit cohorts (for example two provider groups), "
                    "and explicit date bounds."
                    if _has_statistical_test_signal(question, entities)
                    else "I need a bit more detail before I can continue."
                ),
            )
