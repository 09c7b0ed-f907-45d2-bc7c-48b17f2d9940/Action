from src.domain.graphql.request import DataOrigin, DistributionOptions, GraphQLQueryRequest, MetricRequest
from src.domain.graphql.ssot_enums import MetricType


def test_enum_metric_query_omits_numeric_only_kpis_and_distribution() -> None:
    req = GraphQLQueryRequest(
        metrics=[
            MetricRequest(
                metricType=MetricType("HOSPITALIZED_IN"),
                includeStats=True,
                includeDistribution=True,
                distributionOptions=DistributionOptions(binCount=8, lowerBound=0, upperBound=100),
            )
        ],
        dataOrigin=DataOrigin(providerId=[1]),
    )

    query = req.to_graphql_string()

    assert "metricId: HOSPITALIZED_IN" in query
    assert "caseCount" in query
    assert "percents" in query
    assert "normalizedPercents" in query
    assert "cohortSize" in query
    assert "normalizedCohortSize" in query

    # Numeric-only KPIs must not be requested for enum metrics.
    assert "median" not in query
    assert "mean" not in query
    assert "variance" not in query
    assert "confidenceIntervalMean" not in query
    assert "confidenceIntervalMedian" not in query
    assert "interquartileRange" not in query
    assert "quartiles" not in query

    # Distribution is numeric-only and must not be requested for enum metrics.
    assert "distribution(binCount:" not in query


def test_numeric_metric_query_includes_numeric_kpis_and_distribution() -> None:
    req = GraphQLQueryRequest(
        metrics=[
            MetricRequest(
                metricType=MetricType("DTN"),
                includeStats=True,
                includeDistribution=True,
                distributionOptions=DistributionOptions(binCount=12, lowerBound=0, upperBound=180),
            )
        ],
        dataOrigin=DataOrigin(providerId=[1]),
    )

    query = req.to_graphql_string()

    assert "metricId: DTN" in query
    assert "median" in query
    assert "mean" in query
    assert "variance" in query
    assert "confidenceIntervalMean" in query
    assert "confidenceIntervalMedian" in query
    assert "interquartileRange" in query
    assert "quartiles" in query
    assert "distribution(binCount: 12)" in query
