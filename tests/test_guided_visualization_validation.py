from src.actions.guided_visualization_validation import build_guided_plan
from src.domain.langchain import schema as S


def test_build_guided_plan_maps_sex_type_groupby_to_typed_group() -> None:
    plan = build_guided_plan(
        slots={
            "metric": "DTN",
            "chart_type": "LINE",
            "group_by": "SEX_TYPE",
        },
        user_sub="user-1",
    )

    assert plan.charts is not None
    chart = plan.charts[0]
    assert chart.group_by is not None
    assert len(chart.group_by) == 1
    assert isinstance(chart.group_by[0], S.GroupBySex)


def test_build_guided_plan_maps_stroke_type_groupby_to_typed_group() -> None:
    plan = build_guided_plan(
        slots={
            "metric": "DTN",
            "chart_type": "LINE",
            "group_by": "STROKE_TYPE",
        },
        user_sub="user-1",
    )

    assert plan.charts is not None
    chart = plan.charts[0]
    assert chart.group_by is not None
    assert len(chart.group_by) == 1
    assert isinstance(chart.group_by[0], S.GroupByStrokeType)


def test_build_guided_plan_keeps_other_groupby_as_canonical_field() -> None:
    plan = build_guided_plan(
        slots={
            "metric": "DTN",
            "chart_type": "LINE",
            "group_by": "FIRST_CONTACT_PLACE",
        },
        user_sub="user-1",
    )

    assert plan.charts is not None
    chart = plan.charts[0]
    assert chart.group_by is not None
    assert len(chart.group_by) == 1
    assert isinstance(chart.group_by[0], S.GroupByCanonicalField)
    assert chart.group_by[0].field == "FIRST_CONTACT_PLACE"


def test_build_guided_plan_maps_quarter_groupby_to_time_group() -> None:
    plan = build_guided_plan(
        slots={
            "metric": "DTN",
            "chart_type": "LINE",
            "group_by": "QUARTER",
        },
        user_sub="user-1",
    )

    assert plan.charts is not None
    chart = plan.charts[0]
    assert chart.group_by is not None
    assert len(chart.group_by) == 1
    assert isinstance(chart.group_by[0], S.GroupByTime)
    assert chart.group_by[0].grain == "QUARTER"
