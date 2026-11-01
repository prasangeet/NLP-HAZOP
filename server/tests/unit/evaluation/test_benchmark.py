"""Tests for scenario-level benchmark splitting and metrics."""

from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RecordType,
    RelationshipOrigin,
)
from scripts.run_benchmark import (
    grouped_kfold_split,
    score_predictions,
    stable_split,
    strip_target_labels,
)


def make_record(scenario_id: str) -> HazopRecord:
    return HazopRecord(
        scenario_id=scenario_id,
        plant_unit="Storage",
        equipment="T-301",
        deviation="High level",
        causes=("Valve fails open",),
        consequences=("Tank overflow",),
        safeguards=("Overfill protection",),
        recommendations=("Review procedure",),
        record_type=RecordType.SOURCE_RECONSTRUCTED,
        causal_relations=(
            CausalRelation(
                subject="Valve fails open",
                predicate="causes_deviation",
                object="High level",
                origin=RelationshipOrigin.RECONSTRUCTED,
                evidence="Row-level association.",
            ),
        ),
    )


def test_split_is_scenario_disjoint_and_deterministic() -> None:
    records = tuple(make_record(f"S-{index:03}") for index in range(20))
    train_a, test_a = stable_split(records, test_fraction=0.2, seed=42)
    train_b, test_b = stable_split(records, test_fraction=0.2, seed=42)

    assert [r.scenario_id for r in train_a] == [
        r.scenario_id for r in train_b
    ]
    assert [r.scenario_id for r in test_a] == [
        r.scenario_id for r in test_b
    ]
    assert {r.scenario_id for r in train_a}.isdisjoint(
        {r.scenario_id for r in test_a}
    )
    assert len(test_a) == 4


def test_grouped_kfold_is_deterministic_disjoint_and_exhaustive() -> None:
    records = tuple(
        record
        for index in range(12)
        for record in (
            make_record(f"S-{index:03}"),
            make_record(f"S-{index:03}"),
        )
    )
    folds_a = grouped_kfold_split(records, n_splits=4, seed=42)
    folds_b = grouped_kfold_split(records, n_splits=4, seed=42)

    assert [
        [r.scenario_id for r in test] for _, test in folds_a
    ] == [
        [r.scenario_id for r in test] for _, test in folds_b
    ]

    held_out_ids = []
    for train, test in folds_a:
        train_ids = {record.scenario_id for record in train}
        test_ids = {record.scenario_id for record in test}
        assert train_ids.isdisjoint(test_ids)
        assert len(test_ids) == 3
        held_out_ids.extend(test_ids)

    assert sorted(held_out_ids) == [f"S-{index:03}" for index in range(12)]


def test_grouped_kfold_rejects_more_folds_than_scenarios() -> None:
    records = (make_record("S-001"),)
    try:
        grouped_kfold_split(records, n_splits=2, seed=42)
    except ValueError as exc:
        assert "unique scenarios" in str(exc)
    else:
        raise AssertionError("Expected grouped_kfold_split to reject one scenario")


def test_held_out_target_is_stripped_of_all_answer_fields() -> None:
    target = strip_target_labels(make_record("TEST-001"))

    assert target.causes == ()
    assert target.effects == ()
    assert target.consequences == ()
    assert target.safeguards == ()
    assert target.recommendations == ()
    assert target.causal_relations == ()
    assert target.equipment == "T-301"
    assert target.deviation == "High level"


def test_metrics_report_recovery_coverage_and_abstention() -> None:
    cases = [
        {
            "gold_values": ["Tank overflow", "Loss of containment"],
            "predictions": [
                {
                    "value": "Tank overflow",
                    "origin": "retrieval_inferred",
                    "evidence": "row evidence",
                },
                {
                    "value": "Wrong endpoint",
                    "origin": "retrieval_inferred",
                    "evidence": "row evidence",
                },
            ],
            "latency_ms": 10.0,
        },
        {
            "gold_values": ["Pump dry-running"],
            "predictions": [],
            "latency_ms": 5.0,
        },
    ]

    metrics = score_predictions(cases)

    assert metrics["n_evaluated"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["abstention_rate"] == 0.5
    assert metrics["hit_at_1"] == 0.5
    assert metrics["micro_precision"] == 0.5
    assert metrics["micro_recall"] == 1 / 3
    assert metrics["cost_usd"] is None
