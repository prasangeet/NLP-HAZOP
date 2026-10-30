"""Tests for the T-301 and DWSIM CSV dataset loader."""

from collections import Counter
from pathlib import Path

from app.domain.entities.hazop_record import (
    RecordType,
    RelationshipOrigin,
    ValidationStatus,
)
from app.infrastructure.datasets.hazop_csv_loader import HazopCsvLoader

ROOT = Path(__file__).resolve().parents[5]


def test_loads_t301_with_reconstructed_and_synthetic_provenance() -> None:
    path = ROOT / "data/raw/t301/T301_HAZOP_CCPS2001_style.csv"
    records = HazopCsvLoader().load(path, "t301")

    assert len(records) == 94
    counts = Counter(record.record_type for record in records)
    assert counts[RecordType.SOURCE_RECONSTRUCTED] == 40
    assert counts[RecordType.SYNTHETIC] == 54
    assert all(
        record.validation_status == ValidationStatus.UNREVIEWED
        for record in records
    )

    reconstructed = next(
        record
        for record in records
        if record.record_type == RecordType.SOURCE_RECONSTRUCTED
    )
    assert reconstructed.source is not None
    assert reconstructed.source.document_id == "single-2020-t301"
    assert all(
        relation.origin == RelationshipOrigin.RECONSTRUCTED
        for relation in reconstructed.causal_relations
    )


def test_loads_dwsim_as_synthetic_and_preserves_simulation_effect_node() -> None:
    path = ROOT / "data/raw/dwsim/synthetic_hazop_D001_D030.csv"
    records = HazopCsvLoader().load(path, "dwsim-synthetic")

    assert len(records) == 30
    assert all(record.record_type == RecordType.SYNTHETIC for record in records)
    assert all(
        record.validation_status == ValidationStatus.UNREVIEWED
        for record in records
    )

    first = next(record for record in records if record.scenario_id == "D001")
    assert first.safeguards == ()
    assert first.causes
    assert first.consequences
    assert any(
        edge.predicate == "leads_to"
        for edge in first.causal_relations
    )
    assert any(
        edge.predicate == "leads_to_consequence"
        for edge in first.causal_relations
    )
    assert all(
        edge.origin == RelationshipOrigin.SYNTHETIC
        for edge in first.causal_relations
    )
    assert "Review required" in first.remarks


def test_missing_safeguard_placeholder_is_not_loaded_as_a_safeguard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "placeholder.csv"
    path.write_text(
        "Scenario_ID,Equipment,Deviation,Causes,Consequences,"
        "Existing Controls / Safeguards,Recommendations / Candidate Safeguards,"
        "Evidence Source,Validation Status\n"
        "S1,T-1,High temperature,Controller failure,Off-spec product,"
        "NOT_STATED — no existing plant safeguards were provided,"
        "High-temperature alarm,Synthetic_HAZOP,Review required\n",
        encoding="utf-8",
    )

    record = HazopCsvLoader().load(path, "synthetic-test")[0]

    assert record.safeguards == ()
    assert record.recommendations == ("High-temperature alarm",)
    assert record.record_type == RecordType.SYNTHETIC
