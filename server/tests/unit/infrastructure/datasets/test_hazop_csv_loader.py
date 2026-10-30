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


def test_safeguards_parsing_preserves_parentheses_with_commas_and_semicolons(
    tmp_path: Path,
) -> None:
    path = tmp_path / "safeguards_parens.csv"
    path.write_text(
        "Scenario_ID,Equipment,Deviation,Existing_Safeguards\n"
        "S1,T-301,High level,\"CollectingBasin, OverfillProtection "
        "(reported/ontology safeguard concepts, actual installation status requires source interpretation)\"\n"
        "S2,T-301,High level,\"CollectingBasin, OverfillProtection "
        "(reported/ontology safeguard concepts; actual installation status requires source interpretation)\"\n",
        encoding="utf-8",
    )

    records = HazopCsvLoader().load(path, "t301-parens")
    assert len(records) == 2

    for record in records:
        assert len(record.safeguards) == 2
        assert record.safeguards[0] == "CollectingBasin"
        assert record.safeguards[1].startswith("OverfillProtection (reported/ontology safeguard concepts")
        assert record.safeguards[1].endswith("actual installation status requires source interpretation)")


def test_normalization_of_empty_and_missing_cells(tmp_path: Path) -> None:
    path = tmp_path / "empty_cells.csv"
    path.write_text(
        "Scenario_ID,Plant/Unit,Equipment,Deviation,Possible_Causes,Possible_Consequences,Existing_Safeguards,Recommendations,Risk_Ranking\n"
        "S1, - , T-101 , High level , — , N/A , none , not stated , ---\n"
        "S2,,,Low level,,,,,\n",
        encoding="utf-8",
    )

    records = HazopCsvLoader().load(path, "test-empty")
    assert len(records) == 2

    first = records[0]
    assert first.plant_unit == ""
    assert first.equipment == "T-101"
    assert first.deviation == "High level"
    assert first.causes == ()
    assert first.consequences == ()
    assert first.safeguards == ()
    assert first.recommendations == ()
    assert first.risk_ranking is None

    second = records[1]
    assert second.equipment == "Unspecified equipment"
    assert second.deviation == "Low level"
    assert second.causes == ()
    assert second.consequences == ()


def test_effects_column_aliases_and_relationship_linking(tmp_path: Path) -> None:
    path = tmp_path / "effects_alias.csv"
    path.write_text(
        "Scenario_ID,Equipment,Deviation,Causes,Intermediate_Effect,Consequences\n"
        "S1,R-101,High temperature,Cooling loss,Exothermic runaway,Reactor rupture\n",
        encoding="utf-8",
    )

    record = HazopCsvLoader().load(path, "effects-test")[0]
    assert record.effects == ("Exothermic runaway",)
    assert record.causes == ("Cooling loss",)
    assert record.consequences == ("Reactor rupture",)

    predicates = [rel.predicate for rel in record.causal_relations]
    assert "causes_deviation" in predicates
    assert "leads_to" in predicates
    assert "leads_to_consequence" in predicates

    leads_to = next(rel for rel in record.causal_relations if rel.predicate == "leads_to")
    assert leads_to.subject == "High temperature"
    assert leads_to.object == "Exothermic runaway"

    leads_to_cons = next(rel for rel in record.causal_relations if rel.predicate == "leads_to_consequence")
    assert leads_to_cons.subject == "Exothermic runaway"
    assert leads_to_cons.object == "Reactor rupture"


def test_original_row_value_and_row_number_preservation(tmp_path: Path) -> None:
    path = tmp_path / "row_preservation.csv"
    path.write_text(
        "Scenario_ID,Equipment,Deviation,Possible_Causes,Custom_Note\n"
        "S100,P-101,No flow,Pump trip,Field operator note\n"
        "S101,V-102,Stuck open,Actuator failure,Second row note\n",
        encoding="utf-8",
    )

    records = HazopCsvLoader().load(path, "preservation-test")
    assert len(records) == 2

    assert records[0].row_number == 2
    assert records[0].raw_row["Scenario_ID"] == "S100"
    assert records[0].raw_row["Custom_Note"] == "Field operator note"

    assert records[1].row_number == 3
    assert records[1].raw_row["Scenario_ID"] == "S101"
    assert records[1].raw_row["Custom_Note"] == "Second row note"


def test_mixed_reconstructed_and_synthetic_records(tmp_path: Path) -> None:
    path = tmp_path / "mixed.csv"
    path.write_text(
        "Scenario_ID,Equipment,Deviation,Record_Type,Causes,Consequences\n"
        "S1,T-301,High level,PAPER_REPORTED_RECONSTRUCTION,Valve failure,Tank overflow\n"
        "S2,T-301,High level,SYNTHETIC_CONTINUATION,Sensor fault,Off-spec product\n",
        encoding="utf-8",
    )

    records = HazopCsvLoader().load(path, "mixed-test")
    assert len(records) == 2

    first, second = records[0], records[1]
    assert first.record_type == RecordType.SOURCE_RECONSTRUCTED
    assert all(rel.origin == RelationshipOrigin.RECONSTRUCTED for rel in first.causal_relations)
    assert first.validation_status == ValidationStatus.UNREVIEWED

    assert second.record_type == RecordType.SYNTHETIC
    assert all(rel.origin == RelationshipOrigin.SYNTHETIC for rel in second.causal_relations)
    assert second.validation_status == ValidationStatus.UNREVIEWED


def test_relationship_provenance_and_unverified_status(tmp_path: Path) -> None:
    path = tmp_path / "unverified.csv"
    path.write_text(
        "Scenario_ID,Equipment,Deviation,Causes,Consequences,Validation_Status\n"
        "S1,T-301,High level,Inlet valve fails open,Tank overflow,Review required\n",
        encoding="utf-8",
    )

    record = HazopCsvLoader().load(path, "unverified-test")[0]
    assert record.validation_status == ValidationStatus.UNREVIEWED
    assert "Review required" in record.remarks

    for rel in record.causal_relations:
        assert rel.origin == RelationshipOrigin.SYNTHETIC
        assert not rel.is_expert_verified
        assert "not an independently expert-validated causal assertion" in rel.evidence


def test_delimiters_support_pipes_and_newlines(tmp_path: Path) -> None:
    path = tmp_path / "delimiters.csv"
    path.write_text(
        "Scenario_ID,Equipment,Deviation,Causes,Existing_Safeguards\n"
        "S1,T-1,High level,Valve failure | Sensor drift,\"PSV\nHigh level alarm\"\n",
        encoding="utf-8",
    )

    record = HazopCsvLoader().load(path, "delim-test")[0]
    assert record.causes == ("Valve failure", "Sensor drift")
    assert record.safeguards == ("PSV", "High level alarm")

