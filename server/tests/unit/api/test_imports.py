"""Tests for the FastAPI CSV preview endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_CSV = (
    "Scenario_ID,Plant/Unit,Node/Equipment,Deviation,"
    "Possible_Causes,Possible_Consequences,Existing_Safeguards\n"
    "S001,Feed section,P-101,No flow,Pump failure,"
    "Reduced feed supply,Low-flow alarm\n"
)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "nlp-hazop-api",
    }


def test_csv_preview_returns_normalized_record_and_relationships() -> None:
    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", VALID_CSV.encode(), "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "preview_ready"
    assert payload["storage"] == "not_persisted"
    assert payload["summary"]["rows_read"] == 1
    assert payload["summary"]["records_loaded"] == 1
    assert payload["summary"]["relationships_generated"] == 2
    assert payload["records"][0]["scenario_id"] == "S001"
    assert payload["records"][0]["causes"] == ["Pump failure"]
    assert payload["records"][0]["causal_relations"][0]["predicate"] == (
        "causes_deviation"
    )
    assert any("not independently verified" in item for item in payload["warnings"])


def test_csv_preview_supports_utf8_bom_and_header_aliases() -> None:
    content = (
        "\ufeffScenario ID,Deviation,Possible_Causes\n"
        "S002,Low flow,Valve blockage\n"
    ).encode("utf-8")

    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["recognized_columns"] == {
        "scenario_id": "Scenario ID",
        "deviation": "Deviation",
        "causes": "Possible_Causes",
    }


def test_csv_preview_rejects_non_csv_file() -> None:
    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("workbook.xlsx", b"not a csv", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_csv_preview_rejects_missing_required_columns() -> None:
    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", b"Scenario_ID,Cause\nS001,Pump failure\n", "text/csv")},
    )

    assert response.status_code == 422
    assert "Deviation" in response.json()["detail"]


def test_csv_preview_reports_rows_without_scenario_id() -> None:
    content = (
        b"Scenario_ID,Deviation,Possible_Causes\n"
        b"S001,No flow,Pump failure\n"
        b",Low pressure,Valve blockage\n"
    )

    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["records_loaded"] == 1
    assert payload["summary"]["rows_skipped"] == 1
    assert any("Skipped 1 row" in item for item in payload["warnings"])


def test_csv_preview_rejects_invalid_utf8() -> None:
    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", b"Scenario_ID,Deviation\nS001,\xff", "text/csv")},
    )

    assert response.status_code == 422
    assert "UTF-8" in response.json()["detail"]


def test_csv_preview_rejects_empty_csv() -> None:
    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", b"", "text/csv")},
    )

    assert response.status_code == 400


def test_csv_preview_reports_unrecognized_non_empty_columns() -> None:
    content = (
        b"Scenario_ID,Deviation,Possible_Causes,Custom_Notes,Empty_Extra\n"
        b"S001,High pressure,Overfill,Plant operator note,\n"
    )

    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["unrecognized_columns"] == ["Custom_Notes"]
    assert any("Unrecognized non-empty column(s)" in item for item in payload["warnings"])
    assert "Custom_Notes" in payload["summary"]["unrecognized_columns"]


def test_csv_preview_reports_missing_expected_columns() -> None:
    content = (
        b"Scenario_ID,Deviation,Possible_Causes\n"
        b"S001,Low flow,Pump failure\n"
    )

    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    expected_missing = {"consequences", "safeguards", "equipment"}
    assert expected_missing.issubset(set(payload["missing_expected_columns"]))
    assert any("Expected HAZOP column(s) not found" in item for item in payload["warnings"])


def test_csv_preview_exposes_breakdowns_and_not_persisted() -> None:
    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", VALID_CSV.encode(), "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["persisted"] is False
    assert payload["summary"]["persisted"] is False
    assert "relationships_by_origin" in payload["summary"]
    assert payload["summary"]["relationships_by_origin"] == {"unspecified": 2}
    assert "relationships_by_predicate" in payload["summary"]
    assert payload["summary"]["relationships_by_predicate"]["causes_deviation"] == 1
    assert payload["summary"]["relationships_by_predicate"]["leads_to_consequence"] == 1


def test_csv_preview_preserves_raw_row_and_row_number() -> None:
    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", VALID_CSV.encode(), "text/csv")},
    )

    assert response.status_code == 200
    first_record = response.json()["records"][0]
    assert first_record["row_number"] == 2
    assert isinstance(first_record["raw_row"], dict)
    assert first_record["raw_row"]["Scenario_ID"] == "S001"
    assert first_record["raw_row"]["Plant/Unit"] == "Feed section"
    assert first_record["raw_row"]["Deviation"] == "No flow"


def test_csv_preview_safeguards_with_parentheses_commas() -> None:
    content = (
        b"Scenario_ID,Equipment,Deviation,Possible_Causes,Possible_Consequences,Existing_Safeguards\n"
        b"S001,T-301,High level,Inflow,Overflow,\"CollectingBasin, OverfillProtection "
        b"(reported/ontology safeguard concepts, actual installation status requires source interpretation)\"\n"
    )

    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    record = response.json()["records"][0]
    assert len(record["safeguards"]) == 2
    assert record["safeguards"][0] == "CollectingBasin"
    assert "(reported/ontology safeguard concepts, actual installation status requires source interpretation)" in record["safeguards"][1]


def test_csv_preview_with_effects_column_aliases() -> None:
    content = (
        b"Scenario_ID,Equipment,Deviation,Causes,Intermediate Effect,Consequences\n"
        b"S001,R-101,High temperature,Cooling loss,Exothermic runaway,Reactor rupture\n"
    )

    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recognized_columns"]["effects"] == "Intermediate Effect"
    record = payload["records"][0]
    assert record["effects"] == ["Exothermic runaway"]
    assert payload["summary"]["relationships_by_predicate"]["leads_to"] == 1
    assert payload["summary"]["relationships_by_predicate"]["leads_to_consequence"] == 1


def test_csv_preview_handles_malformed_rows_and_summarizes() -> None:
    content = (
        b"Scenario_ID,Deviation,Possible_Causes\n"
        b"S001,No flow,Pump trip,Extra Cell\n"
        b"S002,High temp\n"
    )

    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["malformed_rows"] == 2
    assert payload["summary"]["records_loaded"] == 2
    assert any("2 row(s) have a different number of cells" in item for item in payload["warnings"])
    assert any("Raw rows preserve missing-cell (null) and extra-cell (_extra_cells) data" in item for item in payload["warnings"])

    first_record = payload["records"][0]
    assert first_record["raw_row"]["_extra_cells"] == ["Extra Cell"]

    second_record = payload["records"][1]
    assert second_record["raw_row"]["Possible_Causes"] is None
    assert second_record["raw_row"]["_missing_columns"] == ["Possible_Causes"]


def test_csv_preview_multiple_effects_and_multiple_consequences() -> None:
    content = (
        b"Scenario_ID,Deviation,Effects,Consequences\n"
        b"S001,High temp,Overheating; Pressure rise,Thermal damage; Flange leak\n"
    )

    response = client.post(
        "/api/v1/datasets/preview",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    preds = payload["summary"]["relationships_by_predicate"]
    assert preds["leads_to"] == 2
    assert preds["co_occurs_with_consequence"] == 2
    assert "leads_to_consequence" not in preds

