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
        "Scenario_ID,Deviation,Possible_Causes\n"
        "S001,No flow,Pump failure\n"
        ",Low pressure,Valve blockage\n"
    ).encode()

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
