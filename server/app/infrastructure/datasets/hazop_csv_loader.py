"""Load provenance-labelled HAZOP CSV exports into domain records."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RecordType,
    RelationshipOrigin,
    SourceReference,
    ValidationStatus,
)


_EMPTY = {"", "-", "—", "–", "n/a", "na", "none", "not stated"}


def _pick(row: dict[str, str], *names: str, default: str = "") -> str:
    normalized = {key.strip().casefold(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name.casefold())
        if value is not None and value.strip():
            return value.strip()
    return default


def _split_items(value: str) -> tuple[str, ...]:
    if value.strip().casefold() in _EMPTY:
        return ()
    items: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"\s*[;|]\\s*", value):
        # T-301 reconstruction occasionally uses commas as secondary list
        # separators. DWSIM prose remains intact because its sentences do
        # not use comma-delimited enumerations in cause/consequence fields.
        parts = re.split(r"\s*,\\s*", raw) if ";" in value else [raw]
        for part in parts:
            cleaned = part.strip().strip('"').strip()
            key = cleaned.casefold()
            if cleaned and key not in _EMPTY and key not in seen:
                items.append(cleaned)
                seen.add(key)
    return tuple(items)


def _origin_and_type(
    record_type_text: str,
    dataset_id: str,
) -> tuple[RecordType, RelationshipOrigin]:
    text = record_type_text.casefold()
    if "paper_reported_reconstruction" in text:
        return (
            RecordType.SOURCE_RECONSTRUCTED,
            RelationshipOrigin.RECONSTRUCTED,
        )
    if "synthetic" in text or "dwsim" in dataset_id.casefold():
        return RecordType.SYNTHETIC, RelationshipOrigin.SYNTHETIC
    if "expert" in text:
        return RecordType.EXPERT_VALIDATED, RelationshipOrigin.EXPERT
    if "source_verified" in text:
        return RecordType.SOURCE_VERIFIED, RelationshipOrigin.SOURCE
    return RecordType.UNSPECIFIED, RelationshipOrigin.SYNTHETIC


def _source_reference(
    dataset_id: str,
    scenario_id: str,
    provenance: str,
    origin: RelationshipOrigin,
) -> SourceReference:
    if origin == RelationshipOrigin.RECONSTRUCTED:
        title = "Single et al. (2020), T-301 hexane case reconstruction"
        document_id = "single-2020-t301"
    elif origin == RelationshipOrigin.SYNTHETIC:
        title = f"Synthetic HAZOP dataset: {dataset_id}"
        document_id = f"synthetic-{dataset_id}"
    elif origin == RelationshipOrigin.EXPERT:
        title = f"Expert-validated HAZOP dataset: {dataset_id}"
        document_id = dataset_id
    else:
        title = f"HAZOP dataset: {dataset_id}"
        document_id = dataset_id
    return SourceReference(
        title=title,
        document_id=document_id,
        location=scenario_id,
        provenance_note=provenance,
    )


class HazopCsvLoader:
    """Map the repository's T-301 and DWSIM CSV formats into HAZOP records.

    Cause/deviation and deviation/consequence relations are derived from
    fields in the same worksheet row. Their evidence text explicitly states
    that they are row associations and not independently validated causal
    propositions.
    """

    def load(self, path: str | Path, dataset_id: str) -> tuple[HazopRecord, ...]:
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        records: list[HazopRecord] = []
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                record = self._record_from_row(row, dataset_id)
                if record is not None:
                    records.append(record)
        return tuple(records)

    def _record_from_row(
        self,
        row: dict[str, str],
        dataset_id: str,
    ) -> HazopRecord | None:
        scenario_id = _pick(row, "Scenario_ID", "Scenario ID")
        if not scenario_id:
            return None

        record_type_text = _pick(row, "Record_Type", "Evidence Source", "Source_Type")
        record_type, origin = _origin_and_type(record_type_text, dataset_id)
        deviation = _pick(row, "Deviation", default="Unspecified deviation")
        causes = _split_items(
            _pick(row, "Possible_Causes", "Possible_Cause", "Causes")
        )
        consequences = _split_items(
            _pick(row, "Possible_Consequences", "Possible_Consequence", "Consequences")
        )
        safeguards = _split_items(
            _pick(
                row,
                "Existing_Safeguards",
                "Existing Controls / Safeguards",
                "Candidate_Safeguard",
                "Candidate Safeguard",
            )
        )
        recommendations = _split_items(
            _pick(
                row,
                "Recommendations",
                "Recommendations / Candidate Safeguards",
            )
        )

        provenance = _pick(
            row,
            "Provenance",
            "Evidence_and_Limitation",
            "Remarks / Evidence",
            "Remarks",
            "Evidence Source",
            "Source_Type",
            default=f"Imported from {source_path_name(dataset_id)}",
        )
        validation_note = _pick(
            row,
            "Validation_Status",
            "Validation Status",
            default="not specified",
        )
        relations: list[CausalRelation] = []

        for cause_index, cause in enumerate(causes, start=1):
            relations.append(
                CausalRelation(
                    subject=cause,
                    predicate="causes_deviation",
                    object=deviation,
                    origin=origin,
                    evidence=(
                        f"Row-level association in scenario {scenario_id}: "
                        f"cause '{cause}' is listed with deviation "
                        f"'{deviation}'. Provenance: {provenance}. "
                        "This is a structured worksheet association, not "
                        "an independently expert-validated causal assertion."
                    ),
                    source_id=f"{scenario_id}:cause:{cause_index}",
                )
            )

        for consequence_index, consequence in enumerate(consequences, start=1):
            relations.append(
                CausalRelation(
                    subject=deviation,
                    predicate="leads_to_consequence",
                    object=consequence,
                    origin=origin,
                    evidence=(
                        f"Row-level association in scenario {scenario_id}: "
                        f"deviation '{deviation}' is listed with consequence "
                        f"'{consequence}'. Provenance: {provenance}. "
                        "This is a structured worksheet association, not "
                        "an independently expert-validated causal assertion."
                    ),
                    source_id=f"{scenario_id}:consequence:{consequence_index}",
                )
            )

        # Source-specific records are never promoted to expert-validated.
        validation_status = ValidationStatus.UNREVIEWED
        remarks = _pick(row, "Remarks", "Remarks / Evidence", "Evidence_and_Limitation")
        if validation_note and validation_note.casefold() not in {
            "validated", "expert validated"
        }:
            remarks = f"{remarks} | Validation note: {validation_note}".strip(" |")

        source = _source_reference(
            dataset_id,
            scenario_id,
            provenance,
            origin,
        )

        return HazopRecord(
            scenario_id=scenario_id,
            plant_unit=_pick(row, "Plant/Unit", "Plant / Process"),
            equipment=_pick(
                row,
                "Node/Equipment",
                "Node / Equipment",
                "Equipment",
                "Node",
                default="Unspecified equipment",
            ),
            design_intent=_pick(row, "Design_Intent", "Design Intent"),
            guideword=_pick(row, "Guideword"),
            parameter=_pick(row, "Parameter"),
            deviation=deviation,
            causes=causes,
            consequences=consequences,
            safeguards=safeguards,
            recommendations=recommendations,
            record_type=record_type,
            validation_status=validation_status,
            source=source,
            causal_relations=tuple(relations),
            risk_ranking=_pick(row, "Risk_Ranking", "Risk Ranking") or None,
            remarks=remarks,
        )


def source_path_name(dataset_id: str) -> str:
    """Return a stable display name when only a dataset identifier is known."""
    return dataset_id
