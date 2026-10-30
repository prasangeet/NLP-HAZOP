"""Load provenance-labelled HAZOP CSV exports into domain records."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RecordType,
    RelationshipOrigin,
    SourceReference,
    ValidationStatus,
)

_EMPTY = {"", "-", "—", "–", "n/a", "na", "none", "not stated", "null"}
_NOT_PROVIDED_PREFIXES = (
    "not_stated",
    "not stated",
    "not established",
    "not mapped",
    "no existing",
    "no safeguard",
    "none provided",
    "none listed",
)


def _is_empty(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return True
    folded = cleaned.casefold()
    if folded in _EMPTY:
        return True
    return all(char in "-—–_ \t" for char in cleaned)

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "scenario_id": (
        "scenario_id",
        "scenario id",
        "scenario",
        "id",
    ),
    "plant_unit": (
        "plant/unit",
        "plant / unit",
        "plant / process",
        "plant/process",
        "plant",
        "unit",
        "area",
        "section",
        "node",
    ),
    "equipment": (
        "node/equipment",
        "node / equipment",
        "equipment",
        "equipment / node",
        "equipment/node",
        "item",
        "tag",
        "node",
    ),
    "design_intent": (
        "design_intent",
        "design intent",
        "intent",
    ),
    "guideword": (
        "guideword",
        "guide_word",
        "guide word",
    ),
    "parameter": (
        "parameter",
        "process_parameter",
        "process parameter",
        "variable",
    ),
    "deviation": (
        "deviation",
        "deviations",
    ),
    "causes": (
        "possible_causes",
        "possible_cause",
        "possible causes",
        "possible cause",
        "causes",
        "cause",
        "initiating_event",
        "initiating event",
    ),
    "effects": (
        "simulated_or_inferred_effect",
        "simulated or inferred effect",
        "simulated_effect",
        "simulated effect",
        "inferred_effect",
        "inferred effect",
        "effects",
        "effect",
        "intermediate_effect",
        "intermediate_effects",
        "intermediate effect",
        "intermediate effects",
        "intermediate_state",
        "intermediate_states",
        "intermediate state",
        "intermediate states",
    ),
    "consequences": (
        "possible_consequences",
        "possible_consequence",
        "possible consequences",
        "possible consequence",
        "consequences",
        "consequence",
        "hazard",
        "hazards",
    ),
    "safeguards": (
        "existing_safeguards",
        "existing_safeguard",
        "existing safeguards",
        "existing safeguard",
        "existing controls / safeguards",
        "existing controls/safeguards",
        "safeguards",
        "safeguard",
        "controls",
        "existing_controls",
        "existing controls",
    ),
    "recommendations": (
        "recommendations",
        "recommendation",
        "recommendations / candidate safeguards",
        "candidate_safeguard",
        "candidate_safeguards",
        "candidate safeguard",
        "candidate safeguards",
        "actions",
        "action_items",
    ),
    "risk_ranking": (
        "risk_ranking",
        "risk ranking",
        "risk_rank",
        "risk rank",
        "risk",
        "severity",
    ),
    "record_type": (
        "record_type",
        "record type",
        "evidence source",
        "evidence_source",
        "source_type",
        "source type",
    ),
    "provenance": (
        "provenance",
        "evidence_and_limitation",
        "evidence and limitation",
        "remarks / evidence",
        "remarks/evidence",
        "remarks",
        "evidence source",
        "source_type",
        "source",
        "notes",
        "comments",
    ),
    "validation_status": (
        "validation_status",
        "validation status",
        "validation",
        "status",
    ),
}


def _pick(row: dict[str, str | None], *names: str, default: str = "") -> str:
    normalized = {
        key.strip().lstrip("\ufeff").casefold(): value
        for key, value in row.items()
        if isinstance(key, str)
    }
    for name in names:
        value = normalized.get(name.strip().casefold())
        if isinstance(value, str) and value.strip():
            cleaned = value.strip()
            if not _is_empty(cleaned):
                return cleaned
    return default


def _split_items(value: str) -> tuple[str, ...]:
    cleaned_value = value.strip()
    folded = cleaned_value.casefold()
    if (
        not cleaned_value
        or _is_empty(cleaned_value)
        or folded.startswith(_NOT_PROVIDED_PREFIXES)
    ):
        return ()

    raw_parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in cleaned_value:
        if char in "([":
            depth += 1
            current.append(char)
        elif char in ")]":
            depth = max(0, depth - 1)
            current.append(char)
        elif depth == 0 and char in ";|\n,":
            part = "".join(current).strip().strip("\"'").strip()
            if part:
                raw_parts.append(part)
            current = []
        else:
            current.append(char)

    if current:
        part = "".join(current).strip().strip("\"'").strip()
        if part:
            raw_parts.append(part)

    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_parts:
        cleaned = raw.strip().strip("\"'").strip()
        key = cleaned.casefold()
        if (
            cleaned
            and key not in _EMPTY
            and not key.startswith(_NOT_PROVIDED_PREFIXES)
            and key not in seen
        ):
            items.append(cleaned)
            seen.add(key)
    return tuple(items)


def _origin_and_type(
    record_type_text: str,
    dataset_id: str,
) -> tuple[RecordType, RelationshipOrigin]:
    text = record_type_text.casefold()
    if "paper_reported_reconstruction" in text or "reconstruct" in text:
        return (
            RecordType.SOURCE_RECONSTRUCTED,
            RelationshipOrigin.RECONSTRUCTED,
        )
    if "synthetic" in text:
        return RecordType.SYNTHETIC, RelationshipOrigin.SYNTHETIC
    if "source_verified" in text or "verified" in text:
        return RecordType.SOURCE_VERIFIED, RelationshipOrigin.SOURCE
    if "expert" in text and "valid" in text:
        return RecordType.EXPERT_VALIDATED, RelationshipOrigin.EXPERT

    ds_folded = dataset_id.casefold()
    if "dwsim" in ds_folded or "synthetic" in ds_folded:
        return RecordType.SYNTHETIC, RelationshipOrigin.SYNTHETIC
    if "reconstruct" in ds_folded:
        return RecordType.SOURCE_RECONSTRUCTED, RelationshipOrigin.RECONSTRUCTED
    return RecordType.UNSPECIFIED, RelationshipOrigin.SYNTHETIC


def _source_reference(
    dataset_id: str,
    scenario_id: str,
    provenance: str,
    origin: RelationshipOrigin,
    row_number: int | None = None,
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
        location=f"row {row_number}: {scenario_id}" if row_number else scenario_id,
        provenance_note=provenance,
    )


class HazopCsvLoader:
    """Map T-301 and DWSIM CSV formats into provenance-aware records.

    Cause-to-deviation, deviation-to-effect, and effect-to-consequence edges
    are derived from fields co-listed in one worksheet row. Their evidence
    explicitly identifies them as row associations, not independently
    expert-validated causal propositions.
    """

    def load(self, path: str | Path, dataset_id: str) -> tuple[HazopRecord, ...]:
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return self.load_stream(handle, dataset_id)

    def load_stream(
        self,
        handle: TextIO,
        dataset_id: str,
    ) -> tuple[HazopRecord, ...]:
        """Load records from an already-open text stream.

        This is used by the HTTP upload preview so uploaded CSV content can be
        parsed without writing user-provided files to a temporary directory.
        """
        if not dataset_id.strip():
            raise ValueError("dataset_id must not be empty")

        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file must contain a header row")

        if reader.fieldnames and reader.fieldnames[0]:
            reader.fieldnames[0] = reader.fieldnames[0].lstrip("\ufeff")

        records: list[HazopRecord] = []
        for row_index, row in enumerate(reader, start=2):
            normalized_row = {
                key.lstrip("\ufeff"): value if isinstance(value, str) else ""
                for key, value in row.items()
                if isinstance(key, str)
            }
            raw_row = {
                key: value if isinstance(value, str) else ""
                for key, value in row.items()
                if isinstance(key, str)
            }
            record = self._record_from_row(
                normalized_row,
                dataset_id,
                row_number=row_index,
                raw_row=raw_row,
            )
            if record is not None:
                records.append(record)
        return tuple(records)

    def _record_from_row(
        self,
        row: dict[str, str],
        dataset_id: str,
        row_number: int | None = None,
        raw_row: dict[str, str] | None = None,
    ) -> HazopRecord | None:
        scenario_id = _pick(row, *COLUMN_ALIASES["scenario_id"])
        if not scenario_id:
            return None

        record_type_text = _pick(row, *COLUMN_ALIASES["record_type"])
        record_type, origin = _origin_and_type(
            record_type_text, dataset_id
        )
        deviation = _pick(
            row, *COLUMN_ALIASES["deviation"], default="Unspecified deviation"
        )
        causes = _split_items(_pick(row, *COLUMN_ALIASES["causes"]))
        effects = _split_items(_pick(row, *COLUMN_ALIASES["effects"]))
        consequences = _split_items(
            _pick(row, *COLUMN_ALIASES["consequences"])
        )
        safeguards = _split_items(_pick(row, *COLUMN_ALIASES["safeguards"]))
        recommendations = _split_items(
            _pick(row, *COLUMN_ALIASES["recommendations"])
        )

        provenance = _pick(
            row,
            *COLUMN_ALIASES["provenance"],
            default=f"Imported from {dataset_id}",
        )
        validation_note = _pick(
            row,
            *COLUMN_ALIASES["validation_status"],
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
                        f"cause {cause!r} is listed with deviation "
                        f"{deviation!r}. Record origin: {origin.value}. "
                        f"Provenance: {provenance}. This is not an "
                        "independently expert-validated causal assertion."
                    ),
                    source_id=f"{scenario_id}:cause:{cause_index}",
                )
            )

        if effects:
            for effect_index, effect in enumerate(effects, start=1):
                relations.append(
                    CausalRelation(
                        subject=deviation,
                        predicate="leads_to",
                        object=effect,
                        origin=origin,
                        evidence=(
                            f"Scenario {scenario_id} lists simulated or "
                            f"inferred effect {effect!r} for deviation "
                            f"{deviation!r}. This effect retains the row's "
                            f"origin ({origin.value}); it is not expert "
                            f"validation. Provenance: {provenance}"
                        ),
                        source_id=f"{scenario_id}:effect:{effect_index}",
                    )
                )

            if len(effects) > 1 and len(effects) == len(consequences):
                for pair_index, (effect, consequence) in enumerate(
                    zip(effects, consequences, strict=False), start=1
                ):
                    relations.append(
                        CausalRelation(
                            subject=effect,
                            predicate="leads_to_consequence",
                            object=consequence,
                            origin=origin,
                            evidence=(
                                f"Row-level association in scenario "
                                f"{scenario_id}: effect {effect!r} and "
                                f"consequence {consequence!r} are co-listed. "
                                f"Record origin: {origin.value}. "
                                f"Provenance: {provenance}. This is not an "
                                "independently expert-validated causal assertion."
                            ),
                            source_id=(
                                f"{scenario_id}:effect:{pair_index}:"
                                f"consequence:{pair_index}"
                            ),
                        )
                    )
            else:
                for effect_index, effect in enumerate(effects, start=1):
                    for consequence_index, consequence in enumerate(
                        consequences, start=1
                    ):
                        relations.append(
                            CausalRelation(
                                subject=effect,
                                predicate="leads_to_consequence",
                                object=consequence,
                                origin=origin,
                                evidence=(
                                    f"Row-level association in scenario "
                                    f"{scenario_id}: effect {effect!r} and "
                                    f"consequence {consequence!r} are co-listed. "
                                    f"Record origin: {origin.value}. "
                                    f"Provenance: {provenance}. This is not an "
                                    "independently expert-validated causal assertion."
                                ),
                                source_id=(
                                    f"{scenario_id}:effect:{effect_index}:"
                                    f"consequence:{consequence_index}"
                                ),
                            )
                        )
        else:
            for consequence_index, consequence in enumerate(
                consequences, start=1
            ):
                relations.append(
                    CausalRelation(
                        subject=deviation,
                        predicate="leads_to_consequence",
                        object=consequence,
                        origin=origin,
                        evidence=(
                            f"Row-level association in scenario {scenario_id}: "
                            f"deviation {deviation!r} is listed with "
                            f"consequence {consequence!r}. Record origin: "
                            f"{origin.value}. Provenance: {provenance}. "
                            "This is not an independently expert-validated "
                            "causal assertion."
                        ),
                        source_id=(
                            f"{scenario_id}:consequence:{consequence_index}"
                        ),
                    )
                )

        # Do not convert synthetic or reconstructed records into expert-validated records.
        validation_status = ValidationStatus.UNREVIEWED
        if (
            record_type == RecordType.EXPERT_VALIDATED
            and validation_note.casefold() in {
                "validated",
                "expert validated",
                "expert_validated",
            }
        ):
            validation_status = ValidationStatus.VALIDATED

        remarks = _pick(
            row,
            "Remarks",
            "Remarks / Evidence",
            "Evidence_and_Limitation",
        )
        if validation_note and validation_note.casefold() not in {
            "validated",
            "expert validated",
            "expert_validated",
            "not specified",
        }:
            remarks = (
                f"{remarks} | Validation note: {validation_note}"
            ).strip(" |")

        source = _source_reference(
            dataset_id,
            scenario_id,
            provenance,
            origin,
            row_number=row_number,
        )
        return HazopRecord(
            scenario_id=scenario_id,
            plant_unit=_pick(row, *COLUMN_ALIASES["plant_unit"]),
            equipment=_pick(
                row,
                *COLUMN_ALIASES["equipment"],
                default="Unspecified equipment",
            ),
            design_intent=_pick(row, *COLUMN_ALIASES["design_intent"]),
            guideword=_pick(row, *COLUMN_ALIASES["guideword"]),
            parameter=_pick(row, *COLUMN_ALIASES["parameter"]),
            deviation=deviation,
            causes=causes,
            effects=effects,
            consequences=consequences,
            safeguards=safeguards,
            recommendations=recommendations,
            record_type=record_type,
            validation_status=validation_status,
            source=source,
            causal_relations=tuple(relations),
            risk_ranking=_pick(row, *COLUMN_ALIASES["risk_ranking"]) or None,
            remarks=remarks,
            row_number=row_number,
            raw_row=raw_row if raw_row is not None else dict(row),
        )
