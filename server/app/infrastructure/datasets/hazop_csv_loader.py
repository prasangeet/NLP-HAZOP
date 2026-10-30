"""Load provenance-labelled HAZOP CSV exports into domain records."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, TextIO

from app.domain.ontology.schema import EntityType
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
        "source_type",
        "source type",
        "evidence source",
        "evidence_source",
    ),
    "provenance": (
        "provenance",
        "evidence_and_limitation",
        "evidence and limitation",
        "remarks / evidence",
        "remarks/evidence",
        "remarks",
        "source",
        "notes",
        "comments",
    ),
    "relationship_validation_status": (
        "relationship_validation_status",
        "relationship validation status",
        "edge_validation_status",
        "edge validation status",
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


def _normalize_provenance_label(value: str) -> str:
    """Normalize provenance labels while retaining exact-token semantics."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _origin_and_type(
    record_type_text: str,
    dataset_id: str,
) -> tuple[RecordType, RelationshipOrigin]:
    label = _normalize_provenance_label(record_type_text)
    labels = {
        "paper reported reconstruction": (
            RecordType.SOURCE_RECONSTRUCTED,
            RelationshipOrigin.RECONSTRUCTED,
        ),
        "paper reported reconstructed": (
            RecordType.SOURCE_RECONSTRUCTED,
            RelationshipOrigin.RECONSTRUCTED,
        ),
        "source reconstructed": (
            RecordType.SOURCE_RECONSTRUCTED,
            RelationshipOrigin.RECONSTRUCTED,
        ),
        "reconstructed": (
            RecordType.SOURCE_RECONSTRUCTED,
            RelationshipOrigin.RECONSTRUCTED,
        ),
        "reconstruction": (
            RecordType.SOURCE_RECONSTRUCTED,
            RelationshipOrigin.RECONSTRUCTED,
        ),
        "synthetic": (RecordType.SYNTHETIC, RelationshipOrigin.SYNTHETIC),
        "synthetic continuation": (
            RecordType.SYNTHETIC,
            RelationshipOrigin.SYNTHETIC,
        ),
        "synthetic scenario": (
            RecordType.SYNTHETIC,
            RelationshipOrigin.SYNTHETIC,
        ),
        "synthetic record": (
            RecordType.SYNTHETIC,
            RelationshipOrigin.SYNTHETIC,
        ),
        "source verified": (
            RecordType.SOURCE_VERIFIED,
            RelationshipOrigin.SOURCE,
        ),
        "verified source": (
            RecordType.SOURCE_VERIFIED,
            RelationshipOrigin.SOURCE,
        ),
        "source documented": (
            RecordType.SOURCE_VERIFIED,
            RelationshipOrigin.SOURCE,
        ),
        "expert validated": (
            RecordType.EXPERT_VALIDATED,
            RelationshipOrigin.EXPERT,
        ),
        "expert reviewed": (
            RecordType.EXPERT_VALIDATED,
            RelationshipOrigin.EXPERT,
        ),
    }
    if label in labels:
        return labels[label]

    dataset_label = _normalize_provenance_label(dataset_id)
    dataset_tokens = set(dataset_label.split())
    negations = {
        "not",
        "non",
        "un",
        "unverified",
        "unvalidated",
        "unknown",
    }
    if dataset_tokens.intersection(negations):
        return RecordType.UNSPECIFIED, RelationshipOrigin.UNSPECIFIED
    if "synthetic" in dataset_tokens:
        return RecordType.SYNTHETIC, RelationshipOrigin.SYNTHETIC
    if dataset_tokens.intersection({"reconstruction", "reconstructed"}):
        return (
            RecordType.SOURCE_RECONSTRUCTED,
            RelationshipOrigin.RECONSTRUCTED,
        )
    return RecordType.UNSPECIFIED, RelationshipOrigin.UNSPECIFIED


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
    elif origin == RelationshipOrigin.SOURCE:
        title = f"Source-verified HAZOP dataset: {dataset_id}"
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
            normalized_row: dict[str, str] = {}
            raw_row: dict[str, Any] = {}
            extra_cells: list[str] = []
            missing_columns: list[str] = []

            for key, value in row.items():
                if key is None:
                    if isinstance(value, list):
                        extra_cells.extend(str(item) for item in value)
                    elif value is not None:
                        extra_cells.append(str(value))
                else:
                    clean_key = key.lstrip("\ufeff")
                    if value is None:
                        missing_columns.append(clean_key)
                        raw_row[clean_key] = None
                        normalized_row[clean_key] = ""
                    else:
                        raw_row[clean_key] = value
                        normalized_row[clean_key] = value if isinstance(value, str) else ""

            if extra_cells:
                raw_row["_extra_cells"] = extra_cells
            if missing_columns:
                raw_row["_missing_columns"] = missing_columns

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
        raw_row: dict[str, Any] | None = None,
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
        relationship_validation_note = _pick(
            row,
            *COLUMN_ALIASES["relationship_validation_status"],
            default="",
        )
        validation_note_folded = validation_note.casefold()

        # Validate whether validation evidence supports expert validation
        has_positive_expert_validation = (
            validation_note_folded in {"validated", "expert validated", "expert_validated"}
            and record_type not in (RecordType.SYNTHETIC, RecordType.SOURCE_RECONSTRUCTED)
        )

        if has_positive_expert_validation:
            validation_status = ValidationStatus.VALIDATED
        elif validation_note_folded in {"rejected", "failed"}:
            validation_status = ValidationStatus.REJECTED
        else:
            validation_status = ValidationStatus.UNREVIEWED

        if (
            origin == RelationshipOrigin.EXPERT
            and validation_status != ValidationStatus.VALIDATED
        ):
            origin = RelationshipOrigin.UNSPECIFIED
            if record_type == RecordType.EXPERT_VALIDATED:
                record_type = RecordType.UNSPECIFIED

        # Record-level validation is not a substitute for explicit
        # relationship-level expert validation.
        relationship_status = _normalize_provenance_label(
            relationship_validation_note
        )
        is_verified = (
            origin == RelationshipOrigin.EXPERT
            and relationship_status in {
                "validated",
                "expert validated",
                "explicitly expert validated",
            }
        )

        relations: list[CausalRelation] = []
        validation_phrase = (
            "This causal relationship has explicit relationship-level expert validation."
            if is_verified
            else "This is not an independently expert-validated causal assertion at the relationship level."
        )
        for cause_index, cause in enumerate(causes, start=1):
            relations.append(
                CausalRelation(
                    subject=cause,
                    predicate="causes_deviation",
                    object=deviation,
                    subject_type=EntityType.CAUSE,
                    object_type=EntityType.DEVIATION,
                    origin=origin,
                    evidence=(
                        f"Row-level association in scenario {scenario_id}: "
                        f"cause {cause!r} is listed with deviation "
                        f"{deviation!r}. Record origin: {origin.value}. "
                        f"Provenance: {provenance}. {validation_phrase}"
                    ),
                    source_id=f"{dataset_id}:{scenario_id}:row-{row_number or 0}:cause:{cause_index}",
                    is_expert_verified=is_verified,
                    dataset_id=dataset_id,
                    source_location=(f"row {row_number}" if row_number else scenario_id),
                )
            )

        if effects:
            for effect_index, effect in enumerate(effects, start=1):
                relations.append(
                    CausalRelation(
                        subject=deviation,
                        predicate="leads_to",
                        object=effect,
                        subject_type=EntityType.DEVIATION,
                        object_type=EntityType.EFFECT,
                        origin=origin,
                        evidence=(
                            f"Scenario {scenario_id} lists simulated or "
                            f"inferred effect {effect!r} for deviation "
                            f"{deviation!r}. This effect retains the row's "
                            f"origin ({origin.value}). {validation_phrase} "
                            f"Provenance: {provenance}"
                        ),
                        source_id=f"{dataset_id}:{scenario_id}:row-{row_number or 0}:effect:{effect_index}",
                        is_expert_verified=is_verified,
                        dataset_id=dataset_id,
                        source_location=(f"row {row_number}" if row_number else scenario_id),
                    )
                )

            # Effect-to-consequence causal relationship:
            # ONLY generate a causal relationship when the source explicitly provides
            # enough information to establish that pairing (e.g. single intermediate effect).
            # When multiple effects exist, pairings to consequences are ambiguous in a flat row;
            # do NOT infer positional pairings (zip) or Cartesian products.
            if len(effects) == 1:
                single_effect = effects[0]
                for consequence_index, consequence in enumerate(
                    consequences, start=1
                ):
                    relations.append(
                        CausalRelation(
                            subject=single_effect,
                            predicate="leads_to_consequence",
                            object=consequence,
                            subject_type=EntityType.EFFECT,
                            object_type=EntityType.CONSEQUENCE,
                            origin=origin,
                            evidence=(
                                f"Row-level association in scenario {scenario_id}: "
                                f"single intermediate effect {single_effect!r} and "
                                f"consequence {consequence!r} are co-listed. "
                                f"Record origin: {origin.value}. "
                                f"Provenance: {provenance}. {validation_phrase}"
                            ),
                            source_id=f"{dataset_id}:{scenario_id}:row-{row_number or 0}:effect:1:consequence:{consequence_index}",
                            is_expert_verified=is_verified,
                            dataset_id=dataset_id,
                            source_location=(f"row {row_number}" if row_number else scenario_id),
                        )
                    )
            else:
                # Multiple effects: represent consequence co-occurrence without making
                # an unsupported causal assertion.
                for consequence_index, consequence in enumerate(
                    consequences, start=1
                ):
                    relations.append(
                        CausalRelation(
                            subject=deviation,
                            predicate="co_occurs_with_consequence",
                            object=consequence,
                            subject_type=EntityType.DEVIATION,
                            object_type=EntityType.CONSEQUENCE,
                            origin=origin,
                            evidence=(
                                f"Scenario {scenario_id} co-lists consequence {consequence!r} "
                                f"with multiple intermediate effects ({', '.join(effects)}). "
                                "Explicit causal pairing between intermediate effects and "
                                "consequences is not provided in the source row."
                            ),
                            source_id=f"{dataset_id}:{scenario_id}:row-{row_number or 0}:consequence:{consequence_index}",
                            is_expert_verified=False,
                            dataset_id=dataset_id,
                            source_location=(f"row {row_number}" if row_number else scenario_id),
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
                        subject_type=EntityType.DEVIATION,
                        object_type=EntityType.CONSEQUENCE,
                        origin=origin,
                        evidence=(
                            f"Row-level association in scenario {scenario_id}: "
                            f"deviation {deviation!r} is listed with "
                            f"consequence {consequence!r}. Record origin: "
                            f"{origin.value}. Provenance: {provenance}. "
                            f"{validation_phrase}"
                        ),
                        source_id=f"{dataset_id}:{scenario_id}:row-{row_number or 0}:consequence:{consequence_index}",
                        is_expert_verified=is_verified,
                        dataset_id=dataset_id,
                        source_location=(f"row {row_number}" if row_number else scenario_id),
                    )
                )

        remarks = _pick(
            row,
            "Remarks",
            "Remarks / Evidence",
            "Evidence_and_Limitation",
        )
        if validation_note and validation_note_folded not in {
            "validated",
            "expert validated",
            "expert_validated",
            "not specified",
        }:
            remarks = (
                f"{remarks} | Validation note: {validation_note}"
            ).strip(" |")

        if raw_row:
            malformed_parts = []
            if "_extra_cells" in raw_row:
                malformed_parts.append(
                    f"{len(raw_row['_extra_cells'])} extra cell(s)"
                )
            if "_missing_columns" in raw_row:
                malformed_parts.append(
                    f"missing {len(raw_row['_missing_columns'])} cell(s)"
                )
            if malformed_parts:
                remarks = (
                    f"{remarks} | Malformed row: {'; '.join(malformed_parts)}"
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
            dataset_id=dataset_id,
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
