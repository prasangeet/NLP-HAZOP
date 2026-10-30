"""Dataset metadata stored alongside imported HAZOP scenarios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Durable identity and import statistics for a HAZOP dataset."""

    dataset_id: str
    name: str
    filename: str
    source_sha256: str
    record_count: int
    relationship_count: int
    created_at: str

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "name",
            "filename",
            "source_sha256",
            "created_at",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

        for field_name in ("record_count", "relationship_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        """Serialize the dataset metadata for API responses."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "filename": self.filename,
            "source_sha256": self.source_sha256,
            "record_count": self.record_count,
            "relationship_count": self.relationship_count,
            "created_at": self.created_at,
        }
