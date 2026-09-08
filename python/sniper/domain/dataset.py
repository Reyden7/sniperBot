"""Dataset roles and access states for research isolation."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from sniper.config import Model


class DatasetRole(StrEnum):
    RESEARCH = "RESEARCH"
    VALIDATION = "VALIDATION"
    HOLDOUT = "HOLDOUT"
    FORWARD = "FORWARD"


class DatasetState(StrEnum):
    AVAILABLE = "AVAILABLE"
    SEALED = "SEALED"
    RESERVED = "RESERVED"


class DatasetDeclaration(Model):
    dataset_id: str
    symbol: Literal["EURUSD"] = "EURUSD"
    role: DatasetRole
    state: DatasetState
    start_utc: datetime | None = None
    end_utc_exclusive: datetime | None = None
    parent_dataset_id: str | None = None
    permitted_uses: tuple[str, ...]
    forbidden_uses: tuple[str, ...]


class DatasetRegistry(Model):
    schema_version: Literal[1] = 1
    holdout_unlock_requires_explicit_v2_freeze: Literal[True] = True
    datasets: tuple[DatasetDeclaration, ...]

    def require_research_access(self, dataset_id: str) -> DatasetDeclaration:
        declaration = next((item for item in self.datasets if item.dataset_id == dataset_id), None)
        if declaration is None:
            raise ValueError("dataset is not declared in the research registry")
        if declaration.role in (DatasetRole.HOLDOUT, DatasetRole.FORWARD):
            raise PermissionError(f"{declaration.role.value} dataset is sealed for model research")
        if declaration.state != DatasetState.AVAILABLE:
            raise PermissionError("dataset is not available for model research")
        return declaration
