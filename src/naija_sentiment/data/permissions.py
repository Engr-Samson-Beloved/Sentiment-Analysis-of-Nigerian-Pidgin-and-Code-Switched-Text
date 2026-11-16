"""Validation for the restricted source-permission ledger."""

from pathlib import Path

import pandas as pd

PERMISSION_COLUMNS = [
    "source_reference",
    "permission_status",
    "license_or_consent",
    "evidence_location",
    "restrictions",
]
ALLOWED_STATUSES = {"project_generated", "consent_obtained", "licensed", "public_domain", "pending", "rejected"}
USABLE_STATUSES = {"project_generated", "consent_obtained", "licensed", "public_domain"}


def read_permissions(path: str | Path) -> pd.DataFrame:
    """Load a permission ledger without converting blank cells to NaN."""
    ledger = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_permission_ledger(ledger)
    return ledger


def validate_permission_ledger(ledger: pd.DataFrame) -> None:
    """Validate ledger structure and evidence required for usable sources."""
    missing = set(PERMISSION_COLUMNS) - set(ledger.columns)
    if missing:
        raise ValueError(f"Missing permission columns: {sorted(missing)}")
    if ledger.empty:
        raise ValueError("Permission ledger is empty")
    if ledger.source_reference.str.strip().eq("").any():
        raise ValueError("Missing source_reference in permission ledger")
    if ledger.source_reference.duplicated().any():
        raise ValueError("Duplicate source_reference in permission ledger")
    invalid = sorted(set(ledger.permission_status) - ALLOWED_STATUSES)
    if invalid:
        raise ValueError(f"Invalid permission_status values: {invalid}")
    usable = ledger.permission_status.isin(USABLE_STATUSES)
    for column in ["license_or_consent", "evidence_location"]:
        if ledger.loc[usable, column].str.strip().eq("").any():
            raise ValueError(f"Usable sources require {column}")


def audit_permissions(dataset: pd.DataFrame, ledger: pd.DataFrame) -> dict:
    """Report whether every dataset source has documented, usable permission."""
    validate_permission_ledger(ledger)
    used = set(dataset.source_reference)
    indexed = ledger.set_index("source_reference")
    documented = set(indexed.index)
    missing = sorted(used - documented)
    unusable = sorted(
        reference
        for reference in used & documented
        if indexed.loc[reference, "permission_status"] not in USABLE_STATUSES
    )
    unused = sorted(documented - used)
    return {
        "ready_for_research_use": not missing and not unusable,
        "dataset_source_count": len(used),
        "documented_source_count": len(documented),
        "missing_source_references": missing,
        "unusable_source_references": unusable,
        "unused_ledger_references": unused,
    }
