"""Frozen synthetic ingest artifact used only by the D06 provenance audit."""


def canonicalize_fixture_record(record: dict[str, object]) -> dict[str, object]:
    """Return an already canonical synthetic record without hidden mutation."""

    return dict(record)
