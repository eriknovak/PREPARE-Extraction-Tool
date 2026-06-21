"""Instance-wide guard for blocking model switches during active extraction."""

from sqlmodel import Session, select

from app.models_db import ExtractionJob

# Non-terminal extraction job states.
# ExtractionJob.status comment: pending|running|completed|failed
ACTIVE_EXTRACTION_STATES = ("pending", "running")


def any_extraction_job_active(db: Session) -> bool:
    """True if any extraction job is currently active anywhere in the instance."""
    row = db.exec(
        select(ExtractionJob.id).where(
            ExtractionJob.status.in_(ACTIVE_EXTRACTION_STATES)
        )
    ).first()
    return row is not None
