from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.schemas import MessageOutput
from app.core.database import get_session
from app.core.elastic import es_client

# ================================================
# Route definitions
# ================================================

router = APIRouter()


@router.get(
    "/check",
    response_model=MessageOutput,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Verifies that the API is running and responding to requests, including database and Elasticsearch connectivity",
    response_description="Confirmation message that the health check was successful",
)
def health_check(db: Session = Depends(get_session)):
    """
    Perform a comprehensive health check of the application.

    Validates:
    - Database connectivity (by executing a simple query)
    - Elasticsearch connectivity (by pinging the service)

    Returns:
        MessageOutput: Success message if all checks pass

    Raises:
        HTTPException: 503 Service Unavailable if any service is down
    """
    errors = []

    # Check database connectivity
    try:
        # Execute a simple query to verify database connection
        db.exec(select(1)).one()
    except Exception as e:
        errors.append(f"Database: {str(e)}")

    # Check Elasticsearch connectivity
    try:
        if not es_client.ping():
            errors.append("Elasticsearch: Unable to ping service")
    except Exception as e:
        errors.append(f"Elasticsearch: {str(e)}")

    # If any checks failed, return 503 Service Unavailable
    if errors:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Health check failed", "errors": errors},
        )

    return MessageOutput(message="Health check successful")
