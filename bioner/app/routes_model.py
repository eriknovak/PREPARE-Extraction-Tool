"""Endpoint to switch the active NER model at runtime (activate / swap).

The model is validated and the desired state is persisted here (API process);
the in-memory engine is hot-swapped by the inference worker on its next request
(see ``NERAPI.predict`` / ``model_manager``).
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Model"])

# Wired from main() once the server is built: lets the route validate against,
# and keep in sync, the same metadata that ``/model/info`` reports.
_context: dict = {"server": None, "default_model": None}


def register_model_context(server, default_model: str) -> None:
    """Wire the LitServer and the default model path into the activate route."""
    _context["server"] = server
    _context["default_model"] = default_model


class ActivateModelRequest(BaseModel):
    """Body for ``POST /model/activate``. ``model=null`` reverts to the default."""

    model: str | None = None


@router.post("/model/activate")
def activate_model(request: ActivateModelRequest):
    """Switch the active NER model.

    Pass a model path / HuggingFace id to activate it, or ``null`` to revert to
    the default model the service started with. The target is validated here; on
    an invalid model the current model is left untouched and a 400 is returned.
    The in-memory engine is swapped by the inference worker on its next request.
    """
    default_model = _context["default_model"]
    target = request.model or default_model

    # The default model is already loaded and known-good — skip (network) checks.
    if target != default_model and not model_manager.validate_model(target):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "INVALID_MODEL",
                "message": f"Model could not be found or loaded: {target}",
            },
        )

    metadata = model_manager.read_model_metadata(target)
    model_manager.write_desired(target, metadata)

    # Keep ``/model/info`` consistent within this API process.
    server = _context["server"]
    if server is not None:
        server.model_metadata = metadata

    is_default = target == default_model
    logger.info("Requested activation of NER model '%s' (default=%s)", target, is_default)
    return {"model": metadata, "active_model": target, "is_default": is_default}
