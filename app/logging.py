import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

logger = logging.getLogger("cited_docs_qa")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def log_event(event: str, **fields) -> None:
    payload = {
        "ts": time.time(),
        "request_id": _request_id.get(),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, default=str))
