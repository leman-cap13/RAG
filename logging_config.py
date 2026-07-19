import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED_RECORD_ATTRS = set(logging.LogRecord(
    name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
).__dict__.keys()) | {"message", "asctime"}


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS and key != "request_id"
        }
        if extras:
            payload.update(extras)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


_MAX_EXTRA_LEN = 300


def _truncate(value):
    if isinstance(value, str) and len(value) > _MAX_EXTRA_LEN:
        return f"{value[:_MAX_EXTRA_LEN]}... [+{len(value) - _MAX_EXTRA_LEN} chars]"
    return value


_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"


class TextFormatter(logging.Formatter):
    def __init__(self, use_color=None):
        super().__init__()
        self.use_color = sys.stdout.isatty() if use_color is None else use_color

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname
        request_id = getattr(record, "request_id", "-")

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS and key != "request_id"
        }
        extras_str = " ".join(f"{key}={_truncate(value)}" for key, value in extras.items())

        prefix = f"[{request_id[:8]}] " if request_id != "-" else ""
        line = f"{timestamp} {level:<8} {record.name:<16} {prefix}{record.getMessage()}"
        if extras_str:
            line += f"  {extras_str}"

        if self.use_color:
            line = f"{_LEVEL_COLORS.get(level, '')}{line}{_RESET}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


def setup_logging(level="INFO", fmt="text"):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if fmt == "json" else TextFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    for noisy_logger in ("httpx", "httpcore", "chromadb", "google_genai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
