from .logger import Logger
from logging import getLogger, INFO
from typing import Optional
from collections.abc import Mapping


class StandardLogger(Logger):
    """
    A simple logging class that logs monitored data using python's logging
    package.
    """

    def __init__(
        self,
        logging_level=INFO,
        underlying_logger=None,
    ):
        self.underlying_logger = underlying_logger or getLogger("Mona")
        self.level = logging_level

    def log(self, message: Mapping, context_id: Optional[str]=None, export_timestamp: Optional[float]=None) -> None:
        self.underlying_logger.log(
            self.level,
            {
                "message": message,
                "context_id": context_id,
                "export_timestamp": export_timestamp,
            },
        )

    async def alog(
        self, message: Mapping, context_id: Optional[str]=None, export_timestamp: Optional[float]=None
    ) -> None:
        return self.log(message, context_id, export_timestamp)

    def start_monitoring(self, openai_class_name: str) -> None:
        self.underlying_logger.log(
            self.level, f"Started monitoring for OpenAI's {openai_class_name}"
        )
