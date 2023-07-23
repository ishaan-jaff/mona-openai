from .logger import Logger
import atexit
import json
from typing import Optional
from collections.abc import Mapping


class FileLogger(Logger):
    """
    A simple logging class that saves monitored data in a file.
    """

    def __init__(self, file_name):
        self.file = open(file_name, "w")

        atexit.register(self.close_file)

    def close_file(self) -> None:
        if not self.file.closed:
            self.file.close()

    def log(self, message: Mapping, context_id: Optional[str]=None, export_timestamp: Optional[float]=None) -> None:
        self.file.writelines(
            [
                json.dumps(
                    {
                        "message": message,
                        "context_id": context_id,
                        "export_timestamp": export_timestamp,
                    }
                )
            ]
        )

    async def alog(
        self, message: Mapping, context_id: Optional[str]=None, export_timestamp: Optional[float]=None
    ) -> None:
        # TODO: Imlement actual asyncio usage.
        return self.log(message, context_id, export_timestamp)
