from ..logger import Logger
from mona_sdk import MonaSingleMessage
from .mona_client import get_mona_clients, MonaCredsType
from collections.abc import Callable
import logging
from typing import Optional
from collections.abc import Mapping


class MonaLogger(Logger):
    def __init__(
        self,
        mona_creds: MonaCredsType,
        context_class: str,
        mona_clients_getter: Callable=get_mona_clients,
    ):
        self.client, self.async_client = mona_clients_getter(mona_creds)
        self.context_class = context_class

    def start_monitoring(self, openai_class_name) -> dict:
        """
        Calls Mona's server to init the given context class specifically for
        the given OpenAI class name.
        """
        response = self.client.create_openai_context_class(
            self.context_class, openai_class_name
        )
        error_message = response.get("error_message")
        if error_message:
            logging.warning(
                f"Problem initializing Mona context class"
                f" '{self.context_class}': {error_message}"
            )
        else:
            logging.info(
                f"Made sure Mona context class '{self.context_class}' "
                "is initialised"
            )
        return response

    def log(self, message: Mapping, context_id: Optional[str]=None, export_timestamp: Optional[float]=None) -> None:
        """
        Logs the given message to Mona.
        """
        return self.client.export(
            MonaSingleMessage(
                message=message,
                contextClass=self.context_class,
                contextId=context_id,
                exportTimestamp=export_timestamp,
            )
        )

    async def alog(
        self, message: Mapping, context_id: Optional[str]=None, export_timestamp: Optional[float]=None
    ) -> None:
        """
        Async logs the given message to Mona.
        """
        return await self.async_client.export_async(
            MonaSingleMessage(
                message=message,
                contextClass=self.context_class,
                contextId=context_id,
                exportTimestamp=export_timestamp,
            )
        )
