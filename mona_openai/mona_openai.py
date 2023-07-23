import time
from .loggers.mona_logger.mona_logger import MonaLogger
from .loggers.logger import Logger
from copy import deepcopy
from typing import Optional
from collections.abc import Callable, Mapping
from .util.general_consts import EMPTY_DICT
from .util.typing_util import SupportedOpenAIClassesType

from .exceptions import InvalidLagnchainLLMException
from .endpoints.wrapping_getter import get_endpoint_wrapping
from .loggers.mona_logger.mona_client import get_mona_clients
from .util.func_util import add_conditional_sampling
from .util.async_util import (
    run_in_an_event_loop,
    call_non_blocking_sync_or_async,
)
from .util.openai_util import get_model_param
from .util.tokens_util import get_usage
from .util.stream_util import ResponseGatheringIterator
from .util.validation_util import validate_and_get_sampling_ratio

from .loggers.mona_logger.mona_client import MonaCredsType

MONA_ARGS_PREFIX = "MONA_"
CONTEXT_ID_ARG_NAME = MONA_ARGS_PREFIX + "context_id"
EXPORT_TIMESTAMP_ARG_NAME = MONA_ARGS_PREFIX + "export_timestamp"
ADDITIONAL_DATA_ARG_NAME = MONA_ARGS_PREFIX + "additional_data"


def _get_logging_message(
    api_name: str,
    request_input: Mapping,
    start_time: float,
    is_exception: bool,
    is_async: bool,
    stream_start_time: Optional[float],
    response: Optional[Mapping],
    analysis_getter: Callable[[Mapping, Mapping], dict],
    message_cleaner: Callable[[Mapping], dict],
    additional_data: Mapping,
) -> dict:
    """
    Returns a dict object containing all the monitoring analysis to be used
    for data logging.
    """

    message = {
        "input": request_input,
        "latency": time.time() - start_time,
        "stream_start_latency": stream_start_time - start_time
        if stream_start_time is not None
        else None,
        "is_exception": is_exception,
        "api_name": api_name,
        "is_async": is_async,
    }

    if additional_data:
        message["additional_data"] = additional_data

    if response:
        message["response"] = response
        message["analysis"] = analysis_getter(request_input, response)

    return message_cleaner(message)


# TODO(itai): Consider creating some sturct (as NamedTuple or dataclass) for
#   the specs param.


def monitor(
    openai_class: SupportedOpenAIClassesType,
    mona_creds: MonaCredsType,
    context_class: str,
    specs: Mapping = EMPTY_DICT,
    mona_clients_getter: Callable = get_mona_clients,
) -> SupportedOpenAIClassesType:
    """
    A simple wrapper around "monitor_with_logger" to use with a Mona logger.
    See "monitor_with_logger" for full documentation.

    Args:
        openai_class: An OpenAI API class to wrap with monitoring
            capabilties.
        mona_creds: Either a dict or pair of Mona API key and secret to
            set up Mona's clients from its SDK
        context_class: The Mona context class name to use for
            monitoring. Use a name of your choice.
        specs: A dictionary of specifications such as monitoring
            sampling ratio.
        mona_clients_getter: Used only for testing purposes.
    """
    return monitor_with_logger(
        openai_class,
        MonaLogger(mona_creds, context_class, mona_clients_getter),
        specs,
    )


def monitor_with_logger(openai_class, logger, specs=EMPTY_DICT):
    """
    Returns a Wrapped version of a given OpenAI class with monitoring logic.

    You can use the returned class' "create" and "acreate" functions
    exactly as you would the original class, and monitoring will be
    taken care of for you.

    This client will automatically monitor for you things like latency,
    prompt and response lengths, number of tokens, etc., along with any
    endpoint parameter usage (e.g., it tracks the "temperature" and
    "max_tokens" params you use).

    The logic for what to do with the calculated analysis data is set by the
    given logger object.

    You can also add other named args when calling "create" or
    "acreate" by using a new named argument called
    "MONA_additional_data" and set it to any JSON serializable
    dictionary.
    This allows you to add metadata about the call such as a prompt
    template ID, information about the context in which the API call is
    made, etc...

    Furthermore, you can add to create/acreate functions mona specific
    arguments:
        MONA_context_id: The unique id of the context in which the call
            is made. By using this ID you can export more data to Mona
            to the same context from other places. If not used, the
            "id" field of the OpenAI Endpoint's response will be used.
        MONA_export_timestamp: Can be used to simulate as if the
            current call was made in a different time, as far as Mona
            is concerned.

    Args:
        openai_class: An OpenAI API class to wrap with monitoring
            capabilties.
        logger: A logger object used to log out the calculated analysis.
        specs: A dictionary of specifications such as monitoring
            sampling ratio.
    """

    sampling_ratio = validate_and_get_sampling_ratio(specs)

    base_class = get_endpoint_wrapping(
        openai_class.__name__, specs
    ).wrap_class(openai_class)

    logger.start_monitoring(openai_class.__name__)

    class MonitoredOpenAI(base_class):
        """
        A monitored version of an openai API class.
        """

        @classmethod
        def _get_logging_message(
            cls,
            kwargs_param: Mapping,
            start_time: float,
            is_exception: bool,
            is_async: bool,
            stream_start_time: float,
            response: Mapping,
        ) -> dict:
            """
            Returns a dict to be used for data logging.
            """
            # Recreate the input dict to avoid manipulating the caller's data,
            # and remove Mona-related data.
            request_input = deepcopy(
                {
                    x: kwargs_param[x]
                    for x in kwargs_param
                    if not x.startswith(MONA_ARGS_PREFIX)
                }
            )

            additional_data: Mapping = kwargs_param.get(
                ADDITIONAL_DATA_ARG_NAME, EMPTY_DICT
            )

            return _get_logging_message(
                api_name=openai_class.__name__,
                request_input=request_input,
                start_time=start_time,
                is_exception=is_exception,
                is_async=is_async,
                stream_start_time=stream_start_time,
                response=response,
                analysis_getter=super()._get_full_analysis,
                message_cleaner=super()._get_clean_message,
                additional_data=additional_data,
            )

        @classmethod
        async def _inner_create(
            cls,
            export_function: Callable,
            super_function: Callable,
            args,
            kwargs,
        ):
            """
            The main logic for wrapping create functions with monitoring data
            logging.
            This internal function porovides a template for both sync
            and async activations (helps with wrapping both "create"
            and "acreate").
            """

            is_stream = kwargs.get("stream", False)
            is_async = super_function.__name__ == "acreate"

            response = None

            # will be used only when stream is enabled
            stream_start_time = None

            async def _inner_log_message(is_exception):
                return await call_non_blocking_sync_or_async(
                    export_function,
                    (
                        cls._get_logging_message(
                            kwargs,
                            start_time,
                            is_exception,
                            is_async,
                            stream_start_time,
                            response,
                        ),
                        kwargs.get(
                            CONTEXT_ID_ARG_NAME,
                            response["id"] if response else None,
                        ),
                        kwargs.get(EXPORT_TIMESTAMP_ARG_NAME, start_time),
                    ),
                )

            log_message = add_conditional_sampling(
                _inner_log_message, sampling_ratio
            )

            start_time = time.time()

            async def inner_super_function():
                # Call the actual openai create function without the Mona
                # specific arguments.
                return await call_non_blocking_sync_or_async(
                    super_function,
                    args,
                    {
                        x: kwargs[x]
                        for x in kwargs
                        if not x.startswith(MONA_ARGS_PREFIX)
                    },
                )

            async def inner_handle_exception():
                if not specs.get("avoid_monitoring_exceptions", False):
                    await log_message(True)

            if not is_stream:
                try:
                    response = await inner_super_function()
                except Exception:
                    await inner_handle_exception()
                    raise

                await log_message(False)

                return response

            # From here it's stream handling.

            async def _stream_done_callback(
                final_response, actual_stream_start_time
            ):
                nonlocal response
                nonlocal stream_start_time
                # There is no usage data in returned stream responses, so
                # we add it here.
                response = final_response | {
                    "usage": get_usage(
                        model=get_model_param(kwargs),
                        prompt_texts=base_class._get_all_prompt_texts(kwargs),
                        response_texts=base_class._get_all_response_texts(
                            final_response
                        ),
                    )
                }
                stream_start_time = actual_stream_start_time
                await log_message(False)

            try:
                # Call the actual openai create function without the Mona
                # specific arguments.
                return ResponseGatheringIterator(
                    original_iterator=await inner_super_function(),
                    delta_choice_text_getter=(
                        base_class._get_stream_delta_text_from_choice
                    ),
                    final_choice_getter=base_class._get_final_choice,
                    callback=_stream_done_callback,
                )

            except Exception:
                await inner_handle_exception()
                raise

        @classmethod
        def create(cls, *args, **kwargs) -> dict:
            """
            A monitored version of the openai base class' "create"
            function.
            """
            return run_in_an_event_loop(
                cls._inner_create(logger.log, super().create, args, kwargs)
            )

        @classmethod
        async def acreate(cls, *args, **kwargs) -> dict:
            """
            An async monitored version of the openai base class'
            "acreate" function.
            """
            return await cls._inner_create(
                logger.alog, super().acreate, args, kwargs
            )

    return type(base_class.__name__, (MonitoredOpenAI,), {})


def get_rest_monitor(
    openai_endpoint_name: str,
    mona_creds: MonaCredsType,
    context_class: str,
    specs: Mapping = EMPTY_DICT,
    mona_clients_getter: Callable = get_mona_clients,
) -> type:
    """
    A wrapper around get_rest_monitor_with_logger that automatically uses
    a Mona logger.
    """
    return get_rest_monitor_with_logger(
        openai_endpoint_name,
        MonaLogger(mona_creds, context_class, mona_clients_getter),
        specs,
    )


def get_rest_monitor_with_logger(
    # TODO(itai): Consider understanding endpoint name from complete url.
    openai_endpoint_name: str,
    logger: Logger,
    specs: Mapping = EMPTY_DICT,
) -> type:
    """
    Returns a client class for monitoring OpenAI REST calls not done
    using the OpenAI python client (e.g., for Azure users using their
    endpoints directly). This isn't a wrapper for any http requesting
    library and doesn't call the OpenAI API for you - it's just an easy
    logging client to log requests, responses and exceptions.
    """

    logger.start_monitoring(openai_endpoint_name)

    sampling_ratio = validate_and_get_sampling_ratio(specs)

    wrapping_logic = get_endpoint_wrapping(openai_endpoint_name, specs)

    class RestClient:
        """
        This will be the returned monitoring class. We follow
        OpenAI's way of doing things by using a static classe with
        relevant class methods.
        """

        @classmethod
        def _inner_log_request(
            cls,
            message_logging_function: Callable,
            request_dict: Mapping,
            additional_data: Mapping = EMPTY_DICT,
            context_id: Optional[str] = None,
            export_timestamp: Optional[float] = None,
        ) -> tuple[Callable, Callable]:
            """
            Actual logic for logging requests, responses and exceptions.
            """
            start_time = time.time()

            if additional_data is None:
                additional_data = EMPTY_DICT

            def _inner_log_message(
                is_exception: bool,
                more_additional_data: Mapping,
                response: Optional[Mapping] = None,
            ):
                return message_logging_function(
                    _get_logging_message(
                        api_name=openai_endpoint_name,
                        request_input=request_dict,
                        start_time=start_time,
                        is_exception=is_exception,
                        is_async=False,
                        # TODO(itai): Support stream in REST as well.
                        stream_start_time=None,
                        response=response,
                        analysis_getter=wrapping_logic.get_full_analysis,
                        message_cleaner=wrapping_logic.get_clean_message,
                        additional_data={
                            **additional_data,
                            **more_additional_data,
                        },
                    ),
                    context_id,
                    export_timestamp,
                )

            log_message = add_conditional_sampling(
                _inner_log_message, sampling_ratio
            )

            def log_response(
                response: Mapping, additional_data: Mapping = EMPTY_DICT
            ):
                """
                Only when this function is called, will data be logged
                out. This function should be called with a
                response object from the OpenAI API as close as
                possible to when it is received to allow accurate
                latency logging.
                """
                return log_message(
                    False,
                    more_additional_data=additional_data,
                    response=response,
                )

            def log_exception(additional_data: Mapping = EMPTY_DICT):
                return log_message(True, more_additional_data=additional_data)

            return log_response, log_exception

        @classmethod
        def log_request(
            cls,
            request_dict: Mapping,
            additional_data: Mapping = EMPTY_DICT,
            context_id: Optional[str] = None,
            export_timestamp: Optional[float] = None,
        ):
            """
            Sets up logging for OpenAI request/response objects.

            This function should be called with a request data dict,
            for example, what you would use as "json" when using
            "requests" to post.

            It returns a response logging function to be used with the
            response object, as well as an exception logging function in case
            of exceptions.

            Note that this call does not log anything until one of the
            returned callbacks is called.
            """
            return cls._inner_log_request(
                logger.log,
                request_dict,
                additional_data,
                context_id,
                export_timestamp,
            )

        @classmethod
        def async_log_request(
            cls,
            request_dict: Mapping,
            additional_data: Mapping = EMPTY_DICT,
            context_id: Optional[str] = None,
            export_timestamp: Optional[float] = None,
        ):
            """
            Async version of "log_request". See function's docstring for more
            details.
            """
            return cls._inner_log_request(
                logger.alog,
                request_dict,
                additional_data,
                context_id,
                export_timestamp,
            )

    return RestClient


def _validate_langchain_llm(llm) -> None:
    if not hasattr(llm, "client"):
        raise InvalidLagnchainLLMException(
            "LLM has no client attribute - must be an OpenAI LLM"
        )


def monitor_langchain_llm(
    llm,
    mona_creds: MonaCredsType,
    context_class: str,
    specs: Mapping = EMPTY_DICT,
    mona_clients_getter: Callable = get_mona_clients,
):
    """
    Wraps given llm with automatic mona-monitoring logic.
    """
    _validate_langchain_llm(llm)
    llm.client = monitor(
        llm.client, mona_creds, context_class, specs, mona_clients_getter
    )
    return llm


def monitor_langchain_llm_with_logger(
    llm, logger: Logger, specs: Mapping = EMPTY_DICT
):
    """
    Wraps given llm with monitoring logic, logging the analysis with the given
    logger.
    """
    _validate_langchain_llm(llm)
    llm.client = monitor_with_logger(llm.client, logger, specs)
    return llm
