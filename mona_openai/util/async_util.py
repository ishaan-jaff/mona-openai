import nest_asyncio
import asyncio
from .general_consts import EMPTY_DICT
from inspect import iscoroutinefunction
from collections.abc import Mapping, Coroutine, Callable


def run_in_an_event_loop(coroutine: Coroutine):
    """
    A light wrapper around asyncio.run to avoid crushing when trying to run a
    coroutine in an environment where an event loop is already in place and
    asyncio.run doesn't work.
    """
    try:
        return asyncio.run(coroutine)
    except RuntimeError:
        try:
            return asyncio.get_event_loop().run_until_complete(coroutine)
        except RuntimeError:
            # This happens in environments that already have an event loop
            # that is "run forever". We therefor must allow a "nested" event
            # loop that we can run within the main loop.
            nest_asyncio.apply()
            return asyncio.run(coroutine)


async def call_non_blocking_sync_or_async(
    function: Callable, func_args=(), func_kwargs: Mapping = EMPTY_DICT
):
    """
    A higher order function that allows calling both sync and async
    functions as if they were async, avoid blocking when relevant, and
    maintain one code base for both cases.
    """
    if iscoroutinefunction(function):
        return await function(*func_args, **func_kwargs)
    return function(*func_args, **func_kwargs)
