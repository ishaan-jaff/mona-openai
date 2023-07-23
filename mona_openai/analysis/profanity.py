"""
Logic to create profanity analysis.
"""
from collections.abc import Iterable
from profanity_check import predict_prob, predict

_DECIMAL_PLACES = 2


def get_profanity_prob(texts: Iterable[str]) -> tuple[float, ...]:
    return tuple(round(x, _DECIMAL_PLACES) for x in predict_prob(texts))


def get_has_profanity(texts: Iterable[str]) -> tuple[bool, ...]:
    return tuple(bool(x) for x in predict(texts))
