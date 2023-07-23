"""
A utility module for everything realted to encoding tokens.
"""
import tiktoken
from collections.abc import Iterable


def _get_number_of_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _get_encoding(model: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(model)


def get_usage(model: str, prompt_texts: Iterable[str], response_texts: Iterable[str]) -> dict:
    """
    Returns a usage dict containing the number of tokens in the prompt, in the
    response, and totally.
    """
    enc = _get_encoding(model)

    def get_tokens_sum(texts):
        return sum(_get_number_of_tokens(text, enc) for text in texts)

    usage = {
        "prompt_tokens": get_tokens_sum(prompt_texts),
        "completion_tokens": get_tokens_sum(response_texts),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    return usage
