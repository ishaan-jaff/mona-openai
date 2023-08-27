"""
The Mona wrapping code for OpenAI's Completion API.
"""
from copy import deepcopy
from functools import wraps

from ..analysis.privacy import get_privacy_analyzers, PrivacyAnalyzer
from ..analysis.profanity import get_has_profanity, get_profanity_prob
from ..analysis.textual import get_textual_analyzers, TextualAnalyzer
from ..util.oop_util import create_combined_object
from .endpoint_wrapping import OpenAIEndpointWrappingLogic
from collections.abc import Callable, Mapping, Iterable

COMPLETION_CLASS_NAME = "Completion"


def _get_prompts(request: Mapping) -> Iterable[str]:
    prompts = request.get("prompt", ())
    return (prompts,) if isinstance(prompts, str) else prompts


def _get_choices_texts(response: Mapping) -> Iterable[str]:
    return tuple((choice["text"] for choice in response["choices"]))


def _get_texts(func: Callable) -> Callable:
    def wrapper(self, input: Mapping, response: Mapping):
        return func(self, _get_prompts(input), _get_choices_texts(response))

    return wrapper


def _get_analyzers(analyzers_getter: Callable) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, prompts: Iterable[str], answers: Iterable[str]):
            return func(
                self, analyzers_getter(prompts), analyzers_getter(answers)
            )

        return wrapper

    return decorator


class CompletionWrapping(OpenAIEndpointWrappingLogic):
    def _get_endpoint_name(self) -> str:
        return COMPLETION_CLASS_NAME

    def _internal_get_clean_message(self, message: Mapping) -> Mapping:
        """
        Returns a copy of the given message with relevant data removed, for
        example the actual texts, to avoid sending such information, that
        is sometimes sensitive, to Mona.
        """
        new_message = deepcopy(message)
        if not self._specs.get("export_prompt", False):
            new_message["input"].pop("prompt", None)

        if "response" in message and not self._specs.get(
            "export_response_texts", False
        ):
            for choice in new_message["response"]["choices"]:
                choice.pop("text", None)

        return new_message

    @_get_texts
    @_get_analyzers(get_privacy_analyzers)
    def _get_full_privacy_analysis(
        self, prompts_privacy_analyzers: Iterable[PrivacyAnalyzer], answers_privacy_analyzers: Iterable[PrivacyAnalyzer]
    ) -> dict:
        combined_prompts = create_combined_object(prompts_privacy_analyzers)
        combined_answers = create_combined_object(answers_privacy_analyzers)
        return {
            "prompt_phone_number_count": (
                combined_prompts.get_phone_numbers_count()
            ),
            "answer_unknown_phone_number_count": (
                combined_answers.get_previously_unseen_phone_numbers_count(
                    prompts_privacy_analyzers
                )
            ),
            "prompt_email_count": combined_prompts.get_emails_count(),
            "answer_unknown_email_count": (
                combined_answers.get_previously_unseen_emails_count(
                    prompts_privacy_analyzers
                )
            ),
        }

    @_get_texts
    @_get_analyzers(get_textual_analyzers)
    def _get_full_textual_analysis(
        self, prompts_textual_analyzers: Iterable[TextualAnalyzer], answers_textual_analyzers: Iterable[TextualAnalyzer]
    ) -> dict:
        combined_prompts = create_combined_object(prompts_textual_analyzers)
        combined_answers = create_combined_object(answers_textual_analyzers)
        return {
            "prompt_length": combined_prompts.get_length(),
            "answer_length": combined_answers.get_length(),
            "prompt_word_count": combined_prompts.get_word_count(),
            "answer_word_count": combined_answers.get_word_count(),
            "prompt_preposition_count": (
                combined_prompts.get_preposition_count()
            ),
            "prompt_preposition_ratio": (
                combined_prompts.get_preposition_ratio()
            ),
            "answer_preposition_count": (
                combined_answers.get_preposition_count()
            ),
            "answer_preposition_ratio": (
                combined_answers.get_preposition_ratio()
            ),
            "answer_words_not_in_prompt_count": (
                combined_answers.get_words_not_in_others_count(
                    prompts_textual_analyzers
                )
            ),
            "answer_words_not_in_prompt_ratio": tuple(
                analyzer.get_words_not_in_others_count(
                    prompts_textual_analyzers
                )
                / analyzer.get_word_count()
                if analyzer.get_word_count() > 0
                else 0.0
                for analyzer in answers_textual_analyzers
            ),
        }

    @_get_texts
    def _get_full_profainty_analysis(self, prompts: Iterable[str], answers: Iterable[str]) -> dict:
        return {
            "prompt_profanity_prob": get_profanity_prob(prompts),
            "prompt_has_profanity": get_has_profanity(prompts),
            "answer_profanity_prob": get_profanity_prob(answers),
            "answer_has_profanity": get_has_profanity(answers),
        }

    def get_stream_delta_text_from_choice(self, choice: Mapping) -> str:
        return choice["text"]

    def get_final_choice(self, text: str) -> dict:
        return {"text": text}

    def get_all_prompt_texts(self, request: Mapping) -> Iterable[str]:
        return _get_prompts(request)

    def get_all_response_texts(self, response: Mapping) -> Iterable[str]:
        return _get_choices_texts(response)
