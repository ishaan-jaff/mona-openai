from typing import Type, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import openai

CompletionType = Type["openai.api_resources.completion.Completion"]
ChatCompletionType = Type[
    "openai.api_resources.chat_completion.ChatCompletion"
]
SupportedOpenAIClassesType = Union[CompletionType, ChatCompletionType]
