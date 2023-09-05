from os import environ
import requests
from mona_openai import get_rest_monitor
import litellm
from litellm import completion # pip install litellm

litellm.api_key = environ.get("OPEN_AI_KEY")

MONA_API_KEY = environ.get("MONA_API_KEY")
MONA_SECRET = environ.get("MONA_SECRET")
MONA_CREDS = {
    "key": MONA_API_KEY,
    "secret": MONA_SECRET,
}

# This is the name of the monitoring class on Mona
MONITORING_CONTEXT_NAME = "MONITORED_CHAT_COMPLETION_USE_CASE_NAME"


# Direct REST usage, without OpenAI client

# Get Mona logger
mona_logger = get_rest_monitor(
    "ChatCompletion",
    MONA_CREDS,
    MONITORING_CONTEXT_NAME,
)

# Set up the request data
data = {
    "messages": [
        {"role": "user", "content": "I want to generate some text about "}
    ],
    "max_tokens": 20,
    "temperature": 0.2,
    "model": "gpt-3.5-turbo",
    "n": 1,
}

# The log_request function returns two other function for later logging
# the response or the exception. When we later do that, the logger will
# actually calculate all the relevant metrics and will send them to
# Mona.
response_logger, exception_logger = mona_logger.log_request(
    data, additional_data={"customer_id": "A531251"}
)

try:
    # Send the request to the API using litellm
    response = completion(
        **data
    )
    # Log response to Mona
    response_logger(response)
    print(response["choices"][0]["message"]["content"])

except Exception:
    # Log exception to Mona
    exception_logger()
