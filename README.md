# Mona-OpenAI Integration Client
<p align="center">
  <img src="https://github.com/monalabs/mona-sdk/blob/main/mona_logo.png?raw=true" alt="Mona's logo" width="180"/>
</p>


Use one line of code to get instant live monitoring for your OpenAI usage including:
* Tokens usage
* Hallucination alerts
* Profanity and privacy analyses
* Behavioral drifts and anomalies
* Much much more.

## Setting Up

- TODO: Add part about Mona registration with a link to the relevant landing page where the reader can sign up.
```console
$ pip install mona_openai
```

## Quick Start and Example

Example boilerplate code for both sync and async usage is given in the file "example_usage.py" and here:
```py
from os import environ
import asyncio
import openai

from mona_openai import monitor

openai.api_key = environ.get("OPEN_AI_KEY")

MONA_API_KEY = environ.get("MONA_API_KEY")
MONA_SECRET = environ.get("MONA_SECRET")
MONA_CREDS = {
    "key": MONA_API_KEY,
    "secret": MONA_SECRET,
}
CONTEXT_CLASS_NAME = "SOME_MONITORING_CONTEXT_NAME"


monitored_completion = monitor(
    openai.Completion,
    MONA_CREDS,
    CONTEXT_CLASS_NAME,
)


prompt = "I want to generate some text about "
model = "text-ada-001"
temperature = 0.6
max_tokens = 5
n = 1

# Regular (sync) usage
response = monitored_completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    n=n,
    temperature=temperature,
    MONA_additional_data={"customer_id": "A531251"},
)
print(response.choices[0].text)

# Async usage
response = asyncio.run(monitored_completion.acreate(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    n=n,
    temperature=temperature,
    MONA_additional_data={"customer_id": "A531251"},
))

print(response.choices[0].text)
```
## Supported OpenAI APIs
Currently this client supports `openai.Completion` and `openai.ChatCompletion`. Mona can support processes based on other APIs and also non-OpenAI-based apps.
If you have a differrent use-case, we'd love to hear about it! Please email us at support@monalabs.io.

## Usage
### Initialization

The main and only function exposed in this package is `monitor`.
```py
import openai

from mona_openai import monitor

openai.api_key = environ.get("OPEN_AI_KEY")

monitored_completion = monitor(
    openai.Completion,
    (
        environ.get("MONA_API_KEY"),
        environ.get("MONA_SECRET"),
    ),
    "SOME_MONITORING_CONTEXT_NAME",
    {"analysis": {"profanity": False}}
)

...

monitored_completion.create(...)
```

The `monitor` function returns to you a class that wraps the original openai endpoint class you provide, with an equivalent API (besides some additions listed below).
You can then use the returned class' "create" and "acreate" functions just as you would before, only now, besides getting the requested openAI functionality, this client will log out to Mona's server the parameters you used (e.g., temperature), data about the response from OpenAI's server, and custom analyses about the call (e.g., profanity scores, privacy checks for emails/phone numbers found in the texts, textual analyses, etc...)

The `monitor` function receives the following arguments:
openai_class: An OpenAI API class to wrap with monitoring capabilties.
mona_creds: A dict (containing "key" and "secret") or pair (tuple) of Mona API key and secret to set up Mona's clients from its SDK.
context_class: The Mona context class name to use for monitoring. Use a constant name of your choice.
specs: A dictionary of specifications such as monitoring sampling ratio.

#### Specs
The specs arg allows you to configure what should be monitored. It expects a python dict with the follwoing possible keys:
* sampling_ratio (1): A number between 0 and 1 for how often should the call be logged.
* avoid_monitoring_exceptions (False): Whether or not to log out to Mona when there is an OpenAI exception. Default is to track exceptions - and Mona will alert you on things like a jump in number of exceptions
* export_prompt (False): Whether Mona should export the actual prompt text. Be default set to False to avoid privacy concerns.
* export_response_texts (False): Whether Mona should export the actual response texts. Be default set to False to avoid privacy concerns.
* analysis: A dictionary mapping each analysis type to a boolean value telling the client whether or not to run said analysis and log it to Mona. Possible options currently are "privacy", "profanity", and "textual". By default, all analyses take place and are logged out to Mona.

### Capabilities during API calls

After wrapping your endpoint with `monitor`, you really don't need to do anything else. When using `create` or `acreate` data will be tracked and monitoring will take place.

There are, however, several capabilities that are added to these functions. Specifically, you can add the following arguments to any create call:
* MONA_context_id: The unique id of the context in which the call is made. By using this ID you can export more data to Mona to the same context from other places. If not supplied, the "id" field of the OpenAI Endpoint's response will be used as the Mona context ID automatically.
* MONA_export_timestamp: Can be used to simulate as if the current call was made in a different time, as far as Mona is concerned.
* MONA_additional_data: A JSON-serializable dict with any other data you want to add to the monitoring context. This comes in handy if you want to add more information to the monitoring contex that isn't part of the basic OpenAI API call information. For example, if you are using a specific template ID or if this call is being made for a specific customer ID, these are fields you can add there to help get full context when monitoring with Mona.

Example:
```py
response = asyncio.run(monitored_completion.acreate(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    n=n,
    temperature=temperature,
    MONA_additional_data={"customer_id": "A531251"},
))
print(response.choices[0].text)
```

### Using OpenAI with REST calls instead of OpenAI's Python client
In some cases you may choose to use OpenAI's API directly with REST calls and not using OpenAI's SDK. For these cases we allow a more direct approach for logging to Mona as well, by using the "get_rest_monitor" function. See example below.

```py
# Direct REST usage, without OpenAI client
import requests
from os import environ
from mona_openai import get_rest_monitor


MONA_API_KEY = environ.get("MONA_API_KEY")
MONA_SECRET = environ.get("MONA_SECRET")
MONA_CREDS = {
    "key": MONA_API_KEY,
    "secret": MONA_SECRET,
}
CONTEXT_CLASS_NAME = "SOME_MONITORING_CONTEXT_NAME"

# Get Mona logger
mona_logger = get_rest_monitor(
    "Completion",
    MONA_CREDS,
    CONTEXT_CLASS_NAME,
)

# Set up the API endpoint URL and authentication headers
url = "https://api.openai.com/v1/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {environ.get('OPEN_AI_KEY')}",
}

# Set up the request data
data = {
    "prompt": prompt,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "model": model,
    "n": n,
}

# The log_request function returns two other function for later logging
# the response or the exception. When we later do that, the logger will
# actually calculate all the relevant metrics and will send them to
# Mona.
response_logger, exception_logger = mona_logger.log_request(
    data, additional_data={"customer_id": "A531251"}
)

try:
    # Send the request to the API
    response = requests.post(url, headers=headers, json=data)

    # Check for HTTP errors
    response.raise_for_status()

    # Log response to Mona
    response_logger(response.json())
    print(response.json()["choices"][0]["text"])

except Exception as err:
    # Log exception to Mona
    exception_logger()
```

### Stream support

OpenAI allows receiving responses as a stream of tokens using the "stream" parameter. When this is done, Mona will collect all the tokens in memory and will create the analysis and log out the data the moment the stream is over. You don't need to do anything to make this happen.

Since for streaming responses OpenAI doesn't supply the full usage tokens summary, Mona uses the tiktoken package to calculate the tokens of the prompt and completion and log them for monitoring.

NOTE: Stream is currently only supported with SDK usage, and not with using REST directly.


## Mona SDK

This package uses the mona_sdk package to export the relevant data to Mona. There are several environment variables you can use to configure the SDK's behavior. For example, you can set it up to raise exceptions when exporting data to Mona fails (it doesn't do that by default).

## Monitoring for profanity

Mona uses the alt-profanity-check pacakge (https://pypi.org/project/alt-profanity-check/) to create both boolean predictions and probabilty scores for the existence of profanity both in the prompt and in the responses. We use the built in package methods for that. If you want, for example, to use a different probability threshold for the boolean prediction, you can do that by changing your Mona config on the Mona dashboard.