import logging
import os

import azure.functions as func
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

load_dotenv()

aoai_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

bp_summarize_text = func.Blueprint()
FUNCTION_ROUTE = "summarize_text"

# Load environment variables
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")

# Setup Haystack components
aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    azure_ad_token_provider=aoai_token_provider,
    api_version="2024-06-01",
)


# Setup Pydantic models for validation of the request and response
class FunctionRequestModel(BaseModel):
    """
    Defines the schema that will be expected in the request body. We'll use this to
    ensure that the request contains the correct values and structure, and to allow
    a partially filled request to be processed in case of an error.
    """

    text: str = Field(description="The text to be summarized")
    summary_style: str = Field(description="The style of the summary to be generated")
    num_sentences: int = Field(
        description="The desired number of sentences for the summary", gt=0
    )


LLM_SYSTEM_PROMPT = (
    "You are an expert in summarising and adapting information. "
    "Your task is to:\n"
    "1. Summarise the following text into exactly {} sentences, "
    "written in the style of '{}'.\n"
    "2. Then, list the key factual points as bullet points in plain English.\n\n"
    "Return the summary first, followed by the bullet points."
)


@bp_summarize_text.route(route=FUNCTION_ROUTE)
def summarize_text(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info(
            f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request."
        )
        # Check the request body
        req_body = req.get_json()
        request_obj = FunctionRequestModel(**req_body)
    except Exception as _e:
        return func.HttpResponse(
            (
                "The request body was not in the expected format. Please ensure that it is "
                "a valid JSON object with the following fields: {}"
            ).format(list(FunctionRequestModel.model_fields.keys())),
            status_code=422,
        )
    try:
        # Create the messages to send to the LLM
        input_messages = [
            {
                "role": "system",
                "content": LLM_SYSTEM_PROMPT.format(
                    request_obj.num_sentences, request_obj.summary_style
                ),
            },
            {
                "role": "user",
                "content": request_obj.text,
            },
        ]
        # Send request to LLM
        llm_result = aoai_client.chat.completions.create(
            messages=input_messages,
            model=AOAI_LLM_DEPLOYMENT,
        )
        llm_text_response = llm_result.choices[0].message.content
        # All steps completed successfully, set success=True and return the final result
        return func.HttpResponse(
            body=llm_text_response,
            status_code=200,
        )
    except Exception as _e:
        logging.exception("An error occurred during processing.")
        return func.HttpResponse(
            "An error occurred during processing.",
            status_code=500,
        )
