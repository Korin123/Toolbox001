import json
import logging
import os

import azure.functions as func
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from src.helpers.data_loading import load_pymupdf_pdf
from src.helpers.common import MeasureRunTime

load_dotenv()

bp_translate_document = func.Blueprint()
FUNCTION_ROUTE = "translate_document"

# Azure OpenAI client setup
aoai_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")

aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    azure_ad_token_provider=aoai_token_provider,
    api_version="2024-06-01",
    timeout=90,
    max_retries=2,
)


class FunctionResponseModel(BaseModel):
    success: bool = Field(False)
    translation: str | None = None
    summary: str | None = None
    func_time_taken_secs: float | None = None
    error_text: str | None = None


LLM_PROMPT = (
    "Translate the following text into English. "
    "Then provide a short summary in English in three sentences or less. "
    "Respond in JSON with keys 'translation' and 'summary'."
)


@bp_translate_document.route(route=FUNCTION_ROUTE)
def translate_document(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(
        f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request."
    )
    output = FunctionResponseModel(success=False)
    try:
        func_timer = MeasureRunTime()
        func_timer.start()

        mime_type = req.headers.get("Content-Type")
        if mime_type != "application/pdf":
            return func.HttpResponse(
                (
                    "This function only supports a Content-Type of 'application/pdf'. "
                    f"Supplied file is of type {mime_type}"
                ),
                status_code=400,
            )

        pdf_bytes = req.get_body()
        if len(pdf_bytes) == 0:
            return func.HttpResponse(
                "Please provide a PDF in the request body.", status_code=400
            )

        pdf = load_pymupdf_pdf(pdf_bytes=pdf_bytes)
        text = "\n".join(page.get_text() for page in pdf)

        messages = [
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": text},
        ]

        llm_res = aoai_client.chat.completions.create(
            messages=messages, model=AOAI_LLM_DEPLOYMENT
        )
        llm_text = llm_res.choices[0].message.content

        try:
            llm_json = json.loads(llm_text)
            output.translation = llm_json.get("translation")
            output.summary = llm_json.get("summary")
        except Exception:
            output.error_text = "LLM response could not be parsed."
            output.func_time_taken_secs = func_timer.stop()
            logging.exception(output.error_text)
            return func.HttpResponse(
                body=output.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )

        output.success = True
        output.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )
    except Exception:
        logging.exception("An error occurred during processing.")
        return func.HttpResponse(
            "An error occurred during processing.", status_code=500
        )
