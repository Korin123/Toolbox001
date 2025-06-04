import logging
import os
import requests

import azure.functions as func
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.textanalytics import TextAnalyticsClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from src.helpers.data_loading import load_pymupdf_pdf
from src.helpers.common import MeasureRunTime

load_dotenv()

bp_translate_document_augmented = func.Blueprint()
FUNCTION_ROUTE = "translate_document_augmented"

# Environment
TRANSLATOR_ENDPOINT = os.getenv("TRANSLATOR_ENDPOINT")
TRANSLATOR_REGION = os.getenv("TRANSLATOR_REGION")

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
LANGUAGE_ENDPOINT = os.getenv("LANGUAGE_ENDPOINT")

# Token providers
credential = DefaultAzureCredential()
translator_token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)
text_analytics_client = TextAnalyticsClient(
    endpoint=LANGUAGE_ENDPOINT, credential=credential
)

# Azure OpenAI client
aoai_token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    azure_ad_token_provider=aoai_token_provider,
    api_version="2024-06-01",
    timeout=90,
    max_retries=2,
)


def translate_text(text: str, to_lang: str = "en") -> str:
    """Translate text using Azure Translator."""
    url = f"{TRANSLATOR_ENDPOINT}/translate"
    params = {"api-version": "3.0", "to": to_lang}
    headers = {
        "Authorization": f"Bearer {translator_token_provider()}",
        "Content-Type": "application/json",
    }
    if TRANSLATOR_REGION:
        headers["Ocp-Apim-Subscription-Region"] = TRANSLATOR_REGION
    res = requests.post(url, params=params, headers=headers, json=[{"text": text}])
    res.raise_for_status()
    return res.json()[0]["translations"][0]["text"]


class PageTranslation(BaseModel):
    page: int
    original: str
    translation: str


class FunctionResponseModel(BaseModel):
    success: bool = Field(False)
    pages: list[PageTranslation] | None = None
    entities: list[str] | None = None
    summary: str | None = None
    func_time_taken_secs: float | None = None
    error_text: str | None = None


SUMMARY_PROMPT = (
    "Provide a concise summary of the document."
)


@bp_translate_document_augmented.route(route=FUNCTION_ROUTE)
def translate_document_augmented(req: func.HttpRequest) -> func.HttpResponse:
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

        pages: list[PageTranslation] = []
        combined_translation = []
        for idx, page in enumerate(pdf.pages(), start=1):
            original_text = page.get_text()
            translation = translate_text(original_text)
            pages.append(PageTranslation(page=idx, original=original_text, translation=translation))
            combined_translation.append(translation)
        output.pages = pages

        # Entity extraction on the translated text
        translated_text = "\n".join(combined_translation)
        entities_result = text_analytics_client.recognize_entities([translated_text])
        doc = entities_result[0]
        if not doc.is_error:
            output.entities = [ent.text for ent in doc.entities]

        # Summarize with Azure OpenAI
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": translated_text},
        ]
        llm_res = aoai_client.chat.completions.create(
            messages=messages, model=AOAI_LLM_DEPLOYMENT
        )
        output.summary = llm_res.choices[0].message.content

        output.success = True
        output.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )
    except Exception:
        logging.exception("An error occurred during processing.")
        output.func_time_taken_secs = func_timer.stop() if 'func_timer' in locals() else None
        output.error_text = "An error occurred during processing."
        return func.HttpResponse(
            body=output.model_dump_json(),
            mimetype="application/json",
            status_code=500,
        )
