# function_app/bp_intelligent_extraction.py

import json
import logging
import os
import time
import random
from typing import Optional

import azure.functions as func
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError
from pydantic import BaseModel, Field

# Assuming these shared components are accessible from your project structure
from src.components.doc_intelligence import (
    VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES,
    DocumentIntelligenceProcessor,
    DefaultDocumentPageProcessor
)
from src.helpers.common import MeasureRunTime

load_dotenv()

# --- NEW: Define the different instructions for the LLM ---
# The keys ('extract_details', 'summarize', 'find_risks') will be sent from the Gradio dropdown.
PROMPT_INSTRUCTIONS = {
    "extract_details": (
        "You are an expert investigative analyst. Read the following document content. "
        "Your task is to extract all names of people, all specific locations mentioned (cities, addresses), and all vehicle descriptions. "
        "Provide the output as a clean, readable text or markdown."
    ),
    "summarize": (
        "You are a helpful AI assistant. Read the following document content and provide a concise, "
        "one-paragraph summary of the key events described in the text."
    ),
    "find_risks": (
        "You are a safety officer. Read the following document content and identify "
        "any potential safety hazards or procedural violations. List them as bullet points. "
        "If no hazards are found, state 'No safety hazards identified'."
    ),
}


# --- Setup and Configuration ---
bp_intelligent_extraction = func.Blueprint()
FUNCTION_ROUTE = "intelligent_extraction" # The route for our new function

# Reuse existing client configurations
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
DOC_INTEL_MODEL_ID = "prebuilt-read"
MAX_LLM_RETRIES = 5

aoai_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

# Clients
di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=DefaultAzureCredential(),
)
aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    azure_ad_token_provider=aoai_token_provider,
    api_version="2024-06-01",
)

# Use a simplified processor for just getting text
doc_intel_text_processor = DocumentIntelligenceProcessor(
    page_processor=DefaultDocumentPageProcessor(page_img_order=None), # We only need text
)


# --- Pydantic Models for the Response ---
class FunctionResponseModel(BaseModel):
    success: bool = Field(False)
    llm_result: Optional[str] = None
    selected_instruction: Optional[str] = None
    func_time_taken_secs: Optional[float] = None
    error_text: Optional[str] = None
    di_extracted_text: Optional[str] = None


# --- The Main Azure Function ---
@bp_intelligent_extraction.route(route=FUNCTION_ROUTE)
def intelligent_extraction(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request.")
    output_model = FunctionResponseModel()
    func_timer = MeasureRunTime()
    func_timer.start()

    try:
        # --- 1. Request Parsing (Handles Dropdown Value + File) ---
        logging.info("Parsing multipart form data...")
        form = req.files
        json_payload_part = form.get("json")
        payload = json.loads(json_payload_part.read().decode())
        selected_option = payload.get("selected_option", "extract_details") # Default option
        output_model.selected_instruction = selected_option
        
        file_part = form.get("file")
        if not file_part:
            raise ValueError("File part is missing from the request.")
            
        req_body = file_part.read()
        mime_type = file_part.mimetype
        
        if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
            raise ValueError(f"Unsupported Content-Type: {mime_type}")

        # --- 2. Document Intelligence Step (Always Runs First) ---
        logging.info("Processing document with Document Intelligence...")
        with MeasureRunTime() as di_timer:
            poller = di_client.begin_analyze_document(
                model_id=DOC_INTEL_MODEL_ID,
                analyze_request=AnalyzeDocumentRequest(bytes_source=req_body),
            )
            di_result = poller.result()
        
        processed_docs = doc_intel_text_processor.process_analyze_result(di_result)
        merged_docs = doc_intel_text_processor.merge_adjacent_text_content_docs(processed_docs)
        extracted_text = "\n".join(doc.content for doc in merged_docs if doc.content)
        output_model.di_extracted_text = extracted_text
        logging.info(f"Document Intelligence completed in {di_timer.time_taken:.2f} seconds.")

        if not extracted_text:
            raise ValueError("Document Intelligence extracted no text.")

        # --- 3. LLM Step (Uses the Selected Instruction) ---
        instruction = PROMPT_INSTRUCTIONS.get(selected_option)
        if not instruction:
            raise ValueError(f"Invalid option key selected: {selected_option}")

        logging.info(f"Sending request to LLM with instruction key: '{selected_option}'")
        input_messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Here is the document content to analyze:\n\n{extracted_text}"},
        ]

        for attempt in range(MAX_LLM_RETRIES):
            try:
                with MeasureRunTime() as llm_timer:
                    llm_result = aoai_client.chat.completions.create(
                        messages=input_messages,
                        model=AOAI_LLM_DEPLOYMENT,
                        temperature=0.1
                    )
                break # Success, exit retry loop
            except RateLimitError as e:
                logging.warning(f"Rate limit exceeded on attempt {attempt+1}. Retrying...")
                if attempt == MAX_LLM_RETRIES - 1: raise # Re-raise the last error
                time.sleep((2 ** attempt) + random.uniform(0, 1))

        logging.info(f"LLM call completed in {llm_timer.time_taken:.2f} seconds.")
        message = llm_result.choices[0].message
        if not message or not message.content:
            raise ValueError("LLM returned no content.")

        output_model.llm_result = message.content
        output_model.success = True
        output_model.func_time_taken_secs = func_timer.stop()
        
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )

    except Exception as e:
        logging.exception(f"An error occurred in {FUNCTION_ROUTE}")
        output_model.success = False
        output_model.error_text = str(e)
        output_model.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=500,
        )