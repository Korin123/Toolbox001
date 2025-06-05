import base64
import json
import logging
import mimetypes
import os
import time
from collections import defaultdict
from io import BytesIO
from typing import Optional, Union

import fitz
import gradio as gr
import requests
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from gradio.themes import Base
from helpers import (
    base64_img_str_to_url,
    extract_frames_from_video_bytes,
    pil_img_to_base64_bytes,
    resize_img_by_max,
)
from PIL import Image
from moviepy import VideoFileClip  # <-- Added for video to mp3 conversion
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from azure.storage.blob import BlobServiceClient
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
load_dotenv()

# Get function endpoint from env vars. If not set, default to local host.
# Note: This assumes the function is being run prior to launching the web app.
FUNCTION_HOSTNAME = os.getenv("FUNCTION_HOSTNAME", "http://localhost:7071/")
FUNCTION_KEY = os.getenv("FUNCTION_KEY", "")
FUNCTION_ENDPOINT = os.path.join(FUNCTION_HOSTNAME, "api")

# Create clients for Azure services. If connecting to Azure resources, ensure
# the current Azure identity has the necessary permissions to access the
# storage & cosmosDB accounts.
# You can set the `additionalRoleAssignmentIdentityIds` property when deploying
# the solution to grant the necessary permissions to the logged-in user. See
# the README for more info.
USE_LOCAL_STORAGE_EMULATOR = os.getenv("USE_LOCAL_STORAGE_EMULATOR", "false") == "true"
LOCAL_STORAGE_ACCOUNT_CONNECTION_STRING = os.getenv(
    "LOCAL_STORAGE_ACCOUNT_CONNECTION_STRING"
)
STORAGE_ACCOUNT_ENDPOINT = os.getenv("STORAGE_ACCOUNT_ENDPOINT")
COSMOSDB_ACCOUNT_ENDPOINT = os.getenv("COSMOSDB_ACCOUNT_ENDPOINT")
COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME")
IS_COSMOSDB_AVAILABLE = COSMOSDB_ACCOUNT_ENDPOINT and COSMOSDB_DATABASE_NAME

credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_ENDPOINT, credential=credential)

if not IS_COSMOSDB_AVAILABLE:
    logging.warning(
        (
            "CosmosDB connection info is unavailable. Please ensure that CosmosDB is deployed "
            "and that the CosmosDB environment variables are set."
        )
    )

if USE_LOCAL_STORAGE_EMULATOR:
    # Connect to the local development storage account emulator using a connection string
    logging.info("Connecting to the local development storage account emulator.")
    blob_service_client = BlobServiceClient.from_connection_string(
        LOCAL_STORAGE_ACCOUNT_CONNECTION_STRING
    )
else:
    # Connect to the Azure storage account using identity auth
    logging.info("Connecting to the Azure storage account.")
    blob_service_client = BlobServiceClient(
        account_url=STORAGE_ACCOUNT_ENDPOINT, credential=credential
    )

### Add container for audio in and out
input_container = "audio-in"
output_container = "audio-transcript-out"



# Set authentication values based on the environment variables, defaulting to auth enabled
WEB_APP_USE_PASSWORD_AUTH = (
    os.getenv("WEB_APP_USE_PASSWORD_AUTH", "true").lower() == "true"
)
if WEB_APP_USE_PASSWORD_AUTH:
    try:
        WEB_APP_USERNAME = os.getenv("WEB_APP_USERNAME")
        WEB_APP_PASSWORD = os.getenv("WEB_APP_PASSWORD")
    except KeyError:
        raise RuntimeError(
            "`WEB_APP_USERNAME` and `WEB_APP_PASSWORD` must be set unless `WEB_APP_USE_PASSWORD_AUTH` is set to false."
        )
    auth_message = "Contact the site owner for the password"
    auth = (WEB_APP_USERNAME, WEB_APP_PASSWORD)
else:
    auth_message = None
    auth = None


### Load backend Content Understanding Schemas.
local_backend_cu_schema_path = os.path.join(
    os.path.dirname(__file__),
    "../function_app/config/content_understanding_schemas.json",
)
deployed_backend_cu_schema_path = os.path.join(
    os.path.dirname(__file__), "content_understanding_schemas.json"
)

if os.path.exists(local_backend_cu_schema_path):
    # App is being run locally - load the schema from the function app directory
    backend_cu_schema_path = local_backend_cu_schema_path
elif os.path.exists(deployed_backend_cu_schema_path):
    # App is being run in Azure - the schema config should have been copied during the packaging and
    # deployment process (as defined in azure.yaml).
    backend_cu_schema_path = deployed_backend_cu_schema_path
else:
    raise FileNotFoundError(
        (
            "Backend content understanding schemas file not found. Make sure the schema file is located in "
            "'<project_root>/function_app/config/content_understanding_schemas.json' if running locally, or "
            "is copied into the base directory when deploying the web app to Azure (defined in <project_root>/azure.yaml)."
        )
    )

with open(backend_cu_schema_path, "r") as json_file:
    backend_cu_schemas: dict[str, dict] = json.load(json_file)
    # Create a mapper that maps the backend analyzer ID to the human-friendly schema name.
    # Generate this in a tuple format, as expected by gr.Dropdown components
    BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY = defaultdict(list)
    for modality, mod_schemas in backend_cu_schemas.items():
        for analyzer_id, schema in mod_schemas.items():
            BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY[modality].append(
                (schema["name"], analyzer_id)
            )


### Load (and download) all required demo files.
def download_url_source_to_local(
    url_source: str, output_folder: str, save_name: str = None
) -> str:
    """
    Downloads a file from a URL source to a local path, returning the local
    path of the downloaded file.
    """
    # Download file to local path
    basename = os.path.basename(url_source).split("?")[0]
    if save_name:
        # Clean the save_name automatically
        accepted_chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ "
        )
        save_name = save_name.replace(" ", "_").replace("-", "_").lower()
        save_name = "".join([c for c in save_name if c in accepted_chars])
        ext = os.path.splitext(basename)[1]
        basename = f"{save_name}{ext}"
    local_path = os.path.join(output_folder, basename)
    if not os.path.exists(local_path):
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Downloading {url_source} to {local_path}")
        response = requests.get(url_source)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
    # Return local path
    return local_path


with open("demo_files/doc_intel_processing_samples.json", "r") as json_file:
    doc_intel_url_sources: dict[str, str] = json.load(json_file)

with open(
    "demo_files/content_understanding/video_file_web_links.json", "r"
) as json_file:
    cu_video_file_url_sources: dict[str, str] = json.load(json_file)

# For dropdown components, create a list of tuples of (display_name, filepath).
# For standard file inputs, create a list of file paths.
DEMO_DOC_INTEL_PROCESSING_FILE_TUPLES = [
    (
        name,
        download_url_source_to_local(
            url_source, "demo_files/temp/doc_intel_processing", name
        ),
    )
    for name, url_source in doc_intel_url_sources.items()
]
DEMO_CALL_CENTER_AUDIO_FILES = [
    os.path.join("demo_files/call_center_audio", fn)
    for fn in os.listdir("demo_files/call_center_audio")
    if fn.endswith((".wav"))
]
DEMO_PDF_FILES = [
    os.path.join("demo_files", fn)
    for fn in os.listdir("demo_files")
    if fn.endswith((".pdf"))
]
DEMO_CONTENT_UNDERSTANDING_DOCUMENT_TUPLES = [
    (
        os.path.splitext(os.path.basename(fn))[0],
        os.path.join("demo_files/content_understanding", fn),
    )
    for fn in os.listdir("demo_files/content_understanding")
    if fn.endswith((".pdf"))
] + DEMO_DOC_INTEL_PROCESSING_FILE_TUPLES
DEMO_CONTENT_UNDERSTANDING_VIDEO_TUPLES = [
    (
        os.path.splitext(os.path.basename(fn))[0],
        os.path.join("demo_files/content_understanding", fn),
    )
    for fn in os.listdir("demo_files/content_understanding")
    if fn.endswith((".mp4"))
] + [
    (
        name,
        download_url_source_to_local(url_source, "demo_files/temp/video_files", name),
    )
    for name, url_source in cu_video_file_url_sources.items()
]
DEMO_CONTENT_UNDERSTANDING_AUDIO_TUPLES = [
    (
        os.path.splitext(os.path.basename(fn))[0],
        os.path.join("demo_files/content_understanding", fn),
    )
    for fn in os.listdir("demo_files/content_understanding")
    if fn.endswith((".wav"))
] + [
    (os.path.splitext(os.path.basename(fn))[0], fn)
    for fn in DEMO_CALL_CENTER_AUDIO_FILES
]
DEMO_CONTENT_UNDERSTANDING_IMAGE_TUPLES = [
    (
        os.path.splitext(os.path.basename(fn))[0],
        os.path.join("demo_files/content_understanding", fn),
    )
    for fn in os.listdir("demo_files/content_understanding")
    if fn.endswith((".png", ".jpg", ".jpeg"))
]
DEMO_IMG_FILES = [
    os.path.join("demo_files", fn)
    for fn in os.listdir("demo_files")
    if fn.endswith((".png", ".jpg", ".jpeg"))
]
DEMO_ACCOUNT_OPENING_FORM_FILES = [
    os.path.join("demo_files/account_opening_forms", fn)
    for fn in os.listdir("demo_files/account_opening_forms")
    if fn.endswith((".pdf"))
]
DEMO_VISION_FILES = DEMO_PDF_FILES + DEMO_IMG_FILES
with open("demo_files/text_samples.json", "r") as json_file:
    text_samples_dict: dict[str, str] = json.load(json_file)
    TEXT_SAMPLES_TUPLES = [(label, text) for label, text in text_samples_dict.items()]

with open("demo_files/pii_redaction/pii_redaction_text_samples.json", "r") as json_file:
    pii_redaction_text_samples_dict: dict[str, str] = json.load(json_file)
    PII_REDACTION_TEXT_SAMPLE_TUPLES = [
        (label, text) for label, text in pii_redaction_text_samples_dict.items()
    ]
DEMO_PII_REDACTION_PDF_TUPLES = [
    (
        os.path.splitext(os.path.basename(fn))[0],
        os.path.join("demo_files/pii_redaction", fn),
    )
    for fn in os.listdir("demo_files/pii_redaction")
    if fn.endswith((".pdf"))
]

FILE_EXAMPLES_LABEL = "You can upload your own file, or select one of the following examples and then click 'Process File'."
AUDIO_EXAMPLES_LABEL = FILE_EXAMPLES_LABEL.replace("File", "Audio").replace(
    "upload", "record or upload"
)
TEXT_EXAMPLES_LABEL = FILE_EXAMPLES_LABEL.replace("File", "Text").replace(
    "upload your own file", "insert your own text"
)


def echo_input(input):
    """
    A simple function to echo the input. Useful for updating one gradio
    components when an input component changes (e.g. when a user selects
    an example from a dropdown, populate the preview component with the
    content).
    """
    return input


def milliseconds_to_simple_string(milliseconds):
    """Converts milliseconds to a string in the format 'mm:ss:ms'."""
    minutes = milliseconds // 60000
    seconds = (milliseconds % 60000) // 1000
    ms = milliseconds % 1000
    return f"{minutes:02}:{seconds:02}:{ms:03}"


def send_request(
    route: str,
    json_data: Optional[dict] = None,
    data: Optional[dict] = None,
    files: Optional[dict] = None,
    headers: Optional[dict] = None,
    force_json_content_type: bool = True,
    timeout: float = 60.0,  # Add a timeout parameter with a default value
) -> tuple[int, float, Union[str, dict]]:
    """Sends a POST request to the specified route on the function host."""
    if FUNCTION_KEY:
        headers = headers or {}
        headers["x-functions-key"] = FUNCTION_KEY
    start_time = time.time()
    response = requests.post(
        os.path.join(FUNCTION_ENDPOINT, route),
        json=json_data,
        data=data,
        files=files,
        headers=headers,
        timeout=timeout,  # Pass the timeout to requests.post
    )
    client_side_time_taken = f"{round(time.time() - start_time, 1)} seconds"
    # Check if the response is valid JSON
    if response.headers.get("Content-Type") == "application/json":
        response_content = response.json()
    else:
        response_content = response.content.decode("utf-8")
    # If the output component expects JSON, ensure the output is JSON.
    if response.status_code != 200:
        if force_json_content_type:
            response_content = json.dumps(
                {
                    "Error code": response.status_code,
                    "Error Reason": response.reason,
                    "Content": response_content,
                }
            )
        else:
            response_content = f"Error: [{response.status_code}] {response.reason}, Content: {response_content}"
    return (
        response.status_code,
        client_side_time_taken,
        response_content,
    )


def fitz_pdf_to_images(file: str) -> list[Image.Image]:
    """Converts a PDF to images using PyMuPDF."""
    fitz_pdf = fitz.open(file)
    imgs = []
    for page in fitz_pdf.pages():
        pix = page.get_pixmap(dpi=75)
        imgs.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return imgs


def render_visual_media_input(file: Optional[str]):
    """Renders visual media input into a Gradio component."""
    if not file:
        return gr.Gallery(
            value=None,
            type="pil",
            visible=True,
        )
    mime_type = mimetypes.guess_type(file)[0]
    if mime_type.startswith("image"):
        return gr.Gallery([file], columns=1, visible=True)
    elif mime_type == "application/pdf":
        fitz_pdf = fitz.open(file)
        page_thumbnails = fitz_pdf_to_images(fitz_pdf)
        return gr.Gallery(
            page_thumbnails,
            columns=min(len(page_thumbnails), 2),
            type="pil",
            visible=True,
        )
    else:
        raise NotImplementedError(f"Unsupported mime_type: {mime_type}")


### Form Extraction with confidence scores - HTTP example ###
with gr.Blocks(analytics_enabled=False) as form_extraction_with_confidence_block:
    # Define requesting function, which reshapes the input into the correct schema
    def form_ext_w_conf_upload(file: str):
        if file is None:
            gr.Warning(
                "Please select or upload a PDF/image file, then click 'Process File'."
            )
            return ("", "", None, {})
        # Get response from the API
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            (
                status_code,
                client_side_time_taken,
                response_content,
            ) = send_request(
                route="form_extraction_with_confidence",
                data=data,
                headers=headers,
                force_json_content_type=True,  # JSON gradio block requires this
            )
        # Extract the image with bounding boxes from the result
        if isinstance(response_content, dict) and response_content.get(
            "result_img_with_bboxes"
        ):
            result_img = gr.Image(
                value=Image.open(
                    BytesIO(
                        base64.b64decode(response_content.pop("result_img_with_bboxes"))
                    )
                ),
                height=600,
                width=750,
                visible=True,
            )

        else:
            result_img = gr.Image(value=None, visible=False)
        return (
            status_code,
            client_side_time_taken,
            result_img,
            response_content,
        )

    # Input components
    form_ext_w_conf_instructions = gr.Markdown(
        (
            "This example extracts key information from Bank Account application forms "
            "([Code Link](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_form_extraction_with_confidence.py))."
            "\n\nThe pipeline is as follows:\n"
            "1. PyMuPDF is used to convert the PDFs into images.\n"
            "2. Azure Document Intelligence extracts the raw text from the PDF along with confidence scores for each "
            "word/line.\n"
            "3. The image and raw extracted text are sent together in a multimodal request to GPT-4o. "
            "GPT-4o is instructed to extract all of the key information from the form in a structured format.\n"
            "4. GPT-4o's response is then cross-checked against the Document Intelligence response, combining GPT-4o's "
            "extracted values with the confidence scores from Document Intelligence.\n"
            "5. Finally, we look at all of the extracted values and their confidence scores to determine whether the "
            "result requires human review. If any of the values are missing or their confidence scores are below the "
            "minimum required threshold of 0.8 (80% confidence), the result is flagged for human review.\n\n"
            "This approach allows us to extract all of the required information in a structured format, while also "
            "having confidence that the extracted values are accurate. If the extracted values are below our minimum "
            "required confidence threshold, we can automatically route the form to our human review team. "
            "The final response also includes many of the intermediate outputs from the processing pipeline.\n"
            "To see how this process works, compare the outputs between the handwritten and machine-typed forms.\n"
            "* The machine-typed version has a higher confidence score for each extracted value, resulting in a "
            "`min_extracted_field_confidence_score` above the required threshold, meaning no human review is required.\n"
            "* The handwritten form on the other hand has a lower confidence score for each extracted value, resulting "
            "in a `min_extracted_field_confidence_score` below the required threshold, meaning the result is flagged "
            "for human review."
        ),
        show_label=False,
        line_breaks=True,
    )
    with gr.Row():
        form_ext_w_conf_file_upload = gr.File(
            label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
            file_count="single",
            type="filepath",
        )
        form_ext_w_conf_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )
    # Examples
    form_ext_w_conf_examples = gr.Examples(
        examples=DEMO_ACCOUNT_OPENING_FORM_FILES,
        inputs=[form_ext_w_conf_file_upload],
        label=FILE_EXAMPLES_LABEL,
        outputs=[
            form_ext_w_conf_input_thumbs,
        ],
        fn=render_visual_media_input,
        run_on_click=True,
    )
    form_ext_w_conf_process_btn = gr.Button("Process File", variant="primary")
    # Output components
    with gr.Column(render=False) as form_ext_w_conf_output_row:
        form_ext_w_conf_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            form_ext_w_conf_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            form_ext_w_conf_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        form_ext_w_conf_img_output = gr.Image(
            label="Extracted Field Locations", visible=False
        )
        form_ext_w_conf_output_json = gr.JSON(label="API Response")

    form_ext_w_conf_output_row.render()
    # Actions
    form_ext_w_conf_process_btn.click(
        fn=form_ext_w_conf_upload,
        inputs=[form_ext_w_conf_file_upload],
        outputs=[
            form_ext_w_conf_status_code,
            form_ext_w_conf_time_taken,
            form_ext_w_conf_img_output,
            form_ext_w_conf_output_json,
        ],
    )


### Content understanding examples ###
with gr.Blocks(analytics_enabled=False) as simple_cu_examples_block:
    cu_main_instructions = gr.Markdown(
        (
            "This tab contains Azure Content Understanding examples for Document, Video, Audio, and Image data modalities.\n"
            "Each example uses Azure Content Understanding to process a file, using one of the "
            "[pre-built schemas](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/config/content_understanding_schemas.json) "
            "included in the accelerator. These schemas can be modified to extract the specific fields required for your use case.\n"
            "For more information on the Content Understanding service, see these links:\n"
            "* [Overview](https://learn.microsoft.com/en-au/azure/ai-services/content-understanding/overview)\n"
            "* [Language and region support](https://learn.microsoft.com/en-au/azure/ai-services/content-understanding/language-region-support)\n"
            "* [File requirements, service quotas & limits, schema & field information](https://learn.microsoft.com/en-au/azure/ai-services/content-understanding/service-limits)\n"
        ),
        show_label=False,
        line_breaks=True,
    )

    def get_cu_value_field_name(field_info: dict) -> str:
        """Gets the name of the dictionary field containing the resulting value
        from a content understanding field result."""
        return next(
            (
                field_name
                for field_name, field_value in field_info.items()
                if field_name.startswith("value")
            ),
            None,
        )

    def format_extracted_field_output(
        text: str, confidence: Optional[float] = None
    ) -> str:
        if confidence is None:
            return text
        return f"{text} (Confidence: {round(confidence * 100, 1)}%)"

    def format_simple_cu_field_output(
        field_info: dict, include_confidence: bool = True
    ) -> str:
        field_name = get_cu_value_field_name(field_info)
        confidence = field_info.get("confidence")
        return format_extracted_field_output(
            field_info.get(field_name),
            confidence if (confidence is not None and include_confidence) else None,
        )

    def prebuilt_cu_document_process(
        analyzer_id: str,
        file: str,
        timeout: float = 120.0,  # Allow user to specify a longer timeout for large/complex docs
    ):
        analyzer_route = "content_understanding_document"
        processing_start_time = None
        fields_output = ""
        img_outputs = gr.Gallery(visible=False)
        try:
            if file is None:
                gr.Warning("Please select or upload a file, then click 'Process File'.")
                raise ValueError(
                    "Please select or upload a file, then click 'Process File'."
                )
            # Get response from the API
            with open(file, "rb") as f:
                payload = {"analyzer_id": analyzer_id}
                file_mime_type = mimetypes.guess_type(file)[0]
                files = {
                    "file": (file, f, file_mime_type),
                    "json": ("payload.json", json.dumps(payload), "application/json"),
                }

                status_code, processing_time_taken, response = send_request(
                    route=analyzer_route,
                    files=files,
                    force_json_content_type=True,
                    timeout=timeout,  # Pass through the timeout
                )
            # Format image and field outputs (if available in the response)
            if isinstance(response, dict) and response.get("formatted_fields_md"):
                fields_output = response["formatted_fields_md"]
            if isinstance(response, dict) and response.get("result_imgs_with_bboxes"):
                img_outputs = [
                    Image.open(BytesIO(base64.b64decode(img)))
                    for img in response.pop("result_imgs_with_bboxes").values()
                ]
                img_outputs = gr.Gallery(
                    img_outputs,
                    columns=min(len(img_outputs), 2),
                    type="pil",
                    visible=True,
                )
        except Exception as e:
            logging.exception("Error processing file")
            status_code = 500
            processing_time_taken = (
                f"{round(time.time() - processing_start_time, 1)} seconds"
                if processing_start_time is not None
                else "N/A"
            )
            response = {"Error": str(e)}
        return (
            status_code,
            processing_time_taken,
            img_outputs,
            gr.Markdown(
                f"<h3>Extracted Field Values</h3>\n\n{fields_output}",
                show_label=False,
                visible=True,
                line_breaks=True,
            ),
            gr.JSON(value=response, label="Full Function API Response", visible=True),
        )

    # Input components
    simple_cu_doc_header = gr.Label(
        "Azure Content Understanding - Document Processing", show_label=False
    )
    simple_cu_doc_instructions = gr.Markdown(
        (
            "This example extracts information from documents using Azure Content Understanding using one of the "
            "[pre-built schemas](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/config/content_understanding_schemas.json) "
            "included in the accelerator.\nThese schemas can be modified to extract the specific fields required for your use case "
            "([Code Link](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/content_understanding_document.py)). "
            "For more information: [Azure Content Understanding - Documents Overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/overview).\n"
            "Once a document is processed, the accelerator performs some enrichment and post-processing to draw bounding boxes around the extracted fields.\n"
            "To get started, select one of the existing schemas from the dropdown, then select a demo document or upload your own to have it processed."
        ),
        show_label=False,
        line_breaks=True,
    )
    with gr.Row():
        with gr.Column():
            simple_cu_doc_analyzer_schema_dropdown = gr.Dropdown(
                choices=BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY["document"],
                value=BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY["document"][0][1],
                label="Select field extraction schema",
                interactive=True,
            )
            cu_document_file_dropdown_options = [
                (
                    "",
                    None,
                )
            ] + DEMO_CONTENT_UNDERSTANDING_DOCUMENT_TUPLES
            simple_cu_doc_demo_file_dropdown = gr.Dropdown(
                choices=cu_document_file_dropdown_options,
                value="",
                label="[Optional] Select demo document",
                interactive=True,
                allow_custom_value=True,
            )
            simple_cu_doc_file_upload = gr.File(
                height=100,
                label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
                file_count="single",
                type="filepath",
            )
        with gr.Column():
            simple_cu_doc_input_thumbs = gr.Gallery(
                label="Input File Preview", object_fit="contain", visible=True
            )
    simple_cu_doc_process_btn = gr.Button("Process File", variant="primary")
    # Output components
    with gr.Column() as simple_cu_doc_output_row:
        simple_cu_doc_output_label = gr.Label(value="Result", show_label=False)
        with gr.Row():
            simple_cu_doc_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            simple_cu_doc_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        with gr.Row():
            simple_cu_doc_img_output = gr.Gallery(
                label="Extracted Field Locations", object_fit="contain", visible=False
            )
            with gr.Column():
                simple_cu_doc_fields_output_md = gr.Markdown(
                    "<h3>Extracted Field Values</h3>",
                    show_label=False,
                    visible=False,
                    line_breaks=True,
                )
        simple_cu_doc_output_json = gr.JSON(
            label="Full Function API Response", visible=False
        )

    # Actions
    simple_cu_doc_demo_file_dropdown.change(
        fn=echo_input,
        inputs=[simple_cu_doc_demo_file_dropdown],
        outputs=[simple_cu_doc_file_upload],
    )
    simple_cu_doc_file_upload.change(
        fn=render_visual_media_input,
        inputs=[simple_cu_doc_file_upload],
        outputs=[simple_cu_doc_input_thumbs],
    )
    simple_cu_doc_process_btn.click(
        fn=prebuilt_cu_document_process,
        inputs=[
            simple_cu_doc_analyzer_schema_dropdown,
            simple_cu_doc_file_upload,
        ],
        outputs=[
            simple_cu_doc_status_code,
            simple_cu_doc_time_taken,
            simple_cu_doc_img_output,
            simple_cu_doc_fields_output_md,
            simple_cu_doc_output_json,
        ],
    )

    def prebuilt_cu_video_process(
        analyzer_id: str,
        file: str,
    ):
        analyzer_route = "content_understanding_video"
        processing_start_time = None
        markdown_output = ""
        try:
            if file is None:
                gr.Warning("Please select or upload a file, then click 'Process File'.")
                raise ValueError(
                    "Please select or upload a file, then click 'Process File'."
                )
            # Get response from the API
            with open(file, "rb") as f:
                payload = {"analyzer_id": analyzer_id}
                file_mime_type = mimetypes.guess_type(file)[0]
                files = {
                    "file": (file, f, file_mime_type),
                    "json": ("payload.json", json.dumps(payload), "application/json"),
                }

                status_code, processing_time_taken, response = send_request(
                    route=analyzer_route,
                    files=files,
                    force_json_content_type=True,
                )
            # Format image and field outputs (if available in the response)
            ### 3. Extract key frames from the video input
            if (
                isinstance(response, dict)
                and response.get("cu_raw_response", {}).get("status", "") == "Succeeded"
            ):
                cu_result = response["cu_raw_response"]
                # First, get the processed markdown output containing the extracted fields
                # for each segment. If the field is missing, create an empty string for each segment.
                segment_formatted_field_mds = response.get(
                    "segment_formatted_field_mds",
                    [""] * len(cu_result["result"]["contents"]),
                )

                # Next, extract the first frame from each segment
                all_segment_start_frame_times_ms = [
                    scene_result["startTimeMs"]
                    for scene_result in cu_result["result"]["contents"]
                ]

                with open(file, "rb") as f:
                    file_bytes = f.read()
                start_frame_mapper = extract_frames_from_video_bytes(
                    file_bytes, all_segment_start_frame_times_ms, img_type="pil"
                )
                # Reduce the size of the frames
                start_frame_mapper = {
                    frame_time_ms: resize_img_by_max(img, 480)
                    for frame_time_ms, img in start_frame_mapper.items()
                }

                processed_md_strs = list()

                for segment_idx, segment_contents in enumerate(
                    cu_result["result"]["contents"]
                ):
                    # Get key frames and scene summary
                    start_time_str = milliseconds_to_simple_string(
                        segment_contents["startTimeMs"]
                    )
                    end_time_str = milliseconds_to_simple_string(
                        segment_contents["endTimeMs"]
                    )
                    img_bytes = pil_img_to_base64_bytes(
                        start_frame_mapper[segment_contents["startTimeMs"]]
                    )
                    base64_url = base64_img_str_to_url(
                        img_bytes.decode(), mime_type="image/jpeg"
                    )
                    scene_summary_md = (
                        f"<h3>Scene {segment_idx + 1} [{start_time_str} -> {end_time_str}]</h3>\n\n"
                        f"*Starting Frame*\n"
                        f"![]({base64_url})\n"
                        "<h4>Extracted Fields:</h4>\n\n"
                        f"{segment_formatted_field_mds[segment_idx]}"
                    )
                    processed_md_strs.append(scene_summary_md)
                markdown_output = "\n\n".join(processed_md_strs)
        except Exception as e:
            logging.exception("Error processing file")
            status_code = 500
            processing_time_taken = (
                f"{round(time.time() - processing_start_time, 1)} seconds"
                if processing_start_time is not None
                else "N/A"
            )
            response = {"Error": str(e)}
        return (
            status_code,
            processing_time_taken,
            gr.Markdown(
                markdown_output,
                label="Processed Video Content",
                show_label=True,
                visible=True,
            ),
            gr.JSON(value=response, label="Full Function API Response", visible=True),
        )

    # Input components
    simple_cu_video_header = gr.Label(
        "Azure Content Understanding - Video Processing", show_label=False
    )
    simple_cu_video_instructions = gr.Markdown(
        (
            "This example classifies and generates information about each scene in a video Azure Content Understanding using one of the "
            "[pre-built schemas](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/config/content_understanding_schemas.json) "
            "included in the accelerator.\nThese schemas can be modified to extract the specific fields required for your use case "
            "([Code Link](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/content_understanding_video.py)). "
            "For more information: [Azure Content Understanding - Video Overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/video/overview).\n"
            "Once a video is processed, the accelerator performs some enrichment and post-processing to simplify the markdown output.\n"
            "To get started, select one of the existing schemas from the dropdown, then select a demo video or upload your own to have it processed."
        ),
        show_label=False,
        line_breaks=True,
    )
    with gr.Row():
        with gr.Column():
            simple_cu_video_analyzer_schema_dropdown = gr.Dropdown(
                choices=BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY["video"],
                value=BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY["video"][0][1],
                label="Select field extraction schema",
                interactive=True,
            )
            cu_video_file_dropdown_options = [
                (
                    "",
                    None,
                )
            ] + DEMO_CONTENT_UNDERSTANDING_VIDEO_TUPLES
            simple_cu_video_example_dropdown = gr.Dropdown(
                choices=cu_video_file_dropdown_options,
                value="",
                label="[Optional] Select demo video file",
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Column():
            simple_cu_video_file_upload = gr.Video(
                value=None,
                label="Video File Preview. To upload a different file, Hit the 'X' button to the top right of this element ->",
                sources=["upload"],
                autoplay=True,
                loop=True,
            )
    simple_cu_video_process_btn = gr.Button("Process File", variant="primary")
    # Output components
    with gr.Column() as simple_cu_video_output_row:
        simple_cu_video_output_label = gr.Label(value="Result", show_label=False)
        with gr.Row():
            simple_cu_video_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            simple_cu_video_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        with gr.Row():
            simple_cu_video_fields_output_md = gr.Markdown(
                "<h3>Extracted Field Values</h3>",
                show_label=False,
                visible=False,
                line_breaks=True,
            )
        simple_cu_video_output_json = gr.JSON(
            label="Full Function API Response", visible=False
        )

    # Actions
    simple_cu_video_example_dropdown.change(
        fn=echo_input,
        inputs=[simple_cu_video_example_dropdown],
        outputs=[simple_cu_video_file_upload],
    )
    simple_cu_video_process_btn.click(
        fn=prebuilt_cu_video_process,
        inputs=[
            simple_cu_video_analyzer_schema_dropdown,
            simple_cu_video_file_upload,
        ],
        outputs=[
            simple_cu_video_status_code,
            simple_cu_video_time_taken,
            simple_cu_video_fields_output_md,
            simple_cu_video_output_json,
        ],
    )

    def prebuilt_cu_audio_process(
        analyzer_id: str,
        file: str,
    ):
        analyzer_route = "content_understanding_audio"
        processing_start_time = None
        audio_transcription = None
        fields_output = ""
        try:
            if file is None:
                gr.Warning("Please select or upload a file, then click 'Process File'.")
                raise ValueError(
                    "Please select or upload a file, then click 'Process File'."
                )
            # Get response from the API
            with open(file, "rb") as f:
                payload = {"analyzer_id": analyzer_id}
                file_mime_type = mimetypes.guess_type(file)[0]
                files = {
                    "file": (file, f, file_mime_type),
                    "json": ("payload.json", json.dumps(payload), "application/json"),
                }

                status_code, processing_time_taken, response = send_request(
                    route=analyzer_route,
                    files=files,
                    force_json_content_type=True,
                )
            # Format transcription and field outputs (if available in the response)
            if isinstance(response, dict) and response.get("formatted_fields_md"):
                fields_output = f"<h3>Extracted Field Values</h3>\n\n{response.get('formatted_fields_md')}"
            if isinstance(response, dict) and response.get("condensed_transcription"):
                audio_transcription = response.get("condensed_transcription")
        except Exception as e:
            logging.exception("Error processing file")
            status_code = 500
            processing_time_taken = (
                f"{round(time.time() - processing_start_time, 1)} seconds"
                if processing_start_time is not None
                else "N/A"
            )
            response = {"Error": str(e)}
        return (
            status_code,
            processing_time_taken,
            gr.Markdown(
                "<h3>Processed Call Transcription</h3>",
                show_label=False,
                visible=True,
                line_breaks=True,
            ),
            gr.Textbox(
                value=audio_transcription,
                label="Transcription",
                interactive=False,
                visible=True,
            ),
            gr.Markdown(
                fields_output, show_label=False, line_breaks=True, visible=True
            ),
            gr.JSON(value=response, label="Full Function API Response", visible=True),
        )

    # Input components
    simple_cu_audio_header = gr.Label(
        "Azure Content Understanding - Audio Processing", show_label=False
    )
    simple_cu_audio_instructions = gr.Markdown(
        (
            "This example extracts information from audio files using Azure Content Understanding using one of the "
            "[pre-built schemas](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/config/content_understanding_schemas.json) "
            "included in the accelerator.\nThese schemas can be modified to extract the specific fields required for your use case "
            "([Code Link](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/content_understanding_audio.py)). "
            "For more information: [Azure Content Understanding - Audio Overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/audio/overview).\n"
            "Once an audio file is processed, the accelerator post-processes the raw output to simplify the transcription.\n"
            "To get started, select one of the existing schemas from the dropdown, then select a demo recording or upload your own to have it processed."
        ),
        show_label=False,
        line_breaks=True,
    )
    with gr.Column():
        simple_cu_audio_analyzer_schema_dropdown = gr.Dropdown(
            choices=BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY["audio"],
            value=BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY["audio"][0][1],
            label="Select field extraction schema",
            interactive=True,
        )
        cu_audio_file_dropdown_options = [
            (
                "",
                None,
            )
        ] + DEMO_CONTENT_UNDERSTANDING_AUDIO_TUPLES
        simple_cu_audio_example_dropdown = gr.Dropdown(
            choices=cu_audio_file_dropdown_options,
            value="",
            label="[Optional] Select demo audio file",
            interactive=True,
        )
        simple_cu_audio_file_upload = gr.Audio(
            value=DEMO_CALL_CENTER_AUDIO_FILES[0],
            label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
            sources=["upload", "microphone"],
            type="filepath",
        )
    simple_cu_audio_process_btn = gr.Button("Process File", variant="primary")
    # Output components
    with gr.Column() as simple_cu_audio_output_row:
        simple_cu_audio_output_label = gr.Label(value="Result", show_label=False)
        with gr.Row():
            simple_cu_audio_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            simple_cu_audio_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        with gr.Row():
            with gr.Column():
                simple_cu_audio_transcription_title = gr.Markdown(
                    "<h3>Processed Call Transcription</h3>",
                    show_label=False,
                    visible=False,
                    line_breaks=True,
                )
                simple_cu_audio_transcription_output = gr.Textbox(
                    value="",
                    show_label=False,
                    visible=False,
                )
            with gr.Column():
                simple_cu_audio_fields_output_md = gr.Markdown(
                    "<h3>Extracted Field Values</h3>",
                    show_label=False,
                    visible=False,
                    line_breaks=True,
                )
        simple_cu_audio_output_json = gr.JSON(
            label="Full Function API Response", visible=False
        )

    # Actions
    simple_cu_audio_example_dropdown.change(
        fn=echo_input,
        inputs=[simple_cu_audio_example_dropdown],
        outputs=[simple_cu_audio_file_upload],
    )
    simple_cu_audio_process_btn.click(
        fn=prebuilt_cu_audio_process,
        inputs=[
            simple_cu_audio_analyzer_schema_dropdown,
            simple_cu_audio_file_upload,
        ],
        outputs=[
            simple_cu_audio_status_code,
            simple_cu_audio_time_taken,
            simple_cu_audio_transcription_title,
            simple_cu_audio_transcription_output,
            simple_cu_audio_fields_output_md,
            simple_cu_audio_output_json,
        ],
    )

    def prebuilt_cu_image_process(
        analyzer_id: str,
        file: str,
    ):
        analyzer_route = "content_understanding_image"
        processing_start_time = None
        fields_output = ""

        try:
            if file is None:
                gr.Warning("Please select or upload a file, then click 'Process File'.")
                raise ValueError(
                    "Please select or upload a file, then click 'Process File'."
                )
            # Get response from the API
            with open(file, "rb") as f:
                payload = {"analyzer_id": analyzer_id}
                file_mime_type = mimetypes.guess_type(file)[0]
                files = {
                    "file": (file, f, file_mime_type),
                    "json": ("payload.json", json.dumps(payload), "application/json"),
                }

                status_code, processing_time_taken, response = send_request(
                    route=analyzer_route,
                    files=files,
                    force_json_content_type=True,
                )
                if isinstance(response, dict) and response.get("formatted_fields_md"):
                    fields_output = response["formatted_fields_md"]
        except Exception as e:
            logging.exception("Error processing file")
            status_code = 500
            processing_time_taken = (
                f"{round(time.time() - processing_start_time, 1)} seconds"
                if processing_start_time is not None
                else "N/A"
            )
            response = {"Error": str(e)}
        return (
            status_code,
            processing_time_taken,
            gr.Markdown(
                f"<h3>Extracted Field Values</h3>\n\n{fields_output}",
                show_label=False,
                visible=True,
                line_breaks=True,
            ),
            gr.JSON(value=response, label="Full Function API Response", visible=True),
        )

    # Input components
    simple_cu_image_header = gr.Label(
        "Azure Content Understanding - Image Processing", show_label=False
    )
    simple_cu_image_instructions = gr.Markdown(
        (
            "This example extracts information from images using Azure Content Understanding using one of the "
            "[pre-built schemas](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/config/content_understanding_schemas.json) "
            "included in the accelerator.\nThese schemas can be modified to extract the specific fields required for your use case "
            "([Code Link](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/content_understanding_image.py)). "
            "For more information: [Azure Content Understanding - Images Overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/image/overview).\n"
            "To get started, select one of the existing schemas from the dropdown, then select a demo document or upload your own to have it processed."
        ),
        show_label=False,
        line_breaks=True,
    )
    with gr.Row():
        with gr.Column():
            simple_cu_image_analyzer_schema_dropdown = gr.Dropdown(
                choices=BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY["image"],
                value=BACKEND_CU_SCHEMA_TUPLES_BY_MODALITY["image"][0][1],
                label="Select field extraction schema",
                interactive=True,
            )
            cu_image_file_dropdown_options = [
                (
                    "",
                    None,
                )
            ] + DEMO_CONTENT_UNDERSTANDING_IMAGE_TUPLES
            simple_cu_image_example_dropdown = gr.Dropdown(
                choices=cu_image_file_dropdown_options,
                value="",
                label="[Optional] Select demo image file",
                interactive=True,
                allow_custom_value=True,
            )
            simple_cu_image_file_upload = gr.File(
                height=100,
                label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
                file_count="single",
                type="filepath",
            )
        with gr.Column():
            simple_cu_image_input_thumbs = gr.Gallery(
                label="Input File Preview", object_fit="contain", visible=True
            )
    simple_cu_image_process_btn = gr.Button("Process File", variant="primary")
    # Output components
    with gr.Column() as simple_cu_image_output_row:
        simple_cu_image_output_label = gr.Label(value="Result", show_label=False)
        with gr.Row():
            simple_cu_image_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            simple_cu_image_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        with gr.Row():
            simple_cu_image_fields_output_md = gr.Markdown(
                "<h3>Extracted Field Values</h3>",
                show_label=False,
                visible=False,
                line_breaks=True,
            )
        simple_cu_image_output_json = gr.JSON(
            label="Full Function API Response", visible=False
        )

    # Actions
    simple_cu_image_example_dropdown.change(
        fn=echo_input,
        inputs=[simple_cu_image_example_dropdown],
        outputs=[simple_cu_image_file_upload],
    )
    simple_cu_image_file_upload.change(
        fn=render_visual_media_input,
        inputs=[simple_cu_image_file_upload],
        outputs=[simple_cu_image_input_thumbs],
    )
    simple_cu_image_process_btn.click(
        fn=prebuilt_cu_image_process,
        inputs=[
            simple_cu_image_analyzer_schema_dropdown,
            simple_cu_image_file_upload,
        ],
        outputs=[
            simple_cu_image_status_code,
            simple_cu_image_time_taken,
            simple_cu_image_fields_output_md,
            simple_cu_image_output_json,
        ],
    )


### Form Extraction - Blob to CosmosDB example ###
if IS_COSMOSDB_AVAILABLE:
    with gr.Blocks(analytics_enabled=False) as blob_form_extraction_to_cosmosdb_block:
        # Connect to CosmosDB using identity auth (this requires a role assignment for
        # the current identity)
        cosmos_client = CosmosClient(
            url=COSMOSDB_ACCOUNT_ENDPOINT, credential=credential
        )
        blob_form_extraction_to_cosmosdb_container_client = (
            cosmos_client.get_database_client(
                COSMOSDB_DATABASE_NAME
            ).get_container_client("blob-form-to-cosmosdb-container")
        )

        # Define the processing function, which reshapes the input into the correct schema
        def blob_form_to_cosmosdb_process(file: str):
            start_time = time.time()
            try:
                # Append a random prefix to the filename to avoid conflicts
                random_prefix = str(int(time.time()))
                name, ext = os.path.splitext(os.path.basename(file))
                blob_name = f"{name}_{random_prefix}{ext}"
                # Upload data to blob storage
                blob_client = blob_service_client.get_blob_client(
                    container="blob-form-to-cosmosdb-blobs", blob=blob_name
                )
                with open(file, "rb") as f:
                    data = f.read()
                    blob_client.upload_blob(data)
                # Query the CosmosDB container until a result is found
                gr.Info(
                    "File uploaded to Blob Storage. Polling CosmosDB until the result appears..."
                )
                cosmos_timeout_secs = 30
                query = "SELECT * FROM c WHERE c.filename = @filename"
                parameters = [{"name": "@filename", "value": blob_name}]
                items = None
                cosmos_query_timer = time.time()
                while not items:
                    time.sleep(2)
                    if time.time() - cosmos_query_timer > cosmos_timeout_secs:
                        # If no record is found, return a placeholder result
                        timeout_error_text = f"Timeout: No result found in CosmosDB after {cosmos_timeout_secs} seconds."
                        client_side_time_taken = (
                            f"{round(time.time() - start_time, 1)} seconds"
                        )
                        return (
                            404,
                            client_side_time_taken,
                            {"Error": timeout_error_text},
                        )

                    items = list(
                        blob_form_extraction_to_cosmosdb_container_client.query_items(
                            query=query,
                            parameters=parameters,
                            enable_cross_partition_query=True,
                        )
                    )
                client_side_time_taken = f"{round(time.time() - start_time, 1)} seconds"
                # Delete the blob from storage
                blob_client.delete_blob()
                gr.Info("Task complete. Deleting the file from blob storage...")
                # Return status code, time taken and the first item from the list (it should only have one item)
                return (200, client_side_time_taken, items[0])
            except Exception as e:
                logging.exception(
                    "Error occurred during blob upload and CosmosDB query."
                )
                # Ensure the blob is deleted from storage if an error occurs
                try:
                    blob_client.delete_blob()
                    gr.Error("Error occurred. Deleting the file from blob storage...")
                except Exception as _e:
                    pass
                # Return the result including the error.
                client_side_time_taken = f"{round(time.time() - start_time, 1)} seconds"
                return (500, client_side_time_taken, {"Error": str(e)})

        # Input components
        blob_form_to_cosmosdb_instructions = gr.Markdown(
            (
                "This example extracts key information from Bank Account application forms, triggered by uploading the document to blob storage and with the results written to a CosmosDB NoSQL database - Currently WIP "
                 
                "Note: Any files uploaded as part of this pipeline are deleted immediately after processing.\n"
            ),
            show_label=False,
            line_breaks=True,
        )
        with gr.Row():
            blob_form_to_cosmosdb_file_upload = gr.File(
                label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
                file_count="single",
                type="filepath",
            )
            blob_form_to_cosmosdb_input_thumbs = gr.Gallery(
                label="File Preview", object_fit="contain", visible=True
            )
        # Examples
        blob_form_to_cosmosdb_examples = gr.Examples(
            examples=DEMO_ACCOUNT_OPENING_FORM_FILES,
            inputs=[blob_form_to_cosmosdb_file_upload],
            label=FILE_EXAMPLES_LABEL,
            outputs=[
                blob_form_to_cosmosdb_input_thumbs,
            ],
            fn=render_visual_media_input,
            run_on_click=True,
        )
        blob_form_to_cosmosdb_process_btn = gr.Button("Process File", variant="primary")
        # Output components
        with gr.Column() as blob_form_to_cosmosdb_output_row:
            blob_form_to_cosmosdb_output_label = gr.Label(
                value="Pipeline Result", show_label=False
            )
            with gr.Row():
                blob_form_to_cosmosdb_status_code = gr.Textbox(
                    label="Response Status Code", interactive=False
                )
                blob_form_to_cosmosdb_time_taken = gr.Textbox(
                    label="Time Taken", interactive=False
                )
            blob_form_to_cosmosdb_output_json = gr.JSON(label="CosmosDB Record")
        # Actions
        blob_form_to_cosmosdb_process_btn.click(
            fn=blob_form_to_cosmosdb_process,
            inputs=[blob_form_to_cosmosdb_file_upload],
            outputs=[
                blob_form_to_cosmosdb_status_code,
                blob_form_to_cosmosdb_time_taken,
                blob_form_to_cosmosdb_output_json,
            ],
        )

### Audio Processing ###
with gr.Blocks(analytics_enabled=False) as call_center_audio_processing_block:

    def format_audio_analysis_markdown(response_obj):
        """
        Formats the audio analysis response object into markdown for display.
        Clearly marks transcription, summary, and key information.
        """
        if not isinstance(response_obj, dict):
            return "No analysis available."

        md_lines = []

        summary = (
            response_obj.get("result", {}).get("summary")
            or response_obj.get("result", {}).get("audio_summary")
            or response_obj.get("summary")
            or response_obj.get("audio_summary")
        )
        if summary:
            md_lines.append("## 📝 Summary")
            md_lines.append(f"{summary}\n")

        actions = (
            response_obj.get("result", {}).get("actions")
            or response_obj.get("actions")
            or []
        )
        next_action = (
            response_obj.get("result", {}).get("next_action")
            or response_obj.get("next_action")
        )
        next_action_ts = (
            response_obj.get("result", {}).get("next_action_sentence_timestamp")
            or response_obj.get("next_action_sentence_timestamp")
        )
        if actions or next_action:
            md_lines.append("## 🔑 Key Information: Actions / Next Steps")
            if actions:
                for action in actions:
                    if isinstance(action, dict):
                        desc = action.get("description") or action.get("action") or str(action)
                        ts = action.get("timestamp") or ""
                        md_lines.append(f"- **{desc}**{' at ' + ts if ts else ''}")
                    else:
                        md_lines.append(f"- {action}")
            if next_action:
                md_lines.append(f"- **Next Action:** {next_action}{' at ' + next_action_ts if next_action_ts else ''}")
            md_lines.append("")

        key_information = (
            response_obj.get("result", {}).get("key_information")
            or response_obj.get("key_information")
            or []
        )
        if key_information:
            md_lines.append("## 🏷️ Key Information: Entities, Events, People, Locations")
            for item in key_information:
                if isinstance(item, dict):
                    type_ = item.get("type", "")
                    value = item.get("value", "")
                    role = item.get("role", "")
                    ts = item.get("timestamp", "")
                    context = item.get("context", "")
                    line = f"- **{type_.title()}**: {value}"
                    if role:
                        line += f" ({role})"
                    if ts:
                        line += f" at {ts}"
                    if context:
                        line += f" — {context}"
                    md_lines.append(line)
                else:
                    md_lines.append(f"- {item}")
            md_lines.append("")

        keywords = (
            response_obj.get("result", {}).get("keywords")
            or response_obj.get("keywords")
            or []
        )
        if keywords and not key_information:
            md_lines.append("## 🏷️ Key Information: Keywords")
            for kw in keywords:
                if isinstance(kw, dict):
                    word = kw.get("keyword") or str(kw)
                    sent = kw.get("full_sentence_text") or ""
                    ts = kw.get("timestamp") or ""
                    md_lines.append(f"- **{word}**{' at ' + ts if ts else ''}: {sent}")
                else:
                    md_lines.append(f"- {kw}")
            md_lines.append("")

        transcription = (
            response_obj.get("result", {}).get("transcription")
            or response_obj.get("speech_extracted_text")
            or response_obj.get("transcription")
        )
        if transcription:
            md_lines.append("## 🎤 Transcription")
            md_lines.append("```text")
            md_lines.append(transcription)
            md_lines.append("```")

        return "\n".join(md_lines) if md_lines else "No analysis available."

    def video_to_mp3(video_path):
        audio_path = video_path.rsplit('.', 1)[0] + ".mp3"
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(audio_path, logger=None)
        return audio_path

    def cc_video_audio_upload(video_file, audio_file, transcription_method):
        # Prefer audio if provided, otherwise video
        file = audio_file or video_file
        if file is None:
            gr.Warning("Please upload an audio or video file.")
            return ("", "", "", {})
        # If video file, convert to mp3
        mime_type, _ = mimetypes.guess_type(file)
        if mime_type and mime_type.startswith("video"):
            file = video_to_mp3(file)
            mime_type = "audio/mp3"
        elif mime_type and mime_type.startswith("audio"):
            pass
        else:
            gr.Warning("Unsupported file type. Please upload a valid audio or video file.")
            return ("", "", "", {})
        # Now process the audio as before
        with open(file, "rb") as f:
            payload = {"method": transcription_method}
            files = {
                "audio": (os.path.basename(file), f, mime_type),
                "json": ("payload.json", json.dumps(payload), "application/json"),
            }
            status_code, time_taken, response_json = send_request(
                route="call_center_audio_analysis",
                files=files,
                force_json_content_type=True,
            )
        try:
            response_obj = json.loads(response_json) if isinstance(response_json, str) else response_json
        except Exception:
            response_obj = response_json
        md = format_audio_analysis_markdown(response_obj)
        return status_code, time_taken, md, response_obj

    cc_audio_proc_instructions = gr.Markdown(
        (
            "Upload an audio or video file. Video files will be automatically converted to audio and analyzed for transcription and summarisation."
        ),
        show_label=False,
    )
    with gr.Row():
        video_upload = gr.Video(
            label="Upload Video File (mp4, mov, avi, etc)",
            sources=["upload"],
        )
        audio_upload = gr.File(
            label="Upload Audio File (wav, mp3, etc)",
            file_types=["audio/*"],
        )
    cc_audio_proc_transcription_method = gr.Dropdown(
        label="Select Transcription Method",
        value="fast",
        choices=[
            ("Fast Transcription API", "fast"),
            ("Azure OpenAI Whisper", "aoai_whisper"),
        ],
    )
    cc_audio_proc_start_btn = gr.Button("Process File", variant="primary")
    with gr.Column() as cc_audio_proc_output_row:
        cc_audio_proc_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            cc_audio_proc_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            cc_audio_proc_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        cc_audio_proc_output_md = gr.Markdown(label="Analysis (Markdown)")
        cc_audio_proc_output_json = gr.JSON(label="API Response")
    cc_audio_proc_start_btn.click(
        fn=cc_video_audio_upload,
        inputs=[video_upload, audio_upload, cc_audio_proc_transcription_method],
        outputs=[
            cc_audio_proc_status_code,
            cc_audio_proc_time_taken,
            cc_audio_proc_output_md,
            cc_audio_proc_output_json,
        ],
    )



### Text summarisation Example ###

with gr.Blocks(analytics_enabled=False) as sum_text_block:
    # Define requesting function, which reshapes the input into the correct schema
    def sum_text_send_request(input_text: str, summary_style: str, num_sentences: int):
        return send_request(
            route="summarize_text",
            json_data={
                "text": input_text,
                "summary_style": summary_style,
                "num_sentences": num_sentences,
            },
            force_json_content_type=False,
        )

    # Input components
    sum_text_instructions = gr.Markdown(
        (
            "This simple demo uses a single LLM request to summarise a body of text "
            "into a desired style and number of sentences "
            "([Code Link](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_summarize_text.py))."
        ),
        show_label=False,
    )
    with gr.Row():
        sum_text_style_dropdown = gr.Dropdown(
            label="Select or enter the style of the generated summary",
            value="Police Report",
            choices=["Police Report", "Incident Log", "Breifing Note", "Plain English"],
            allow_custom_value=True,
        )
        sum_text_num_sentences = gr.Number(
            label="Enter the desired number of sentences in the summary",
            value=3,
        )
    # Examples
    sum_text_input_text = gr.Textbox(
        value="", lines=3, label="Enter the text to be summarised"
    )
    sum_text_example_dropdown_options = [
        (
            "",
            None,
        )
    ] + TEXT_SAMPLES_TUPLES
    sum_text_example_dropdown = gr.Dropdown(
        value="",
        choices=sum_text_example_dropdown_options,
        label=TEXT_EXAMPLES_LABEL,
        allow_custom_value=True,
    )
    sum_text_example_dropdown.change(
        fn=echo_input,
        inputs=[sum_text_example_dropdown],
        outputs=[sum_text_input_text],
    )
    # Output components
    with gr.Column(render=False) as sum_text_output_row:
        sum_text_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            sum_text_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            sum_text_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        sum_text_output_text = gr.Textbox(label="API Response", interactive=False)
    sum_text_get_response_btn = gr.Button("Process Text", variant="primary")
    sum_text_output_row.render()
    # Actions
    sum_text_get_response_btn.click(
        fn=sum_text_send_request,
        inputs=[sum_text_input_text, sum_text_style_dropdown, sum_text_num_sentences],
        outputs=[
            sum_text_status_code,
            sum_text_time_taken,
            sum_text_output_text,
        ],
    )

with gr.Blocks(analytics_enabled=False) as pii_redaction_block:
    # Define requesting functions, which reshapes the input into the correct schema
    def pii_redaction_text_send_request(input_text: Optional[str]):
        processing_start_time = time.time()
        redacted_text_output = ""
        try:
            status_code, processing_time_taken, response = send_request(
                route="pii_redaction_text",
                json_data={
                    "text": input_text,
                },
                force_json_content_type=True,
            )
            if isinstance(response, dict) and response.get("redacted_text"):
                redacted_text_output = response["redacted_text"]
        except Exception as e:
            logging.exception("Error processing file")
            status_code = 500
            processing_time_taken = (
                f"{round(time.time() - processing_start_time, 1)} seconds"
                if processing_start_time is not None
                else "N/A"
            )
            redacted_text_output = {"Error": str(e)}
        return (
            status_code,
            processing_time_taken,
            gr.JSON(value=response, label="Full Function API Response", visible=True),
            gr.Textbox(
                label="Redacted Text",
                value=redacted_text_output,
                interactive=False,
                visible=True,
            ),
            gr.Gallery(
                label="Input PDF Pages",
                interactive=False,
                visible=False,
            ),
            gr.Gallery(
                label="Redacted PDF Pages",
                interactive=False,
                visible=False,
            ),
        )

    def pii_redaction_pdf_send_request(file: Optional[str]):
        processing_start_time = time.time()
        redacted_text_output = ""
        redacted_pdf_page_imgs = None
        input_doc_pdf_page_imgs = None

        try:
            if file is None:
                gr.Warning(
                    "Please select or upload a PDF file, then click 'Process PDF File'."
                )
                return ("", "", {})
            # Get response from the API
            mime_type = mimetypes.guess_type(file)[0]
            with open(file, "rb") as f:
                data = f.read()
                headers = {"Content-Type": mime_type}
                status_code, processing_time_taken, response = send_request(
                    route="pii_redaction_pdf",
                    data=data,
                    headers=headers,
                    force_json_content_type=True,
                )
                if isinstance(response, dict) and response.get("redacted_text"):
                    redacted_text_output = response["redacted_text"]
                if isinstance(response, dict) and response.get(
                    "input_doc_pdf_page_imgs"
                ):
                    input_doc_pdf_page_imgs = [
                        Image.open(BytesIO(base64.b64decode(b64_img)))
                        for b64_img in response.pop("input_doc_pdf_page_imgs")
                    ]
                if isinstance(response, dict) and response.get(
                    "redacted_pdf_page_imgs"
                ):
                    redacted_pdf_page_imgs = [
                        Image.open(BytesIO(base64.b64decode(b64_img)))
                        for b64_img in response.pop("redacted_pdf_page_imgs")
                    ]
        except Exception as e:
            logging.exception("Error processing file")
            status_code = 500
            processing_time_taken = (
                f"{round(time.time() - processing_start_time, 1)} seconds"
                if processing_start_time is not None
                else "N/A"
            )
            response = {"Error": str(e)}
        return (
            status_code,
            processing_time_taken,
            gr.JSON(value=response, label="Full Function API Response", visible=True),
            gr.Textbox(
                label="Redacted Text",
                value=redacted_text_output,
                interactive=False,
                visible=True,
            ),
            gr.Gallery(
                label="Input PDF Pages",
                value=input_doc_pdf_page_imgs,
                type="pil",
                object_fit="contain",
                columns=min(len(input_doc_pdf_page_imgs), 2),
                show_label=True,
                interactive=False,
                visible=True,
            ),
            gr.Gallery(
                label="Redacted PDF Pages",
                value=redacted_pdf_page_imgs,
                type="pil",
                object_fit="contain",
                columns=min(len(redacted_pdf_page_imgs), 2),
                show_label=True,
                interactive=False,
                visible=True,
            ),
        )

    # Input components
    pii_redaction_instructions = gr.Markdown(
        (
            "This demo uses the Azure Language Service's PII Redaction API to remove PII from text and documents."
            "([Code Link](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_pii_redaction.py))."
            "\n\nThe pipeline is as follows:\n"
            "1. If the input is a PDF, Azure Document Intelligence or PyMuPDF are used to extract the raw text from the PDF.\n"
            "2. The raw text is sent to the Azure Language Service to detect all PII entities in the text.\n"
            "3. The PII entities are redacted in-place, returning the processed text (and document) with all PII entities redacted.\n\n"
            "The response includes the final result, as well as many of the intermediate outputs from the processing pipeline."
        ),
        show_label=False,
    )
    # Examples
    with gr.Row():
        pii_redaction_dummy_none_obj = gr.Number(value=lambda: None, visible=False)
        with gr.Column():
            pii_redaction_text_example_dropdown_options = [
                (
                    "",
                    None,
                )
            ] + PII_REDACTION_TEXT_SAMPLE_TUPLES
            pii_redaction_text_example_dropdown = gr.Dropdown(
                label="[Optional] Select an example text to be processed",
                value="",
                choices=pii_redaction_text_example_dropdown_options,
                allow_custom_value=True,
            )
            pii_redaction_input_text = gr.Textbox(
                value="", lines=3, label="Enter the text to be processed"
            )
            pii_redaction_process_text_btn = gr.Button(
                "Process Text", variant="primary"
            )
        with gr.Column():
            pii_redaction_pdf_example_dropdown_options = [
                (
                    "",
                    None,
                )
            ] + DEMO_PII_REDACTION_PDF_TUPLES
            pii_redaction_pdf_example_dropdown = gr.Dropdown(
                label="[Optional] Select an example PDF to be processed",
                value="",
                choices=pii_redaction_pdf_example_dropdown_options,
                allow_custom_value=True,
            )
            pii_redaction_input_pdf = gr.File(
                label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
                file_count="single",
                type="filepath",
                file_types=[".pdf"],
            )
            pii_redaction_process_pdf_btn = gr.Button("Process PDF", variant="primary")

    # Output components
    with gr.Column() as pii_redaction_output_row:
        pii_redaction_output_label = gr.Label(value="Result", show_label=False)
        with gr.Row():
            pii_redaction_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            pii_redaction_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        pii_redaction_output_text = gr.Textbox(
            label="Redacted Text", interactive=False, visible=False
        )
        with gr.Row():
            pii_redaction_input_pdf_pages = gr.Gallery(
                label="Input PDF Pages",
                interactive=False,
                visible=False,
            )
            pii_redaction_output_pdf_pages = gr.Gallery(
                label="Redacted PDF Pages",
                interactive=False,
                visible=False,
            )
        pii_redaction_output_json = gr.JSON(label="API Response", visible=False)

    # Actions
    pii_redaction_text_example_dropdown.change(
        fn=echo_input,
        inputs=[pii_redaction_text_example_dropdown],
        outputs=[pii_redaction_input_text],
    )
    pii_redaction_pdf_example_dropdown.change(
        fn=echo_input,
        inputs=[pii_redaction_pdf_example_dropdown],
        outputs=[pii_redaction_input_pdf],
    )
    pii_redaction_process_text_btn.click(
        fn=pii_redaction_text_send_request,
        inputs=[pii_redaction_input_text],
        outputs=[
            pii_redaction_status_code,
            pii_redaction_time_taken,
            pii_redaction_output_json,
            pii_redaction_output_text,
            pii_redaction_input_pdf_pages,  # Return hidden PDF page element
            pii_redaction_output_pdf_pages,  # Return hidden PDF page element
        ],
    )
    pii_redaction_process_pdf_btn.click(
        fn=pii_redaction_pdf_send_request,
        inputs=[pii_redaction_input_pdf],
        outputs=[
            pii_redaction_status_code,
            pii_redaction_time_taken,
            pii_redaction_output_json,
            pii_redaction_output_text,
            pii_redaction_input_pdf_pages,
            pii_redaction_output_pdf_pages,
        ],
    )
### Doc Intelligence Extraction + LLM Name Extraction Example ###
with gr.Blocks(analytics_enabled=False) as di_llm_ext_names_block:
    # Format API response into markdown
    def format_investigation_fields(response_json: dict) -> str:
        if not isinstance(response_json, dict):
            return "Invalid response format."

        fields = response_json.get("result", {})
        if not fields:
            return "No result found in response."

        md_lines = []

        for key, value in fields.items():
            if key == "structured_people" and value:
                md_lines.append("### Structured People")
                for person in value:
                    md_lines.append(f"- **Name:** {person.get('name', '')}")
                    if person.get("role"):
                        md_lines.append(f"    - Role: {person['role']}")
                    if person.get("affiliations"):
                        md_lines.append(f"    - Affiliations: {', '.join(person['affiliations'])}")
                    if person.get("vehicles"):
                        for v in person["vehicles"]:
                            md_lines.append(f"    - Vehicle: {v.get('description', '')} ({v.get('reg_number', '')})")
                    if person.get("locations"):
                        md_lines.append(f"    - Locations: {', '.join(person['locations'])}")
                    if person.get("contact_info"):
                        md_lines.append(f"    - Contact Info: {', '.join(person['contact_info'])}")
                md_lines.append("")
            elif key == "structured_events" and value:
                md_lines.append("### Structured Events")
                for event in value:
                    md_lines.append(f"- **Type:** {event.get('type', '')}")
                    if event.get("datetime"):
                        md_lines.append(f"    - Datetime: {event['datetime']}")
                    if event.get("location"):
                        md_lines.append(f"    - Location: {event['location']}")
                    if event.get("suspects"):
                        md_lines.append(f"    - Suspects: {', '.join(event['suspects'])}")
                    if event.get("officers"):
                        md_lines.append(f"    - Officers: {', '.join(event['officers'])}")

                   

                    if event.get("vehicles"):
                        for v in event["vehicles"]:
                            md_lines.append(f"    - Vehicle: {v.get('description', '')} ({v.get('reg_number', '')})")
                    if event.get("witness_description"):
                        md_lines.append(f"    - Witness Description: {event['witness_description']}")
                md_lines.append("")
            elif key == "times":
                continue
            else:
                title = key.replace("_", " ").title()
                if isinstance(value, list) and value:
                    md_lines.append(f"### {title}")
                    for item in value:
                        md_lines.append(f"- {item}")
                    md_lines.append("")
                elif isinstance(value, dict) and value:
                    md_lines.append(f"### {title}")
                    for k, v in value.items():
                        md_lines.append(f"- **{k}**: {v}")
                    md_lines.append("")
                elif isinstance(value, str) and value.strip():
                    md_lines.append(f"### {title}")
                    md_lines.append(value.strip())
                    md_lines.append("")

        return "\n".join(md_lines) if md_lines else "No fields extracted."

    # Upload + API call handler
    def di_llm_ext_names_upload(file: str):
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            status_code, time_taken, response = send_request(
                route="doc_intel_extract_city_names",
                data=data,
                headers=headers,
                force_json_content_type=True,
            )
        formatted_md = format_investigation_fields(response)
        return status_code, time_taken, formatted_md, response

    # UI components
    di_llm_ext_names_instructions = gr.Markdown(
        (
            "This example uses Azure Document Intelligence to extract the raw text from a PDF or image file, then "
            "sends the instructions and extracted text to GPT-4o "
            "\n\nThe pipeline is as follows:\n"
            "1. Azure Document Intelligence extracts the raw text from the PDF along with confidence scores.\n"
            "2. GPT-4o is instructed to extract a list of key entities and then summarises them.\n\n"
            "The response includes the final result, as well as intermediate outputs from the processing pipeline."
        ),
        show_label=False,
    )

    with gr.Row():
        di_llm_ext_names_file_upload = gr.File(
            label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
            file_count="single",
            type="filepath",
        )
        di_llm_ext_names_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )

    di_llm_ext_names_examples = gr.Examples(
        examples=DEMO_VISION_FILES,
        inputs=[di_llm_ext_names_file_upload],
        label=FILE_EXAMPLES_LABEL,
        outputs=[di_llm_ext_names_input_thumbs],
        fn=render_visual_media_input,
        run_on_click=True,
    )

    form_ext_w_conf_process_btn = gr.Button("Process File", variant="primary")

    with gr.Column() as di_llm_ext_names_output_row:
        di_llm_ext_names_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            di_llm_ext_names_status_code = gr.Textbox(label="Response Status Code", interactive=False)
            di_llm_ext_names_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        di_llm_ext_names_output_md = gr.Markdown(label="Formatted Response")
        di_llm_ext_names_output_json = gr.JSON(label="API Response")

    form_ext_w_conf_process_btn.click(
        fn=di_llm_ext_names_upload,
        inputs=[di_llm_ext_names_file_upload],
        outputs=[
            di_llm_ext_names_status_code,
            di_llm_ext_names_time_taken,
            di_llm_ext_names_output_md,
            di_llm_ext_names_output_json,
        ],
    )


### Local PDF processing and extraction example ###
### NOTE: This pipeline is disabled in function_app.py due to changing from key-based auth to token-based auth.
### This is because the Haystack generator is instantiated with a token, and does not support cycling of keys.
### You must use key-based auth or update the code to automatically cycle keys if the token expires.
# with gr.Blocks(analytics_enabled=False) as local_pdf_prc_block:
#     # Define requesting function, which reshapes the input into the correct schema
#     def local_pdf_process_upload(file: str):
#         # Get response from the API
#         mime_type = mimetypes.guess_type(file)[0]
#         with open(file, "rb") as f:
#             data = f.read()
#             headers = {"Content-Type": mime_type}
#             return send_request(
#                 route="pymupdf_extract_city_names",
#                 data=data,
#                 headers=headers,
#                 force_json_content_type=True,  # JSON gradio block requires this
#             )

#     # Input components
#     local_pdf_process_instructions = gr.Markdown(
#         (
#             "This example uses PyMuPDF to extract the raw text from a PDF or image file, then "
#             "sends a multimodal request (images and text) to GPT-4o "
#             "([Code Link](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_pymupdf_extract_city_names.py))."
#             "\n\nThe pipeline is as follows:\n"
#             "1. PyMuPDF extracts the raw text from the PDF.\n"
#             "2. The raw extracted text is sent to GPT-4o and it is instructed to extract a list of city names that "
#             "appear in the file.\n\n"
#             "The response includes the final result, as well as many of the intermediate outputs from the processing pipeline."
#         ),
#         show_label=False,
#     )
#     with gr.Row():
#         local_pdf_process_file_upload = gr.File(
#             label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
#             file_count="single",
#             type="filepath",
#         )
#         local_pdf_process_input_thumbs = gr.Gallery(
#             label="File Preview", object_fit="contain", visible=True
#         )
#     # Examples
#     local_pdf_process_examples = gr.Examples(
#         label=FILE_EXAMPLES_LABEL,
#         examples=DEMO_PDF_FILES,
#         inputs=[local_pdf_process_file_upload],
#         outputs=[local_pdf_process_input_thumbs],
#         fn=render_visual_media_input,
#         run_on_click=True,
#     )
#     local_pdf_process_process_btn = gr.Button("Process File", variant="primary")
#     # Output components
#     with gr.Column() as local_pdf_process_output_row:
#         local_pdf_process_output_label = gr.Label(
#             value="API Response", show_label=False
#         )
#         with gr.Row():
#             local_pdf_process_status_code = gr.Textbox(
#                 label="Response Status Code", interactive=False
#             )
#             local_pdf_process_time_taken = gr.Textbox(
#                 label="Time Taken", interactive=False
#             )
#         local_pdf_process_output_json = gr.JSON(label="API Response")
#     # Actions
#     local_pdf_process_process_btn.click(
#         fn=local_pdf_process_upload,
#         inputs=[local_pdf_process_file_upload],
#         outputs=[
#             local_pdf_process_status_code,
#             local_pdf_process_time_taken,
#             local_pdf_process_output_json,
#         ],
#     )

### Doc Intelligence Processing Example ###
with gr.Blocks(analytics_enabled=False) as di_proc_block:
    # Define requesting function, which reshapes the input into the correct schema
    def di_proc_process_upload(
        file: str,
        include_page_images_after_content: bool,
        extract_and_crop_inline_figures: bool,
        pages_per_chunk: int,
    ):
        if file is None:
            gr.Warning(
                "Please select or upload an audio file, then click 'Process File'."
            )
            return ("", "", {})
        # Get response from the API
        with open(file, "rb") as f:
            payload = {
                "include_page_images_after_content": include_page_images_after_content,
                "extract_and_crop_inline_figures": extract_and_crop_inline_figures,
                "pages_per_chunk": pages_per_chunk,
            }
            file_mime_type = mimetypes.guess_type(file)[0]
            files = {
                "file": (file, f, file_mime_type),
                "json": ("payload.json", json.dumps(payload), "application/json"),
            }

            request_outputs = send_request(
                route="multimodal_doc_intel_processing",
                files=files,
                force_json_content_type=False,
            )
        return request_outputs

    # Input components
    di_proc_instructions = gr.Markdown(
        (
            "This example uses showcases a custom-built Azure Document Intelligence Processor for converting raw API "
            "responses into more usable and useful formats "
            "([Code Link](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_multimodal_doc_intel_processing.py))."
            "\n\nThis processor is fully configurable and can be used to completely customize the processing, formatting, "
            "and chunking of files processed with Azure Document Intelligence. Some features include:\n"
            "* Automatic extraction of rich content from images and PDF/documents, including tables, figures and more.\n"
            "* Conversion and formatting of text content, images for each page and figure, and pandas dataframes for tables.\n"
            "* Automatic correction of image rotation when extracting page and figure images (if not corrected, this can completely destroy LLM extraction accuracy)\n"
            "* Custom formatting of all content outputs, allowing for completely dynamic formatting and inclusion/exclusion of content.\n"
            "* Chunking of content into smaller parts (e.g. into chunks of X pages) which can then be processed in parallel (e.g. the Map Reduce pattern).\n"
            "* Automatic conversion of the content to the OpenAI message format, ready for processing with an LLM.\n"
            "\n\nTo try it out, upload a document or select one of the demo documents from the dropdown below. "
            "While the demo below is a very simple example of what is possible with the component, a more detailed deep dive can be found in the "
            "[Code walkthrough notebook](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/notebooks/Document%20Intelligence%20Processor%20Walkthrough.ipynb)."
        ),
        show_label=False,
    )
    with gr.Row():
        di_proc_file_upload = gr.File(
            label="Upload File. To upload a different file, Hit the 'X' button to the top right of this element ->",
            file_count="single",
            type="filepath",
            value=lambda: DEMO_DOC_INTEL_PROCESSING_FILE_TUPLES[0][1],
        )
        di_proc_input_thumbs = render_visual_media_input(
            DEMO_DOC_INTEL_PROCESSING_FILE_TUPLES[0][1]
        )
    # Examples
    di_proc_example_dropdown = gr.Dropdown(
        label="Select a demo PDF or image file",
        choices=DEMO_DOC_INTEL_PROCESSING_FILE_TUPLES,
        allow_custom_value=True,
    )
    with gr.Row():
        di_proc_output_page_checkbox = gr.Checkbox(
            label="Include page images after each page's content",
            value=True,
            info="If checked, the output will include the page image after each page's text content.",
        )
        di_proc_output_figures_inline_checkbox = gr.Checkbox(
            label="Extract & crop inline figures",
            value=True,
            info="If checked, any figures that were identified within the document will be cropped from the page image and inserted into the outputs as an image file.",
        )
        di_proc_pages_per_chunk_number = gr.Number(
            label="Pages per chunk of content",
            value=3,
            minimum=1,
            maximum=1000,
            precision=0,
            info="The demo automatically splits the output content into chunks. This sets the number of pages of content per chunk.",
        )
    di_proc_process_btn = gr.Button("Process File", variant="primary")
    # Output components
    with gr.Column() as di_proc_output_row:
        di_proc_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            di_proc_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            di_proc_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        di_proc_output_md = gr.Markdown(
            label="Processed Markdown Content", line_breaks=True
        )
    # Actions
    di_proc_example_dropdown.change(
        fn=echo_input,
        inputs=[di_proc_example_dropdown],
        outputs=[di_proc_file_upload],
    )
    di_proc_file_upload.change(
        fn=render_visual_media_input,
        inputs=[di_proc_file_upload],
        outputs=[di_proc_input_thumbs],
    )
    di_proc_process_btn.click(
        fn=di_proc_process_upload,
        inputs=[
            di_proc_file_upload,
            di_proc_output_page_checkbox,
            di_proc_output_figures_inline_checkbox,
            di_proc_pages_per_chunk_number,
        ],
        outputs=[
            di_proc_status_code,
            di_proc_time_taken,
            di_proc_output_md,
        ],
    )

### Document Translation Example ###
with gr.Blocks(analytics_enabled=False) as doc_translate_block:
    def translate_doc_request(file: Optional[str]):
        translated_text = ""
        summary_text = ""
        if file is None:
            gr.Warning(
                "Please select or upload a PDF file to translate."
            )
            return ("", "", translated_text, summary_text)
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            status_code, time_taken, response = send_request(
                route="translate_document",
                data=data,
                headers=headers,
                force_json_content_type=True,
            )
        if isinstance(response, dict):
            translated_text = response.get("translation", "")
            summary_text = response.get("summary", "")
        return status_code, time_taken, translated_text, summary_text

    doc_translate_instructions = gr.Markdown(
        "Upload a PDF document. The file will be translated to English and summarised below."
    )
    with gr.Row():
        with gr.Column():
            doc_translate_file_upload = gr.File(
                label="Upload PDF", file_count="single", type="filepath"
            )
            doc_translate_input_thumbs = gr.Gallery(
                label="File Preview", object_fit="contain", visible=True
            )
        with gr.Column():
            with gr.Row():
                doc_translate_status_code = gr.Textbox(
                    label="Response Status Code", interactive=False
                )
                doc_translate_time_taken = gr.Textbox(
                    label="Time Taken", interactive=False
                )
            doc_translate_translation = gr.Textbox(
                label="Translated Text", lines=10
            )
            doc_translate_summary = gr.Textbox(label="Summary", lines=3)

    doc_translate_file_upload.change(
        fn=render_visual_media_input,
        inputs=[doc_translate_file_upload],
        outputs=[doc_translate_input_thumbs],
    )
    doc_translate_file_upload.change(
        fn=translate_doc_request,
        inputs=[doc_translate_file_upload],
        outputs=[
            doc_translate_status_code,
            doc_translate_time_taken,
            doc_translate_translation,
            doc_translate_summary,
        ],
    )
    
# Gradio tab for audio upload and transcript viewing

def audio_transcription_tab(blob_service_client, input_container="audio-in", output_container="audio-transcript-out"):
    
    def upload_to_blob(file):
        try:
            file_path = file.name if hasattr(file, "name") else file
            blob_name = os.path.basename(file_path)
            container_client = blob_service_client.get_container_client(input_container)
            with open(file_path, "rb") as data:
                container_client.upload_blob(blob_name, data, overwrite=True)
            return f"✅ Uploaded `{blob_name}` to `{input_container}`."
        except Exception as e:
            return f"❌ Upload failed: {str(e)}"

    def list_output_blobs():
        try:
            container_client = blob_service_client.get_container_client(output_container)
            return [blob.name for blob in container_client.list_blobs()]
        except Exception as e:
            return [f"❌ Failed to list transcripts: {str(e)}"]

    def get_transcript_choices():
        return gr.update(choices=list_output_blobs())

    def show_selected_blob(name):
        try:
            container_client = blob_service_client.get_container_client(output_container)
            blob = container_client.download_blob(name)
            return blob.readall().decode("utf-8")
        except Exception as e:
            return f"❌ Failed to read transcript: {str(e)}"

    def download_transcript_json(name):
        try:
            container_client = blob_service_client.get_container_client(output_container)
            blob = container_client.download_blob(name)
            content = blob.readall()

            json_data = json.loads(content)
            pretty = json.dumps(json_data, indent=2)

            temp_path = os.path.join(tempfile.gettempdir(), name)
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(pretty)

            return temp_path
        except Exception as e:
            logging.warning(f"Failed to prep download: {e}")
            return None

    # Gradio UI layout
    with gr.Row():
        with gr.Column():
            audio_input = gr.File(
                label="Upload audio or video file",
                type="filepath",
                file_types=["audio", "video"]
            )
            upload_btn = gr.Button("Upload")
            upload_status = gr.Textbox(label="Status", interactive=False)

            upload_btn.click(
                fn=upload_to_blob,
                inputs=audio_input,
                outputs=upload_status
            )

        with gr.Column():
            refresh_btn = gr.Button("Refresh Transcripts")
            transcript_dropdown = gr.Dropdown(choices=[], label="Available Transcripts")
            transcript_output = gr.Textbox(label="Transcript Content", lines=10)

            download_btn = gr.Button("Download JSON")
            download_file = gr.File(label="Your JSON Download", interactive=False)

            refresh_btn.click(
                fn=get_transcript_choices,
                inputs=[],
                outputs=transcript_dropdown
            )

            transcript_dropdown.change(
                fn=show_selected_blob,
                inputs=transcript_dropdown,
                outputs=transcript_output
            )

            download_btn.click(
                fn=download_transcript_json,
                inputs=transcript_dropdown,
                outputs=download_file
            )

## New video processing block
def video_processing_tab(blob_service_client, input_container="video-in", output_container="video-processed-out"):
    
    def upload_video_to_blob(file):
        try:
            file_path = file.name if hasattr(file, "name") else file
            blob_name = os.path.basename(file_path)
            container_client = blob_service_client.get_container_client(input_container)
            with open(file_path, "rb") as data:
                container_client.upload_blob(blob_name, data, overwrite=True)
            return f"✅ Uploaded `{blob_name}` to `{input_container}`."
        except Exception as e:
            return f"❌ Upload failed: {str(e)}"

    def list_video_outputs():
        try:
            container_client = blob_service_client.get_container_client(output_container)
            return [blob.name for blob in container_client.list_blobs()]
        except Exception as e:
            return [f"❌ Failed to list outputs: {str(e)}"]

    def update_video_dropdown():
        return gr.update(choices=list_video_outputs())

    def show_video_output(name):
        try:
            container_client = blob_service_client.get_container_client(output_container)
            blob = container_client.download_blob(name)
            return blob.readall().decode("utf-8")
        except Exception as e:
            return f"❌ Failed to read output: {str(e)}"

    def download_video_output(name):
        try:
            container_client = blob_service_client.get_container_client(output_container)
            blob = container_client.download_blob(name)
            content = blob.readall()

            json_data = json.loads(content)
            pretty = json.dumps(json_data, indent=2)

            temp_path = os.path.join(tempfile.gettempdir(), name)
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(pretty)

            return temp_path
        except Exception as e:
            logging.warning(f"Failed to prep video download: {e}")
            return None

    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="Upload video file",
                type="filepath",
                file_types=["video"]
            )
            video_upload_btn = gr.Button("Upload")
            video_upload_status = gr.Textbox(label="Status", interactive=False)

            video_upload_btn.click(
                fn=upload_video_to_blob,
                inputs=video_input,
                outputs=video_upload_status
            )

        with gr.Column():
            video_refresh_btn = gr.Button("Refresh Processed Outputs")
            video_dropdown = gr.Dropdown(choices=[], label="Available Outputs")
            video_output_display = gr.Textbox(label="Output Content", lines=10)

            video_download_btn = gr.Button("Download JSON")
            video_download_file = gr.File(label="Your JSON Download", interactive=False)

            video_refresh_btn.click(
                fn=update_video_dropdown,
                inputs=[],
                outputs=video_dropdown
            )

            video_dropdown.change(
                fn=show_video_output,
                inputs=video_dropdown,
                outputs=video_output_display
            )

            video_download_btn.click(
                fn=download_video_output,
                inputs=video_dropdown,
                outputs=video_download_file
            )


version="0.13"
## Version Control ###
## v0.13 - Updated with translation tab
## v0.12 - Updated with video processing block
##  v0.11 - Updated with Audio Batch Processing and Blob Storage with downnload
##  v0.10 - Updated with Audio in and out blob storage
##  v0.9 - Updated with version and CSS
##  v0.8 - Updated with video/audio processing and markdown for Doc Intelligence
##  v0.7 - Audio Fixes, and updated with markdown for Doc Intelligence
##  v0.6 - Updated with markdown for Audio
##  v0.5 - Updated with markdown for Document Intelligenc
##  v0.4 - Updated with markdown for Form Extraction
##  v0.3 - Updated with markdown for PII Redaction

with gr.Blocks(
    title=f"Briefcase the AI Toolbox {version}",
    theme=Base(),
    css="footer {visibility: hidden}",
    analytics_enabled=False,
) as demo:
    gr.Markdown(
        (
            f"## Briefcase the AI Toolbox {version}\n\n"
            "\n\nThis demo app showcases a few examples of different processing pipelines that incorporate Azure AI "
        ),
        show_label=False,
    )
#    with gr.Tab("Azure Content Understanding Demos (HTTP)"):
#        simple_cu_examples_block.render()
#    with gr.Tab("Form Extraction with Confidence Scores (HTTP)"):
#        form_extraction_with_confidence_block.render()
    with gr.Tab("Audio Processing (HTTP)"):
        call_center_audio_processing_block.render()
#    with gr.Tab("Multimodal Document Intelligence Processing (HTTP)"):
#        di_proc_block.render()
    if IS_COSMOSDB_AVAILABLE:
        # Only render the CosmosDB tab if CosmosDB database info is available
        with gr.Tab("Form Extraction (Blob -> CosmosDB)"):
            blob_form_extraction_to_cosmosdb_block.render()
    with gr.Tab("Summarise Text (HTTP)"):
        sum_text_block.render()
    with gr.Tab("PII Redaction (HTTP)"):
        pii_redaction_block.render()
    with gr.Tab("Key Information Extraction, Doc Intelligence (HTTP)"):
        di_llm_ext_names_block.render()
    with gr.Tab("Document Translation (HTTP)"):
        doc_translate_block.render()
    with gr.Tab("Audio Batch Processing (HTTP)"):
        audio_transcription_tab(blob_service_client)
#    with gr.Tab("Video Processing"):
#        video_processing_tab(blob_service_client)


if __name__ == "__main__":
    # Start server by running: `gradio demo_app.py`, then navigate to http://localhost:8000
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        auth_message=auth_message,
        auth=auth,
    )