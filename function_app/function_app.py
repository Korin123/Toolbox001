import logging
import os
from datetime import datetime, timedelta
import json
import traceback
import requests
import asyncio
import time
import azure.functions as func
import azure.durable_functions as df
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from dotenv import load_dotenv
from src.helpers.azure_function import (
    check_if_azurite_storage_emulator_is_running,
    check_if_env_var_is_set,
)
from azure.identity import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from azure.storage.blob import BlobServiceClient as SyncBlobServiceClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",)

_logger = logging.getLogger("azure")
_logger.setLevel(logging.WARNING)

speech_endpoint = os.getenv("SPEECH_ENDPOINT")
if not speech_endpoint:
    raise Exception("SPEECH_ENDPOINT environment variable is not set")

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)

### Get Environment Variables and Check Deployments

IS_CONTENT_UNDERSTANDING_DEPLOYED = check_if_env_var_is_set("CONTENT_UNDERSTANDING_ENDPOINT")
IS_AOAI_DEPLOYED = check_if_env_var_is_set("AOAI_ENDPOINT")
IS_DOC_INTEL_DEPLOYED = check_if_env_var_is_set("DOC_INTEL_ENDPOINT")
IS_SPEECH_DEPLOYED = check_if_env_var_is_set("SPEECH_ENDPOINT")
IS_LANGUAGE_DEPLOYED = check_if_env_var_is_set("LANGUAGE_ENDPOINT")
IS_STORAGE_ACCOUNT_AVAILABLE = (
    os.getenv("AzureWebJobsStorage") == "UseDevelopmentStorage=true"
    and check_if_azurite_storage_emulator_is_running()
) or all([
    check_if_env_var_is_set("AzureWebJobsStorage__accountName"),
    check_if_env_var_is_set("AzureWebJobsStorage__blobServiceUri"),
    check_if_env_var_is_set("AzureWebJobsStorage__queueServiceUri"),
    check_if_env_var_is_set("AzureWebJobsStorage__tableServiceUri"),
])
IS_COSMOSDB_AVAILABLE = check_if_env_var_is_set("COSMOSDB_DATABASE_NAME") and check_if_env_var_is_set("CosmosDbConnectionSetting__accountEndpoint")

### Registering Blueprints

if IS_AOAI_DEPLOYED:
    from bp_summarize_text import bp_summarize_text
    app.register_blueprint(bp_summarize_text)
if IS_DOC_INTEL_DEPLOYED and IS_AOAI_DEPLOYED:
    from bp_doc_intel_extract_city_names import bp_doc_intel_extract_city_names
    from bp_form_extraction_with_confidence import bp_form_extraction_with_confidence
    app.register_blueprint(bp_doc_intel_extract_city_names)
    app.register_blueprint(bp_form_extraction_with_confidence)
if IS_SPEECH_DEPLOYED and IS_AOAI_DEPLOYED:
    from bp_call_center_audio_analysis import bp_call_center_audio_analysis
    app.register_blueprint(bp_call_center_audio_analysis)
if IS_DOC_INTEL_DEPLOYED:
    from bp_multimodal_doc_intel_processing import bp_multimodal_doc_intel_processing
    app.register_blueprint(bp_multimodal_doc_intel_processing)
if IS_CONTENT_UNDERSTANDING_DEPLOYED:
    from bp_content_understanding_audio import bp_content_understanding_audio
    from bp_content_understanding_document import bp_content_understanding_document
    from bp_content_understanding_image import bp_content_understanding_image
    from bp_content_understanding_video import bp_content_understanding_video
    app.register_blueprint(bp_content_understanding_document)
    app.register_blueprint(bp_content_understanding_video)
    app.register_blueprint(bp_content_understanding_audio)
    app.register_blueprint(bp_content_understanding_image)
if IS_LANGUAGE_DEPLOYED:
    from bp_pii_redaction import bp_pii_redaction
    app.register_blueprint(bp_pii_redaction)

if IS_STORAGE_ACCOUNT_AVAILABLE and IS_COSMOSDB_AVAILABLE and IS_AOAI_DEPLOYED and IS_DOC_INTEL_DEPLOYED:
    from extract_blob_field_info_to_cosmosdb import get_structured_extraction_func_outputs
    COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME")

    @app.function_name("blob_form_extraction_to_cosmosdb")
    @app.blob_trigger(
        arg_name="inputblob1",
        path="blob-form-to-cosmosdb-blobs/{name}",
        connection="AzureWebJobsStorage",
    )
    @app.cosmos_db_output(
        arg_name="outputdocument",
        connection="CosmosDbConnectionSetting",
        database_name=COSMOSDB_DATABASE_NAME,
        container_name="blob-form-to-cosmosdb-container",
    )
    def extract_blob_pdf_fields_to_cosmosdb(inputblob, outputdocument):
        output_result = get_structured_extraction_func_outputs(inputblob)
        outputdocument.set(func.Document.from_dict(output_result))


### file waiting

import aiohttp
from azure.storage.blob.aio import BlobServiceClient

async def wait_for_blob_ready_async(blob_url, container_name, max_retries=6, delay=5):
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=f"https://{os.getenv('STORAGE_ACCOUNT_NAME')}.blob.core.windows.net",
            credential=credential
        )
        container_client = blob_service_client.get_container_client(container_name)
        blob_name = blob_url.split("/")[-1]
        blob_client = container_client.get_blob_client(blob_name)

        for _ in range(max_retries):
            props = await blob_client.get_blob_properties()
            if props.size > 0:
                await blob_service_client.close()
                return True
            await asyncio.sleep(delay)

        await blob_service_client.close()
        return False
    except Exception as e:
        logging.error(f"❌ Error checking blob readiness in {container_name}: {e}")
        return False


def is_blob_ready_sync(blob_url, container_name, max_retries=6, delay=5):
    try:
        credential = DefaultAzureCredential()
        from azure.storage.blob import BlobServiceClient as SyncBlobServiceClient
        storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
        blob_service_client = SyncBlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=credential
        )
        container_client = blob_service_client.get_container_client(container_name)
        blob_name = blob_url.split("/")[-1]
        blob_client = container_client.get_blob_client(blob_name)
        for _ in range(max_retries):
            props = blob_client.get_blob_properties()
            if props.size > 0:
                return True
            time.sleep(delay)
        return False
    except Exception as e:
        logging.error(f"❌ Error checking blob readiness (sync) in {container_name}: {e}")
        return False


### Blob trigger for audio processing using Azure Durable Functions

@app.function_name("audio_blob_trigger")
@app.blob_trigger(
    arg_name="inputblob2",
    path="audio-in/{name}",
    connection="AzureWebJobsStorage",
)
async def audio_blob_trigger(inputblob2: func.InputStream):
    logging.warning("🔥 Blob trigger fired!")
    blob_name = inputblob2.name.replace("audio-in/", "")
    logging.warning(f"Blob name: {blob_name}")

    try:
        await asyncio.sleep(7)  # Simulate some processing delay
        storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
        if not storage_account_name:
            raise ValueError("Missing STORAGE_ACCOUNT_NAME environment variable")
        content_url = f"https://{storage_account_name}.blob.core.windows.net/audio-in/{blob_name}"
        logging.warning(f"✅ Recordings URL: {content_url}")
        # Wait until blob is actually ready before processing in audio-in
        if not await wait_for_blob_ready_async(content_url, "audio-in"):
            logging.error("❌ Blob not ready after retries. Skipping processing.")
            return
        # Call the audiomono container app to convert to mono and upload to audio-preprocessed
        mono_url = await call_audiomono_container(
            blob_url=content_url,
            storage_account_name=storage_account_name,
            source_container="audio-in",
            dest_container="audio-preprocessed"
        )
        if not mono_url:
            logging.error(f"❌ Mono conversion/upload failed for {blob_name}")
            return
        logging.info(f"🎵 Mono file available at: {mono_url}")
        # Wait until mono file is ready in audio-preprocessed (optional, for further logic)
        if not await wait_for_blob_ready_async(mono_url, "audio-preprocessed"):
            logging.error("❌ Mono blob not ready after upload. Skipping further processing.")
            return
        # Optionally, trigger further processing/orchestration here if needed
    except Exception as e:
        logging.error(f"❌ Error in blob trigger: {e}")
        logging.error(traceback.format_exc())



@app.orchestration_trigger(context_name="context")
def audio_processing_orchestrator(context):
    try:
        input_data = context.get_input()
        logging.info(f"[Orchestrator] Started with input: {input_data}")
        content_url = input_data.get("content_url")
        blob_name = input_data.get("blob_name")

        logging.info("[Orchestrator] Calling start_batch_activity")
        batch_info = yield context.call_activity("start_batch_activity", {
            "content_url": content_url,
            "blob_name": blob_name
        })
        logging.info(f"[Orchestrator] batch_info: {batch_info}")

        logging.info("[Orchestrator] Calling poll_batch_activity")
        result_data = yield context.call_activity("poll_batch_activity", batch_info)
        logging.info(f"[Orchestrator] result_data from poll_batch_activity: {result_data}")

        if not result_data or "error" in result_data:
            logging.error(f"❌ Error in batch processing: {result_data.get('error') if result_data else 'No result_data'}")
            return {"error": result_data.get("error") if result_data else "No result_data"}

        # FIX: Actually call write_output_activity here
        logging.info("[Orchestrator] Calling write_output_activity")
        write_result = yield context.call_activity("write_output_activity", result_data)
        logging.info(f"[Orchestrator] write_output_activity result: {write_result}")

    except Exception as e:
        logging.error(f"❌ Orchestration failed: {e}")
        return {"error": str(e)}



@app.activity_trigger(input_name="input_data")
def start_batch_activity(input_data):
    logging.warning(f"[Activity] Received input: {input_data}")
    content_url = input_data["content_url"]
    blob_name = input_data["blob_name"]

    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    if not storage_account_name:
        raise ValueError("Missing STORAGE_ACCOUNT_NAME environment variable")

    credential = DefaultAzureCredential()

    # Use sync blob readiness check to avoid asyncio issues in activity context
    # FIX: Use the correct container name
    if not is_blob_ready_sync(content_url, "audio-preprocessed"):
        raise Exception(f"Blob {blob_name} not ready after retries. Aborting transcription.")

    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    logging.warning(f"Token obtained for: {token.token[:20]}...")

    transcription_url = f"{speech_endpoint}/speechtotext/v3.1/transcriptions"
    payload = {
        "displayName": f"Transcription - {blob_name}",
        "locale": "en-GB",
        "contentUrls": [
            f"https://{storage_account_name}.blob.core.windows.net/audio-preprocessed/{blob_name}"
        ],
        "properties": {
            "wordLevelTimestampsEnabled": True,
            "displayFormWordLevelTimestampsEnabled": True,
            "diarizationEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "None",
            "transcriptionMode": "Batch",
            "timeToLive": "PT1H"
        }
    }

    token = credential.get_token("https://cognitiveservices.azure.com/.default").token
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    response = requests.post(transcription_url, json=payload, headers=headers)
    if response.status_code not in [200, 201, 202]:
        logging.error(f"[Activity] Failed to start transcription: {response.status_code} - {response.text}")
        raise Exception("Batch transcription start failed")

    transcription_location = response.headers["Location"]
    logging.warning(f"[Activity] Transcription started: {transcription_location}")
    result = {"status_url": transcription_location, "blob_name": blob_name}
    logging.warning(f"[Activity] Returning from start_batch_activity: {result}")
    return result



@app.activity_trigger(input_name="batch_info")
def poll_batch_activity(batch_info):
    logging.warning(f"[Activity] Polling real batch job: {batch_info}")
    try:
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        headers = {"Authorization": f"Bearer {token}"}

        status_url = batch_info["status_url"]
        blob_name = batch_info["blob_name"]

        for attempt in range(30):
            response = requests.get(status_url, headers=headers)
            if response.status_code != 200:
                logging.error(f"❌ Polling failed: {response.status_code} - {response.text}")
                raise Exception(f"Polling failed: {response.status_code} - {response.text}")

            data = response.json()
            status = data.get("status")
            logging.info(f"[Polling Attempt {attempt + 1}] Status: {status}")

            if status == "Succeeded":
                transcript_url = data.get("resultsUrls", {}).get("transcription")

                if not transcript_url:
                    logging.warning("⚠️ No 'transcription' in resultsUrls. Trying files fallback...")
                    files_url = data.get("links", {}).get("files")
                    if not files_url:
                        raise Exception("No fallback 'files' link provided by API.")

                    logging.info(f"🔍 Fetching file list from: {files_url}")
                    files_response = requests.get(files_url, headers=headers)
                    files_response.raise_for_status()
                    files = files_response.json().get("values", [])

                    logging.info(f"🗂️ Available files: {[f.get('name') for f in files]}")
                    transcription_file = next((f for f in files if f.get("kind") == "Transcription"), None)
                    if not transcription_file:
                        logging.warning("⏳ Transcription file not ready. Retrying...")
                        time.sleep(15)
                        continue

                    transcript_url = transcription_file.get("links", {}).get("contentUrl")
                    if not transcript_url:
                        raise Exception("Transcription file found but missing 'contentUrl'.")

                logging.info(f"📥 Fetching transcript JSON from: {transcript_url}")
                result_response = requests.get(transcript_url)
                result_response.raise_for_status()
                result_json = result_response.json()

                phrases = result_json.get("recognizedPhrases", [])
                segments = []
                for p in phrases:
                    text = p.get("display", "")
                    if not text and "nBest" in p and p["nBest"]:
                        text = p["nBest"][0].get("display", "")
                    segment = {
                        "speaker": p.get("speaker", "Unknown"),
                        "text": text,
                        "offset": p.get("offset"),
                        "duration": p.get("duration")
                    }
                    segments.append(segment)

                # Optionally, group by speaker for better readability
                grouped = {}
                for seg in segments:
                    speaker = seg["speaker"]
                    grouped.setdefault(speaker, []).append(seg["text"])

                # Build a readable transcript
                full_text = ""
                for speaker, texts in grouped.items():
                    full_text += f"\nSpeaker {speaker}:\n" + "\n".join(texts)

                result = {
                    "result": {
                        "transcript": full_text.strip(),
                        "segments": segments,
                        "speakers_detected": list(grouped.keys())
                    },
                    "blob_name": blob_name
                }
                logging.info(f"[Activity] poll_batch_activity returning: {result}")
                return result

            elif status in ["Failed", "Rejected"]:
                logging.error(f"❌ Transcription failed with response: {json.dumps(data, indent=2)}")
                raise Exception(f"Transcription failed: {json.dumps(data)}")

            time.sleep(30)
            logging.info(f"⏳ Waiting for 30 seconds before next polling attempt...")

        logging.error("❌ Polling timed out after maximum attempts.")
        return {"error": "Polling timed out", "blob_name": blob_name}
    except Exception as e:
        logging.error(f"[Activity] Exception in poll_batch_activity: {e}")
        logging.error(traceback.format_exc())
        return {"error": str(e), "blob_name": batch_info.get("blob_name")}

from azure.core.exceptions import ClientAuthenticationError, ResourceExistsError, HttpResponseError

@app.activity_trigger(input_name="result_data")
def write_output_activity(result_data):
    logging.info(f"[Activity] Writing result to blob: {result_data}")
    try:
        credential = DefaultAzureCredential()
        storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
        if not storage_account_name:
            raise ValueError("Missing STORAGE_ACCOUNT_NAME")
        # Use only the sync BlobServiceClient here
        from azure.storage.blob import BlobServiceClient as SyncBlobServiceClient
        blob_service_client = SyncBlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=credential
        )
        container_name = "audio-transcript-out"
        blob_name = result_data["blob_name"].rsplit(".", 1)[0] + ".json"
        output_container = blob_service_client.get_container_client(container_name)
        output_blob = output_container.get_blob_client(blob_name)
        blob_url = output_blob.url
        logging.info(f"[Activity] Using storage account: {storage_account_name}")
        logging.info(f"[Activity] AzureWebJobsStorage: {os.getenv('AzureWebJobsStorage')}")
        logging.info(f"[Activity] Full blob URL: {blob_url}")
        try:
            logging.info(f"[Activity] Azure identity: {credential.__class__.__name__}")
        except Exception:
            pass
        content = json.dumps(result_data["result"], indent=2)
        try:
            output_blob.upload_blob(content, overwrite=True)
            if output_blob.exists():
                blob_names = []
                for b in output_container.list_blobs():
                    blob_names.append(b.name)
                logging.info(f"[Activity] Output written to: {container_name}/{blob_name}")
                logging.info(f"[Activity] Verified blob exists: {blob_url}")
                logging.info(f"[Activity] Blobs currently in {container_name}: {blob_names}")
                return "OK"
            else:
                logging.error(f"[Activity] Blob upload reported success but blob does NOT exist: {blob_url}")
                return "ERROR"
        except ClientAuthenticationError as e:
            logging.error("[Activity] ClientAuthenticationError: Check managed identity or credentials.")
            logging.error(f"[Activity] Failed to upload blob: {e}")
            return "ERROR"
        except HttpResponseError as e:
            logging.error(f"[Activity] HttpResponseError: {e.status_code} - {e.message}")
            logging.error(f"[Activity] Failed to upload blob: {e}")
            return "ERROR"
        except Exception as e:
            logging.error(f"[Activity] Failed to upload blob: {e}")
            return "ERROR"
    except Exception as e:
        logging.error(f"[Activity] Failed to write output: {e}")
        return "ERROR"


async def call_audiomono_container(blob_url, storage_account_name, source_container, dest_container):
    """
    Calls the audiomono container app to convert stereo audio to mono.
    """
    audiomono_url = os.getenv("AUDIOMONO_ENDPOINT")  # e.g., https://audiomono-function-app.azurewebsites.net
    if not audiomono_url:
        logging.error("AUDIOMONO_ENDPOINT environment variable is not set")
        return None
    payload = {
        "blob_url": blob_url,
        "storage_account_name": storage_account_name,
        "source_container": source_container,
        "dest_container": dest_container
    }
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{audiomono_url}/convert-to-mono", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("mono_url")
                else:
                    text = await resp.text()
                    logging.error(f"Failed to call audiomono container: {resp.status} - {text}")
                    return None
    except Exception as e:
        logging.error(f"Exception calling audiomono container: {e}")
        return None


@app.function_name("audio_preprocessed_blob_trigger")
@app.blob_trigger(
    arg_name="inputblob3",
    path="audio-preprocessed/{name}",
    connection="AzureWebJobsStorage",
)
@app.durable_client_input(client_name="client")
async def audio_preprocessed_blob_trigger(
    inputblob3: func.InputStream,
    client: df.DurableOrchestrationClient
):
    logging.warning("🔥 Preprocessed blob trigger fired!")
    blob_name = inputblob3.name.replace("audio-preprocessed/", "")
    logging.warning(f"Preprocessed blob name: {blob_name}")

    try:
        await asyncio.sleep(7)  # Simulate some processing delay
        storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
        if not storage_account_name:
            raise ValueError("Missing STORAGE_ACCOUNT_NAME environment variable")
        content_url = f"https://{storage_account_name}.blob.core.windows.net/audio-preprocessed/{blob_name}"
        logging.warning(f"✅ Preprocessed audio URL: {content_url}")
        # Wait until blob is actually ready before processing in audio-preprocessed
        if not await wait_for_blob_ready_async(content_url, "audio-preprocessed"):
            logging.error("❌ Preprocessed blob not ready after retries. Skipping processing.")
            return

        input_payload = {
            "content_url": content_url,
            "blob_name": blob_name
        }

        instance_id = await client.start_new("audio_processing_orchestrator", None, input_payload)
        logging.info(f"🎯 Orchestration started: {instance_id}")
    except Exception as e:
        logging.error(f"❌ Error in preprocessed blob trigger: {e}")
        logging.error(traceback.format_exc())

##### Adding video processing orchestration

# --- 1) BLOB TRIGGER: START ORCHESTRATOR ----------------------------------

@app.function_name("video_blob_trigger")
@app.blob_trigger(
    arg_name="inputblob_video",
    path="video-in/{name}",
    connection="AzureWebJobsStorage",
)
@app.durable_client_input(client_name="starter")
async def video_blob_trigger(
    inputblob_video: func.InputStream,
    starter: df.DurableOrchestrationClient
):
    logging.warning("🎥 Video blob trigger fired!")

    blob_name = inputblob_video.name.replace("video-in/", "")
    account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    if not account_name:
        logging.error("Missing STORAGE_ACCOUNT_NAME")
        return

    content_url = f"https://{account_name}.blob.core.windows.net/video-in/{blob_name}"
    logging.warning(f"🎥 Raw video URL: {content_url}")

    try:
        # 1. Wait until the blob is fully uploaded
        if not await wait_for_blob_ready_async(content_url, "video-in"):
            logging.error("❌ Video blob not ready after retries. Skipping.")
            return

        # 2. Call Video Indexer to start processing; get back a videoId
        video_indexer_account = os.getenv("VIDEO_INDEXER_ACCOUNT")
        video_id = await call_video_indexer(content_url, video_indexer_account)
        if not video_id:
            logging.error(f"❌ Video indexing submission failed for: {blob_name}")
            return

        logging.info(f"✅ Video indexing submitted, videoId: {video_id}")

        # 3. Kick off the durable orchestrator, passing videoId & blobName
        instance_id = await starter.start_new(
            "video_processing_orchestrator",
            client_input={ "videoId": video_id, "blobName": blob_name }
        )
        logging.warning(f"🎬 Orchestrator started with ID = {instance_id}")

    except Exception as e:
        logging.error(f"❌ Error in video_blob_trigger: {e}")
        logging.error(traceback.format_exc())


# --- 2) ORCHESTRATOR FUNCTION --------------------------------------------

@app.orchestration_trigger(context_name="context")
def video_processing_orchestrator(context: df.DurableOrchestrationContext):
    """
    1) Poll Video Indexer until processed
    2) Write insights JSON back into blob storage
    """
    input_data = context.get_input()
    video_id = input_data.get("videoId")
    blob_name = input_data.get("blobName")

    # 1. Poll until Video Indexer marks it as "Processed"
    poll_result = yield context.call_activity("poll_video_index_activity", {
        "videoId": video_id,
        "blobName": blob_name
    })
    if not poll_result or poll_result.get("error"):
        error_msg = poll_result.get("error") if poll_result else "Unknown polling error"
        logging.error(f"❌ Polling failed: {error_msg}")
        return { "error": error_msg }

    # 2. Write the final insights JSON into "video-processed-out"
    write_result = yield context.call_activity("write_video_output_activity", poll_result)
    return write_result


# --- 3) ACTIVITY: POLL VIDEO INDEXER --------------------------------------

@app.activity_trigger(input_name="input_data")
def poll_video_index_activity(input_data: dict) -> dict:
    """
    Polls Azure Video Indexer until the "state" is "Processed".
    Returns { "insightsJson": <full JSON>, "blobName": <original> }
    or { "error": "...", "blobName": <original> }.
    """
    try:
        account_id = os.getenv("VIDEO_INDEXER_ACCOUNT")
        location = os.getenv("VIDEO_INDEXER_LOCATION")
        video_id = input_data["videoId"]
        blob_name = input_data["blobName"]

        # 1. Acquire AAD token via managed identity
        token = DefaultAzureCredential().get_token(
            "https://api.videoindexer.ai/.default"
        ).token

        headers = {
            "Ocp-Apim-Subscription-Key": os.getenv("VIDEO_INDEXER_KEY"),
            "Authorization": f"Bearer {token}"
        }
        index_url = f"https://api.videoindexer.ai/{location}/Accounts/{account_id}/Videos/{video_id}/Index"

        # 2. Poll for up to ~7.5 minutes (30 attempts × 15s delay)
        for attempt in range(30):
            resp = requests.get(index_url, headers=headers, params={"language": "English"})
            if resp.status_code != 200:
                raise Exception(f"Polling error {resp.status_code}: {resp.text}")

            data = resp.json()
            state = data.get("state")
            logging.info(f"[Polling Attempt {attempt+1}] State = {state}")

            if state == "Processed":
                return { "insightsJson": data, "blobName": blob_name }
            elif state in ("Failed", "Rejected"):
                return { "error": f"Indexing failed: {json.dumps(data)}", "blobName": blob_name }

            time.sleep(15)

        return { "error": "Polling timed out", "blobName": blob_name }

    except Exception as e:
        logging.error(f"❌ Exception in poll_video_index_activity: {e}")
        logging.error(traceback.format_exc())
        return { "error": str(e), "blobName": input_data.get("blobName") }


# --- 4) ACTIVITY: WRITE INSIGHTS TO BLOB ----------------------------------

@app.activity_trigger(input_name="input_data")
def write_video_output_activity(input_data: dict) -> str:
    """
    Writes the insights JSON into the "video-processed-out" container.
    Blob name will be "<originalName>-insights.json".
    """
    try:
        insights = input_data["insightsJson"]
        blob_name = input_data["blobName"]
        storage_account = os.getenv("STORAGE_ACCOUNT_NAME")

        credential = DefaultAzureCredential()
        client = SyncBlobServiceClient(
            account_url=f"https://{storage_account}.blob.core.windows.net",
            credential=credential
        )
        container = client.get_container_client("video-processed-out")

        output_name = blob_name.rsplit(".", 1)[0] + "-insights.json"
        content = json.dumps(insights, indent=2)
        container.upload_blob(output_name, content, overwrite=True)

        logging.info(f"✅ Wrote insights blob: video-processed-out/{output_name}")
        return "OK"

    except Exception as e:
        logging.error(f"❌ Failed to write video output: {e}")
        logging.error(traceback.format_exc())
        return "ERROR"


# --- HELPERS USED DURING PROCESSING ---------------------------------------

async def wait_for_blob_ready_async(blob_url: str, container_name: str, max_retries: int = 6, delay: int = 5) -> bool:
    """
    Polls the blob until its size > 0 or until max_retries is exhausted.
    """
    try:
        credential = credential()
        account_name = os.getenv("STORAGE_ACCOUNT_NAME")
        blob_service_client = AsyncBlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=credential
        )
        container_client = blob_service_client.get_container_client(container_name)
        blob_name = blob_url.split("/")[-1]
        blob_client = container_client.get_blob_client(blob_name)

        for _ in range(max_retries):
            props = await blob_client.get_blob_properties()
            if props.size and props.size > 0:
                await blob_service_client.close()
                return True
            await asyncio.sleep(delay)

        await blob_service_client.close()
        return False

    except Exception as e:
        logging.error(f"❌ Error checking blob readiness ({container_name}): {e}")
        return False


async def call_video_indexer(blob_url: str, video_indexer_account: str) -> str:
    """
    Posts to a custom Video Indexer endpoint which in turn calls Azure Video Indexer.
    Expects the endpoint to return either {"videoId": "..."} or {"insightsUrl": "..."}.
    """
    processor_url = os.getenv("VIDEOPROCESS_ENDPOINT")
    account_id = video_indexer_account or os.getenv("VIDEO_INDEXER_ACCOUNT")

    if not processor_url or not account_id:
        logging.error("Missing VIDEOPROCESS_ENDPOINT or VIDEO_INDEXER_ACCOUNT")
        return None

    payload = {
        "blob_url": blob_url,
        "account_id": account_id
    }

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{processor_url}/process", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("videoId") or data.get("insightsUrl")
                else:
                    text = await resp.text()
                    logging.error(f"❌ Video Indexer call failed: {resp.status} - {text}")
                    return None
    except Exception as e:
        logging.error(f"❌ Exception in call_video_indexer: {e}")
        return None