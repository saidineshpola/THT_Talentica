
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from assets.base_classes import BillData

# ------------- Configuration ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_FALLBACK") # Replace or set env var
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_FALLBACK":
    try:
        from dotenv import load_dotenv
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            # To prevent failing entirely if .env is missing or key not set,
            # provide a placeholder. AsyncOpenAI will fail later if this is used.
            logger_config = logging.getLogger("config_logger")
            logger_config.warning(
                "OPENAI_API_KEY environment variable not set and not found in .env. "
                "Using a placeholder. API calls will fail if not replaced."
            )
            OPENAI_API_KEY = "placeholder_api_key_ensure_set_in_env_or_dotenv"
    except ImportError:
        logger_config = logging.getLogger("config_logger")
        logger_config.warning(
            "python-dotenv not installed. OPENAI_API_KEY must be set in the environment. "
            "Using a placeholder. API calls will fail if not replaced."
        )
        OPENAI_API_KEY = "placeholder_api_key_ensure_set_in_env_or_dotenv"
    except Exception as e:
        logger_config = logging.getLogger("config_logger")
        logger_config.error(f"Error loading .env: {e}")
        OPENAI_API_KEY = "placeholder_api_key_ensure_set_in_env_or_dotenv"


# TODO: Update these paths to your local environment
UPLOAD_DIR = Path("C:/Users/user/Desktop/projects/THT_Talentica/assets/bills/uploads")
ASSETS_BASE_DIR = Path("C:/Users/user/Desktop/projects/THT_Talentica/assets/bills") # For processed copies, JSONs, exports

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
DEFAULT_MODEL = "gpt-4o-mini"


logger = logging.getLogger("document_management.text_extractors")

# ------------- Helper Functions ---------------
def setup_directories():
    """Ensures all necessary directories exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_BASE_DIR / "pdfs").mkdir(parents=True, exist_ok=True)
    (ASSETS_BASE_DIR / "images").mkdir(parents=True, exist_ok=True)
    (ASSETS_BASE_DIR / "json").mkdir(parents=True, exist_ok=True)
    (ASSETS_BASE_DIR / "exports").mkdir(parents=True, exist_ok=True)
    logger.info(f"Standard directories ensured. Uploads expected in: {UPLOAD_DIR}")

async def extract_bill_data_with_llm(raw_text: str) -> Dict[str, Any]:
    prompt = f"""
    Extract bill information as JSON matching the Pydantic schema below.
    If the text is empty or contains no bill information, return a JSON object
    with null or empty values for all fields.
    Ensure the 'id', 'original_filename', 'raw_text', and 'q_and_a' fields are NOT part of your JSON output;
    they will be added by the system.

    Schema for your output (excluding id, original_filename, raw_text, q_and_a):
    {{
      "store_name": "Optional[str]",
      "date": "Optional[str]",
      "items": [
        {{
          "name": "str",
          "qty": "Optional[int]",
          "price": "Optional[float]"
        }}
      ],
      "total": "Optional[float]",
      "gst": "Optional[float]"
    }}

    Bill Text (if any):
    ---
    {raw_text if raw_text.strip() else "No text extracted from document."}
    ---
    JSON Output:
    """
    try:
        completion = await aclient.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert bill parsing assistant. Extract information into the provided JSON structure. If no bill data is found, provide empty/null fields for the requested schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        extracted_json_str = completion.choices[0].message.content
        logger.debug(f"OpenAI response for bill parsing: {extracted_json_str}")
        return json.loads(extracted_json_str)
    except Exception as e:
        logger.error(f"LLM bill data extraction failed: {e}", exc_info=True)
        return {
            "store_name": None, "date": None, "items": [],
            "total": None, "gst": None
        }

async def save_bill_json(bill_data: BillData, base_dir: Path = ASSETS_BASE_DIR):
    """Saves the bill data to a JSON file in the 'json' subdirectory of base_dir."""
    json_dir = base_dir / "json"
    json_path = json_dir / f"{bill_data.id}.json"
    try:
        with open(json_path, "w") as f:
            json.dump(bill_data.model_dump(), f, indent=2)
        logger.info(f"Bill data for {bill_data.id} (file: {bill_data.original_filename}) saved to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save bill JSON for {bill_data.id} to {json_path}: {e}", exc_info=True)

async def _internal_process_and_store_bill(
    file_content: bytes,
    original_doc_filename: str, # Filename from UPLOAD_DIR
    text_extraction_func: callable,
    file_type_for_log: str,
    assets_copy_subdir: str, # "pdfs" or "images"
    app_state: Any, # Should be an instance of AppState
    image_format: Optional[str] = None
) -> BillData:
    """
    Internal helper: processes file content, extracts data, stores BillData, and saves a copy of the file.
    """
    logger.info(f"Internal processing for {file_type_for_log} (original: '{original_doc_filename}'), size: {len(file_content)} bytes")

    # Directory where the COPY of the original file will be stored
    copy_storage_dir = ASSETS_BASE_DIR / assets_copy_subdir
    
    # Use only the basename for the copy's filename
    safe_filename_for_copy = Path(original_doc_filename).name 
    file_path_for_copy = copy_storage_dir / safe_filename_for_copy

    try:
        with open(file_path_for_copy, "wb") as f:
            f.write(file_content)
        logger.info(f"Copy of {file_type_for_log} '{original_doc_filename}' saved to {file_path_for_copy}")
    except Exception as e:
        logger.error(f"Failed to save copy of {file_type_for_log} '{original_doc_filename}' to {file_path_for_copy}: {e}", exc_info=True)
        # Continue processing if saving copy failed, as content is in memory

    # Extract text using the appropriate function
    if image_format: # For images
        raw_text = await text_extraction_func(file_content, image_format)
    else: # For PDFs
        raw_text = await text_extraction_func(file_content)

    # Parse data using LLM
    parsed_llm_data = await extract_bill_data_with_llm(raw_text)

    # Create and store BillData object
    bill_id = await app_state.get_next_id()
    bill = BillData(
        id=bill_id,
        original_filename=original_doc_filename, # Store the full original filename as passed
        raw_text=raw_text,
        **parsed_llm_data # Unpack LLM extracted fields
    )

    app_state.bills_store[bill.id] = bill # Store in memory
    await save_bill_json(bill) # Save initial JSON (uses ASSETS_BASE_DIR by default)

    logger.info(f"Successfully parsed and stored {file_type_for_log} as bill ID: {bill.id} (from original file: {original_doc_filename})")
    return bill, app_state
