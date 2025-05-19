import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import datetime
import difflib

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
EXAMPLES_DIR = Path("C:/Users/user/Desktop/projects/THT_Talentica/assets/bills/examples") # For few-shot examples
FEEDBACK_DIR = Path("C:/Users/user/Desktop/projects/THT_Talentica/assets/bills/feedback") # For storing corrections

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Model configuration
DEFAULT_MODEL = "gpt-4o-mini"
HIGH_ACCURACY_MODEL = "gpt-4o"  # Use more capable model when accuracy is critical

# Feature flags for enabling/disabling enhancements
ENABLE_FEW_SHOT = False         # Enable few-shot learning
ENABLE_FEEDBACK = False        # Enable feedback-based improvements
ENABLE_HIGH_ACCURACY = False   # Use more powerful model when needed

logger = logging.getLogger("document_management.utils")

# ------------- Helper Functions ---------------
def setup_directories():
    """Ensures all necessary directories exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_BASE_DIR / "pdfs").mkdir(parents=True, exist_ok=True)
    (ASSETS_BASE_DIR / "images").mkdir(parents=True, exist_ok=True)
    (ASSETS_BASE_DIR / "json").mkdir(parents=True, exist_ok=True)
    (ASSETS_BASE_DIR / "exports").mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Standard directories ensured. Uploads expected in: {UPLOAD_DIR}")

async def load_few_shot_examples() -> List[Dict[str, Any]]:
    """Load few-shot examples from the examples directory."""
    examples = []
    
    if not ENABLE_FEW_SHOT:
        logger.debug("Few-shot learning disabled, skipping example loading")
        return examples
        
    try:
        example_files = list(EXAMPLES_DIR.glob("*.json"))
        if not example_files:
            logger.warning(f"No few-shot examples found in {EXAMPLES_DIR}")
            return examples
            
        for file_path in example_files[:3]:  # Limit to 3 examples for prompt size
            try:
                with open(file_path, "r") as f:
                    example = json.load(f)
                    # Validate that it has the necessary fields
                    if "raw_text" in example and any(field in example for field in ["store_name", "date", "items", "total"]):
                        examples.append({
                            "raw_text": example["raw_text"],
                            "extracted": {k: v for k, v in example.items() 
                                        if k not in ["id", "original_filename", "raw_text", "q_and_a"]}
                        })
            except Exception as e:
                logger.error(f"Failed to load example {file_path}: {e}")
                
        logger.info(f"Loaded {len(examples)} few-shot examples")
        return examples
    except Exception as e:
        logger.error(f"Error loading few-shot examples: {e}")
        return examples

def create_few_shot_prompt(examples: List[Dict[str, Any]]) -> str:
    """Create a prompt section with few-shot examples."""
    if not examples:
        return ""
        
    prompt = "\nHere are some examples of bill text and corresponding extracted data:\n\n"
    
    for i, example in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += "Bill Text:\n"
        prompt += f"{example['raw_text'][:500]}... (truncated)\n\n"  # Truncate long texts
        prompt += "Extracted JSON:\n"
        prompt += f"{json.dumps(example['extracted'], indent=2)}\n\n"
        prompt += "---\n\n"
        
    return prompt

async def record_feedback(bill_id: str, field: str, original_value: Any, corrected_value: Any):
    """Record user feedback with proper BillData field validation."""
    if not ENABLE_FEEDBACK:
        return
        
    valid_fields = {
        "store_name", "date", "total", "gst",
        "items"  # Array field
    }
    
    # Validate field path
    field_parts = field.split(".")
    base_field = field_parts[0].split("[")[0] if "[" in field_parts[0] else field_parts[0]
    
    if base_field not in valid_fields:
        logger.error(f"Invalid field {field} for feedback recording")
        return
        
    feedback_file = FEEDBACK_DIR / f"{bill_id}_feedback.json"
    
    feedback_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "field": field,
        "original_value": original_value,
        "corrected_value": corrected_value,
        "field_type": "array_item" if base_field == "items" else "simple"
    }
    
    try:
        # Load or create feedback history
        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                feedback_history = json.load(f)
        else:
            feedback_history = {"bill_id": bill_id, "feedback": []}
            
        feedback_history["feedback"].append(feedback_entry)
        
        with open(feedback_file, "w") as f:
            json.dump(feedback_history, f, indent=2)
            
        logger.info(f"Recorded feedback for bill {bill_id}, field {field}")
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")

async def consider_adding_as_example(bill_id: str):
    """Check if this bill with its corrections would make a good example."""
    feedback_file = FEEDBACK_DIR / f"{bill_id}_feedback.json"
    bill_file = ASSETS_BASE_DIR / "json" / f"{bill_id}.json"
    
    if not feedback_file.exists() or not bill_file.exists():
        return
        
    try:
        # Load feedback and bill data
        with open(feedback_file, "r") as f:
            feedback = json.load(f)
        with open(bill_file, "r") as f:
            bill_data = json.load(f)
            
        # Check if we have at least 3 corrected fields (suggesting this is a good example)
        if len(feedback["feedback"]) >= 3:
            # Create a corrected version with feedback applied
            example_path = EXAMPLES_DIR / f"{bill_id}_example.json"
            
            # Apply all corrections to bill data
            corrected_bill = bill_data.copy()
            for fb in feedback["feedback"]:
                field_path = fb["field"].split(".")
                
                # Handle nested fields like items[0].name
                if len(field_path) == 1:
                    corrected_bill[field_path[0]] = fb["corrected_value"]
                else:
                    # This is simplified - would need more complex logic for deeply nested paths
                    parent, key = field_path[0], field_path[1]
                    if parent in corrected_bill:
                        corrected_bill[parent][key] = fb["corrected_value"]
            
            # Save as example
            with open(example_path, "w") as f:
                json.dump(corrected_bill, f, indent=2)
                
            logger.info(f"Added bill {bill_id} as an example for few-shot learning")
            
    except Exception as e:
        logger.error(f"Failed to consider bill {bill_id} as example: {e}")

def calculate_confidence_score(raw_text: str) -> float:
    """Calculate confidence score to determine if high accuracy mode is needed."""
    # Simple heuristic: low confidence if text is short or has many non-alphanumeric chars
    if not raw_text or len(raw_text) < 50:
        return 0.2
    
    alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in raw_text) / len(raw_text)
    numeric_count = sum(c.isdigit() for c in raw_text)
    numeric_ratio = numeric_count / len(raw_text)
    
    # Typical bill should have decent amount of numbers (prices, quantities)
    if numeric_ratio < 0.05:  # Less than 5% numerics could be an issue
        return 0.4
        
    # Very noisy text might need higher accuracy model
    if alphanumeric_ratio < 0.7:  # More than 30% special characters
        return 0.5
        
    # Reasonable length text with good alphanumeric ratio is probably fine
    return 0.8

async def extract_bill_data_with_llm(raw_text: str) -> Dict[str, Any]:
    """Extract bill data using LLM with few-shot learning and accuracy controls."""
    confidence_score = calculate_confidence_score(raw_text)
    logger.debug(f"Confidence score for extraction: {confidence_score}")
    
    # Determine if we need high accuracy model
    use_high_accuracy = ENABLE_HIGH_ACCURACY and confidence_score < 0.6
    model = HIGH_ACCURACY_MODEL if use_high_accuracy else DEFAULT_MODEL
    logger.info(f"Using model {model} for bill extraction (confidence: {confidence_score})")
    
    # Load few-shot examples if enabled
    examples = await load_few_shot_examples() if ENABLE_FEW_SHOT else []
    few_shot_content = create_few_shot_prompt(examples)
    # TODO:
    # Specific prompts and examples for different types of bills can be added here

    prompt = f"""
    Extract bill information as JSON matching this exact Pydantic schema:

    class Item(BaseModel):
        name: str
        qty: Optional[int]
        price: Optional[float]

    class BillData(BaseModel):
        store_name: Optional[str]
        date: Optional[str]  # Format: YYYY-MM-DD
        items: List[Item]
        total: Optional[float]
        gst: Optional[float]

    Rules:
    1. Date must be in YYYY-MM-DD format
    2. All numeric values (qty, price, total, gst) must be numbers, not strings
    3. Only include items that are clearly product/service entries
    4. Set fields to null if information is not clearly present
    5. Extract the complete store name from the header
    6. Include GST/tax amount only if explicitly mentioned

    {few_shot_content}

    Bill Text:
    {raw_text if raw_text.strip() else "No text provided"}

    Return ONLY the JSON object matching the schema above.
    """
    
    system_prompt = """You are an expert bill parsing assistant. Extract information exactly matching the specified schema.
    Be precise and only include fields that are explicitly present in the bill."""
    
    try:
        completion = await aclient.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for more consistent outputs
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

async def update_bill_field(bill_id: str, field_path: str, new_value: Any) -> Tuple[bool, Optional[BillData]]:
    """Update a specific field in the bill data and record feedback."""
    json_path = ASSETS_BASE_DIR / "json" / f"{bill_id}.json"
    
    try:
        # Load current bill data
        with open(json_path, "r") as f:
            bill_data = json.load(f)
            
        # Get current value for feedback
        field_parts = field_path.split(".")
        current_value = None
        
        # Handle nested fields (e.g., items[0].name)
        if len(field_parts) == 1:
            current_value = bill_data.get(field_parts[0])
            # Update the field
            bill_data[field_parts[0]] = new_value
        else:
            # Simple handling for first-level array items like items[0].name
            # A more robust solution would need recursive traversal for deeply nested paths
            parent, key = field_parts[0], field_parts[1]
            if "[" in parent and "]" in parent:
                # Handle array indexing like items[0]
                array_name = parent.split("[")[0]
                index = int(parent.split("[")[1].split("]")[0])
                
                if array_name in bill_data and len(bill_data[array_name]) > index:
                    current_value = bill_data[array_name][index].get(key)
                    bill_data[array_name][index][key] = new_value
            elif parent in bill_data:
                current_value = bill_data[parent].get(key)
                bill_data[parent][key] = new_value
        
        # Save updated bill data
        with open(json_path, "w") as f:
            json.dump(bill_data, f, indent=2)
            
        # Record feedback for learning # Disabled for now
        if ENABLE_FEEDBACK:
            await record_feedback(bill_id, field_path, current_value, new_value)
        
        # Return updated bill data as BillData object
        updated_bill = BillData(**bill_data)
        
        logger.info(f"Updated field {field_path} for bill {bill_id}")
        return True, updated_bill
    
    except Exception as e:
        logger.error(f"Failed to update field {field_path} for bill {bill_id}: {e}", exc_info=True)
        return False, None

async def verify_extraction_accuracy(bill_data: BillData) -> Dict[str, Any]:
    """Verify extracted data against the original text."""
    if not ENABLE_HIGH_ACCURACY:
        return {"verified": False, "message": "High accuracy verification disabled"}
        
    verification_prompt = f"""
    Verify this extracted bill data against the original text:

    EXTRACTED DATA:
    {json.dumps({
        "store_name": bill_data.store_name,
        "date": bill_data.date,
        "items": [item.model_dump() for item in bill_data.items],
        "total": bill_data.total,
        "gst": bill_data.gst
    }, indent=2)}

    ORIGINAL TEXT:
    {bill_data.raw_text[:2000]}

    Verify each field:
    1. store_name: Must match header of bill
    2. date: Must be in YYYY-MM-DD format
    3. items: Each must have name, optional qty and price
    4. total: Must match final amount including taxes
    5. gst: Must only include actual GST/tax amount

    
    Return JSON with:
    {{
        "accuracy_score": float between 0.0 and 1.0,
        "issues": list of strings describing specific problems,
        "corrections": dictionary of fields needing correction
    }}
    """
    
    try:
        completion = await aclient.chat.completions.create(
            model=HIGH_ACCURACY_MODEL,  # Use higher capability model for verification
            messages=[
                {"role": "system", "content": "You are an expert bill verification system that checks extraction accuracy."},
                {"role": "user", "content": verification_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1  # Lower temperature for more consistent verification
        )
        
        verification_result = json.loads(completion.choices[0].message.content)
        
        # If verification found issues with high confidence, auto-apply corrections
        if "accuracy_score" in verification_result and verification_result["accuracy_score"] < 0.8:
            if "corrections" in verification_result and verification_result["corrections"]:
                logger.info(f"Auto-applying corrections for bill {bill_data.id} based on verification")
                
                # Apply each correction
                for field, value in verification_result["corrections"].items():
                    await update_bill_field(bill_data.id, field, value)
                    
        return verification_result
    
    except Exception as e:
        logger.error(f"Bill verification failed: {e}", exc_info=True)
        return {"verified": False, "error": str(e)}

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

async def answer_bill_query(bill_id: str, query: str) -> str:
    """
    Answer user queries about a specific bill using the extracted data.
    """
    json_path = ASSETS_BASE_DIR / "json" / f"{bill_id}.json"
    
    try:
        # Load current bill data
        with open(json_path, "r") as f:
            bill_data = json.load(f)
            
        query_prompt = f"""
        This is the data extracted from a bill:
        ```json
        {json.dumps(bill_data, indent=2)}
        ```
        
        The user is asking this question about the bill:
        "{query}"
        
        Answer the question based only on the information in the bill data. Be precise and to the point.
        If the question cannot be answered based on the available data, clearly state that.
        """
        
        completion = await aclient.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about bill data."},
                {"role": "user", "content": query_prompt}
            ],
            temperature=0.7  # Allow some creativity in responses
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Failed to answer query for bill {bill_id}: {e}", exc_info=True)
        return f"Sorry, I couldn't process your question about bill {bill_id} due to an error."

async def _internal_process_and_store_bill(
    file_content: bytes,
    original_doc_filename: str, # Filename from UPLOAD_DIR
    text_extraction_func: callable,
    file_type_for_log: str,
    assets_copy_subdir: str, # "pdfs" or "images"
    app_state: Any, # Should be an instance of AppState
    image_format: Optional[str] = None
) -> Tuple[BillData, Any]:
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

    # Parse data using LLM with enhanced extraction
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
    
    # Run verification if high accuracy is enabled
    if ENABLE_HIGH_ACCURACY:
        verification_result = await verify_extraction_accuracy(bill)
        logger.info(f"Bill {bill.id} verification result: {verification_result}")

    logger.info(f"Successfully parsed and stored {file_type_for_log} as bill ID: {bill.id} (from original file: {original_doc_filename})")
    return bill, app_state