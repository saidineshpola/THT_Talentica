# import httpx
# import logging
# from pathlib import Path
# from mcp.server.fastmcp import FastMCP

# # Set up logging
# log_file_path = Path(__file__).parent / "logs" / "app.log"
# log_file_path.parent.mkdir(exist_ok=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file_path),  # Log to file
#         logging.StreamHandler()              # Log to console
#     ]
# )
# logger = logging.getLogger(__name__)

# mcp = FastMCP("My App")

# @mcp.tool()
# def calculate_bmi(weight_kg: float, height_m: float) -> float:
#     """Calculate BMI given weight in kg and height in meters"""
#     logger.info(f"Calculating BMI for weight={weight_kg}kg, height={height_m}m")
#     bmi = weight_kg / (height_m**2)
#     logger.info(f"Calculated BMI: {bmi}")
#     return bmi

# @mcp.tool()
# async def fetch_weather(city: str) -> str:
#     """Fetch current weather for a city"""
#     logger.info(f"Fetching weather for city: {city}")
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.get(f"https://api.weather.com/{city}")
#             logger.info(f"Weather data fetched successfully for {city}")
#             return response.text
#     except Exception as e:
#         logger.error(f"Error fetching weather for {city}: {str(e)}")
#         raise

import os
import json
import logging
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import pymupdf as fitz 
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio
from mcp.server.fastmcp import FastMCP

# ------------- Configuration ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
DEFAULT_MODEL = "gpt-4o-mini"

# Setup logging
log_file_path = Path("logs/document_management.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

# Remove default handlers from root logger if any
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),  # Log to file
        logging.StreamHandler()              # Log to console
    ]
)
logger = logging.getLogger("document_management.api")

# ------------- Pydantic Schemas (Data Models) ---------------
class Item(BaseModel):
    name: str
    qty: Optional[int] = None
    price: Optional[float] = None

class BillData(BaseModel):
    id: str
    store_name: Optional[str] = None
    date: Optional[str] = None
    items: List[Item] = Field(default_factory=list)
    total: Optional[float] = None
    gst: Optional[float] = None
    raw_text: Optional[str] = None  # Storing extracted text can be useful

# For Chit-Chat
class ChatResponse(BaseModel):
    reply: str

# For Querying
class QueryResponse(BaseModel):
    answer: str

# Application State (simplified in-memory store)
class AppState:
    def __init__(self):
        self.bills_store: Dict[str, BillData] = {}
        self._next_bill_id_counter: int = 0
        self._lock = asyncio.Lock()  # For async-safe ID generation

    async def get_next_id(self) -> str:
        async with self._lock:
            self._next_bill_id_counter += 1
            return f"bill_{self._next_bill_id_counter}"

app_state = AppState()

# ------------- Helper Functions ---------------
async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    raw_text = ""
    try:
        # Use PyMuPDF (fitz) to extract text from the PDF bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text += page.get_text("text") + "\n"
        doc.close()

        if not raw_text.strip():
            logger.warning("No text could be extracted from the provided PDF.")
        
        logger.info(f"Extracted raw text length: {len(raw_text)}")
        return raw_text

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}", exc_info=True)
        raise ValueError(f"Text extraction from PDF failed: {str(e)}")

async def extract_text_from_image(image_bytes: bytes, image_format: str) -> str:
    """Extract text from image bytes using PyMuPDF."""
    raw_text = ""
    try:
        # For better OCR results with PyMuPDF
        doc = fitz.open(stream=image_bytes, filetype=image_format)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get text directly - might work for some images with embedded text
            text_from_page = page.get_text("text")
            
            # If direct text extraction fails, try OCR-based approach
            if not text_from_page.strip():
                # PyMuPDF doesn't have direct OCR, but we'll extract what we can
                logger.info("Direct text extraction yielded no results, attempting alternative methods...")
            
            raw_text += text_from_page + "\n"
        
        doc.close()
        
        if not raw_text.strip():
            logger.warning("No text could be extracted from the provided image.")
        
        logger.info(f"Extracted raw text length: {len(raw_text)}")
        return raw_text
        
    except Exception as e:
        logger.error(f"Failed to extract text from image: {e}", exc_info=True)
        # Return empty string instead of raising exception to allow LLM to still attempt interpretation
        return ""

async def extract_bill_data_with_llm(raw_text: str) -> Dict[str, Any]:
    """Use LLM to extract structured bill data from raw text."""
    prompt = f"""
    Extract bill information as JSON matching the Pydantic schema below.
    If the text is empty or contains no bill information, return a JSON object
    with null or empty values for all fields except 'id' and 'raw_text'.
    Ensure the 'id' field is NOT part of your JSON output; it will be added by the system.
    Ensure 'raw_text' is NOT part of your JSON output; it will be added by the system.

    Schema for your output (excluding id and raw_text):
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
    
    completion = await aclient.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert bill parsing assistant. Extract information into the provided JSON structure. If no bill data is found, provide empty/null fields."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    extracted_json_str = completion.choices[0].message.content
    logger.debug(f"OpenAI response for bill parsing: {extracted_json_str}")
    
    return json.loads(extracted_json_str)

# ------------- FastMCP Setup ---------------
mcp = FastMCP("Bill Processing MCP", dependencies=[
    # Add any dependencies here if needed"
    "pymupdf",
    "openai",
    "pydantic",
    "asyncio",
    
    ])

@mcp.tool()
async def parse_pdf_file(pdf_content: bytes, filename: str) -> BillData:
    """
    Parse a PDF file to extract and structure bill data.
    
    Args:
        pdf_content: The binary content of the PDF file
        filename: Name of the uploaded file
        
    Returns:
        Structured bill data extracted from the PDF
    """
    try:
        logger.info(f"Received PDF, size: {len(pdf_content)} bytes, filename: {filename}")

        # Extract text from PDF
        raw_text = await extract_text_from_pdf(pdf_content)
        
        # Extract structured data from text using LLM
        logger.info("Calling OpenAI for structured bill data extraction from PDF.")
        parsed_llm_data = await extract_bill_data_with_llm(raw_text)
        
        # Store the parsed bill
        bill_id = await app_state.get_next_id()
        bill = BillData(id=bill_id, raw_text=raw_text, **parsed_llm_data)
        
        app_state.bills_store[bill.id] = bill
        logger.info(f"Successfully parsed and stored bill with ID: {bill.id}")
        return bill
    
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}", exc_info=True)
        raise ValueError(f"PDF processing failed: {str(e)}")

@mcp.tool()
async def parse_bill_image(image_content: bytes, filename: str, image_format: str = "png") -> BillData:
    """
    Parse a bill image to extract and structure bill data.
    
    Args:
        image_content: The binary content of the image file
        filename: Name of the uploaded file
        image_format: Format of the image (png, jpg, jpeg, etc.)
        
    Returns:
        Structured bill data extracted from the image
    """
    try:
        # Validate format
        if image_format not in ['png', 'jpg', 'jpeg', 'webp', 'tiff', 'bmp']:
            image_format = 'png'  # Default to PNG if format is unrecognized
        
        logger.info(f"Received image, format: {image_format}, size: {len(image_content)} bytes")
        
        # Extract text from image
        raw_text = await extract_text_from_image(image_content, image_format)
        
        # Extract structured data using LLM
        logger.info("Calling OpenAI for structured bill data extraction from image.")
        parsed_llm_data = await extract_bill_data_with_llm(raw_text)
        
        # Store the parsed bill
        bill_id = await app_state.get_next_id()
        bill = BillData(id=bill_id, raw_text=raw_text, **parsed_llm_data)
        
        app_state.bills_store[bill.id] = bill
        logger.info(f"Successfully parsed and stored bill with ID: {bill.id}")
        return bill
        
    except Exception as e:
        logger.error(f"Image parsing failed: {e}", exc_info=True)
        raise ValueError(f"Image processing failed: {str(e)}")

@mcp.tool()
async def query_bill(bill_id: str, question: str) -> QueryResponse:
    """
    Query information from a previously parsed bill.
    
    Args:
        bill_id: The ID of the bill to query
        question: The question to ask about the bill
        
    Returns:
        An answer to the question based on the bill data
    """
    if bill_id not in app_state.bills_store:
        logger.warning(f"Query attempt for non-existent bill ID: {bill_id}")
        raise ValueError(f"Bill with ID '{bill_id}' not found.")
    
    bill_data = app_state.bills_store[bill_id]
    bill_json = bill_data.model_dump_json(indent=2)

    prompt = f"""
    You are a helpful assistant. Given the following bill data in JSON format,
    please answer the user's question.

    Bill Data:
    {bill_json}

    Question: {question}

    Answer:
    """
    try:
        logger.info(f"Querying bill {bill_id} with question: '{question}'")
        completion = await aclient.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You answer questions based on provided JSON bill data."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message.content
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Bill query failed for bill {bill_id}: {e}", exc_info=True)
        raise ValueError(f"Query processing failed: {str(e)}")

@mcp.tool()
async def general_chat(user_message: str) -> ChatResponse:
    """
    Provide general conversation capability.
    
    Args:
        user_message: The message from the user
        
    Returns:
        A response to the user's message
    """
    try:
        logger.info(f"Chit-chat message: '{user_message}'")
        completion = await aclient.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a friendly conversational assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        reply = completion.choices[0].message.content
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"Chit-chat failed: {e}", exc_info=True)
        raise ValueError(f"Chit-chat processing failed: {str(e)}")

@mcp.tool()
async def export_bill_data(bill_id: str, filename: Optional[str] = None) -> Dict[str, str]:
    """
    Save a parsed bill by its ID to a JSON file.
    
    Args:
        bill_id: The ID of the bill to export
        filename: Optional custom filename for the exported bill
        
    Returns:
        Information about the exported bill file
    """
    if bill_id not in app_state.bills_store:
        logger.warning(f"Export attempt for non-existent bill ID: {bill_id}")
        raise ValueError(f"Bill with ID '{bill_id}' not found for export.")
    
    bill_data = app_state.bills_store[bill_id]
    
    export_dir = "exported_bills"  # Server-side directory
    os.makedirs(export_dir, exist_ok=True)
    
    actual_filename = filename or f"bill_{bill_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(export_dir, actual_filename)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(bill_data.model_dump(), f, indent=2)
        logger.info(f"Bill {bill_id} exported to {file_path}.")
        return {
            "message": "Bill exported successfully on the server.",
            "server_path": file_path
        }
    except Exception as e:
        logger.error(f"Bill export failed for bill {bill_id}: {e}", exc_info=True)
        raise ValueError(f"Failed to save bill on server: {str(e)}")

@mcp.tool()
async def get_bill_by_id(bill_id: str) -> BillData:
    """
    Retrieve a parsed bill by its ID.
    
    Args:
        bill_id: The ID of the bill to retrieve
        
    Returns:
        The bill data associated with the ID
    """
    logger.info(f"Getting bill with ID: {bill_id}")
    if bill_id not in app_state.bills_store:
        raise ValueError(f"Bill with ID '{bill_id}' not found.")
    return app_state.bills_store[bill_id]

@mcp.tool()
async def list_all_bills() -> List[BillData]:
    """
    List all parsed bills currently in memory.
    
    Returns:
        A list of all stored bill data
    """
    logger.info("Listing all bills")
    return list(app_state.bills_store.values())