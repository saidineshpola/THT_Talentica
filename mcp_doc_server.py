import os
import json
import logging
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import fitz  # PyMuPDF
from openai import AsyncOpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
from fastapi_mcp import FastApiMCP

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
class QueryRequest(BaseModel):
    bill_id: str
    question: str
    
class QueryResponse(BaseModel):
    answer: str

# For Base64 PDF Processing
class Base64PDFRequest(BaseModel):
    base64_pdf: str

# For Bill Export
class ExportRequest(BaseModel):
    bill_id: str
    filename: Optional[str] = None

class ExportResponse(BaseModel):
    message: str
    server_path: str

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

# ------------- FastAPI App Setup ---------------
app = FastAPI(
    title="Bill Processing API",
    description="API for processing and extracting structured data from bills",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ------------- FastAPI Endpoint Definitions (Regular API) ---------------
@app.post("/api/parse-pdf", response_model=BillData, operation_id="parse_pdf_file")
async def parse_pdf_file(pdf_file: UploadFile = File(...)):
    """
    Upload a PDF file to extract and structure bill data.
    """
    try:
        pdf_bytes = await pdf_file.read()
        logger.info(f"Received PDF, size: {len(pdf_bytes)} bytes, filename: {pdf_file.filename}")

        # Extract text from PDF
        raw_text = await extract_text_from_pdf(pdf_bytes)
        
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
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")

@app.post("/api/parse-image", response_model=BillData, operation_id="parse_bill_image")
async def parse_bill_image(image_file: UploadFile = File(...)):
    """
    Upload a bill image to extract and structure bill data.
    """
    try:
        image_bytes = await image_file.read()
        content_type = image_file.content_type.lower() if image_file.content_type else ""
        
        # Extract format from content type (e.g., "image/png" -> "png")
        image_format = content_type.split("/")[1] if "/" in content_type else ""
        
        # Validate format
        if image_format not in ['png', 'jpg', 'jpeg', 'webp', 'tiff', 'bmp']:
            image_format = 'png'  # Default to PNG if format is unrecognized
        
        logger.info(f"Received image, format: {image_format}, size: {len(image_bytes)} bytes")
        
        # Extract text from image
        raw_text = await extract_text_from_image(image_bytes, image_format)
        
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
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


@app.post("/api/query-bill", response_model=QueryResponse, operation_id="query_parsed_bill")
async def query_bill(request: QueryRequest):
    """
    Query information from a previously parsed bill.
    """
    bill_id = request.bill_id
    question = request.question
    
    if bill_id not in app_state.bills_store:
        logger.warning(f"Query attempt for non-existent bill ID: {bill_id}")
        raise HTTPException(status_code=404, detail=f"Bill with ID '{bill_id}' not found.")
    
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
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse, operation_id="general_chat")
async def general_chat(user_message: str = Body(..., embed=True)):
    """
    General conversation capability.
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
        raise HTTPException(status_code=500, detail=f"Chit-chat processing failed: {str(e)}")

@app.post("/api/export-bill", response_model=ExportResponse, operation_id="export_bill_data")
async def export_bill_data(request: ExportRequest):
    """
    Saves a parsed bill by its ID to a JSON file on the server.
    """
    bill_id = request.bill_id
    filename = request.filename
    
    if bill_id not in app_state.bills_store:
        logger.warning(f"Export attempt for non-existent bill ID: {bill_id}")
        raise HTTPException(status_code=404, detail=f"Bill with ID '{bill_id}' not found for export.")
    
    bill_data = app_state.bills_store[bill_id]
    
    export_dir = "exported_bills"  # Server-side directory
    os.makedirs(export_dir, exist_ok=True)
    
    actual_filename = filename or f"bill_{bill_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(export_dir, actual_filename)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(bill_data.model_dump(), f, indent=2)
        logger.info(f"Bill {bill_id} exported to {file_path}.")
        return ExportResponse(
            message="Bill exported successfully on the server.",
            server_path=file_path
        )
    except Exception as e:
        logger.error(f"Bill export failed for bill {bill_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save bill on server: {str(e)}")

@app.get("/api/bill/{bill_id}", response_model=BillData, operation_id="get_bill_by_id")
async def get_bill_by_id(bill_id: str):
    """
    Retrieves a parsed bill by its ID.
    """
    logger.info(f"Getting bill with ID: {bill_id}")
    if bill_id not in app_state.bills_store:
        raise HTTPException(status_code=404, detail=f"Bill with ID '{bill_id}' not found.")
    return app_state.bills_store[bill_id]

@app.get("/api/bills", response_model=List[BillData], operation_id="list_all_bills")
async def list_all_bills():
    """
    Lists all parsed bills currently in memory.
    """
    logger.info("Listing all bills")
    return list(app_state.bills_store.values())

# ------------- FastAPI MCP Integration ---------------
# Create the FastAPI MCP instance
mcp = FastApiMCP(
    app,  
    # base_url="http://api-host:8000",
    name="Bill Processing MCP",  
    description="MCP server for bill parsing, querying, and management",  
    describe_all_responses=True,
   # describe_full_response_schema=True
)

# Mount the MCP server to your FastAPI app
mcp.mount()

# ------------- Main Entrypoint ---------------
if __name__ == "__main__":
    import uvicorn
    # Start the FastAPI server with MCP integration
    logger.info("Starting FastAPI Bill Processing Server with MCP integration...")
    uvicorn.run(app, host="0.0.0.0", port=8000,) # reload=True)