import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio # For potential asyncio.to_thread if a sync library part is unavoidable

import fitz  # PyMuPDF
import openai # Use the new OpenAI client
from openai import AsyncOpenAI

# MCP Imports
from mcp.server.fastmcp import FastMCP, Context as MCPContext
from mcp.server.fastmcp import Image as MCPImage # For handling image inputs
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
# ------------- Configuration ----------------
OPENAI_API_KEY = "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" #os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
DEFAULT_MODEL = "gpt-4o-mini"

# Setup logging
# Setup logging
log_file_path = Path("logs/document_management.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

# Remove default handlers from root logger if any
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # Log to file
        logging.StreamHandler()             # Log to console
    ]
)
logger = logging.getLogger("document_management.api")

# ------------- Pydantic Schemas (Data Models) ---------------
# These remain the same as they define the structure of your bill data
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
    raw_text: Optional[str] = None # Storing extracted text can be useful

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
        self._lock = asyncio.Lock() # For async-safe ID generation

    async def get_next_id(self) -> str:
        async with self._lock:
            self._next_bill_id_counter += 1
            return f"bill_{self._next_bill_id_counter}"

app_state = AppState()

# ------------- MCP Server Setup ----------------
mcp = FastMCP(
    name="PureBillProcessingMCPServer",
    description="A standalone MCP server for bill parsing from images, querying, and chit-chat.",
    # dependencies=["PyMuPDF", "openai"], # Optional: For mcp tooling
)

# ------------- MCP Tools & Resources ---------------
import fitz  # PyMuPDF

@mcp.tool(
    name="ParsePDFFile",
    description="Upload a PDF file, extract text, and return structured BillData.",
)
async def parse_pdf_file_tool(pdf_file, mcp_ctx):
    """
    MCP Tool to parse a PDF file, extract text, and structure it using an LLM.
    The client is expected to provide the PDF via an MCP File object.
    """
    pdf_bytes = pdf_file.data
    logger.info(f"MCP Tool 'ParsePDFFile' received PDF, size: {len(pdf_bytes)} bytes.")
    mcp_ctx.info(f"Processing PDF file...")

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
            mcp_ctx.info("Warning: No text extracted from the PDF.")

        logger.info(f"Extracted raw text length: {len(raw_text)}")
        mcp_ctx.info(f"Extracted {len(raw_text)} characters of text from PDF.")

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}", exc_info=True)
        mcp_ctx.info(f"Error during PDF text extraction: {str(e)}")
        raise ValueError(f"Text extraction from PDF failed: {str(e)}")

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
    {raw_text if raw_text.strip() else "No text extracted from PDF."}
    ---
    JSON Output:
    """
    try:
        logger.info("Calling OpenAI for structured bill data extraction from PDF.")
        completion = await aclient.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert bill parsing assistant. Extract information into the provided JSON structure. If no bill data is found, provide empty/null fields."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        extracted_json_str = completion.choices[0].message.content
        logger.debug(f"OpenAI response for bill parsing from PDF: {extracted_json_str}")

        parsed_llm_data = json.loads(extracted_json_str)

        bill_id = await app_state.get_next_id()
        bill = BillData(id=bill_id, raw_text=raw_text, **parsed_llm_data)

        app_state.bills_store[bill.id] = bill
        mcp_ctx.info(f"Successfully parsed and stored bill with ID: {bill.id}")
        return bill
    except Exception as e:
        logger.error(f"MCP Tool 'ParsePDFFile' - LLM extraction or data handling failed: {e}", exc_info=True)
        mcp_ctx.info(f"Error during LLM extraction from PDF: {str(e)}")
        raise ValueError(f"LLM extraction or data processing from PDF failed: {str(e)}")


@mcp.tool(
    name="ParseBillImage",
    description="Upload a bill image, perform OCR & extraction, return structured BillData.",
)
async def parse_bill_image_tool(bill_image, mcp_ctx):
    """
    MCP Tool to parse a bill image, extract text, and structure it using an LLM.
    The client is expected to provide the image via an MCP Image object.
    """
    image_bytes = bill_image.data
    image_format = bill_image.format # e.g., 'png', 'jpeg'
    logger.info(f"MCP Tool 'ParseBillImage' received image, format: {image_format}, size: {len(image_bytes)} bytes.")
    mcp_ctx.info(f"Processing image of format {image_format}...") # Example of using MCP context

    raw_text = ""
    try:
        # Use PyMuPDF (fitz) to extract text from the image bytes
        # fitz needs filetype for images when opening from stream
        doc = fitz.open(stream=image_bytes, filetype=image_format)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text += page.get_text("text") + "\n"
        doc.close()
        
        if not raw_text.strip():
            logger.warning("No text could be extracted from the provided image.")
            # Decide how to handle: error, or empty BillData with raw_text?
            # For now, let's proceed and let the LLM decide, or raise an error.
            # raise ValueError("No text extracted from image.") # Option
            mcp_ctx.info("Warning: No text extracted from the image via OCR.")


        logger.info(f"Extracted raw text length: {len(raw_text)}")
        mcp_ctx.info(f"Extracted {len(raw_text)} characters of text from image.")

    except Exception as e:
        logger.error(f"Failed to extract text from image: {e}", exc_info=True)
        mcp_ctx.info(f"Error during image text extraction: {str(e)}")
        # This error should be propagated in a way MCP client understands,
        # often by raising an exception the MCP framework handles.
        raise ValueError(f"OCR or text extraction failed: {str(e)}")

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
    {raw_text if raw_text.strip() else "No text extracted from image."}
    ---
    JSON Output:
    """
    try:
        logger.info("Calling OpenAI for structured bill data extraction.")
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
        
        parsed_llm_data = json.loads(extracted_json_str)
        
        bill_id = await app_state.get_next_id()
        # Create BillData instance, ensuring 'id' and 'raw_text' are included
        bill = BillData(id=bill_id, raw_text=raw_text, **parsed_llm_data)
        
        app_state.bills_store[bill.id] = bill
        mcp_ctx.info(f"Successfully parsed and stored bill with ID: {bill.id}")
        return bill
    except Exception as e:
        logger.error(f"MCP Tool 'ParseBillImage' - LLM extraction or data handling failed: {e}", exc_info=True)
        mcp_ctx.info(f"Error during LLM extraction: {str(e)}")
        raise ValueError(f"LLM extraction or data processing failed: {str(e)}")

@mcp.tool(
    name="QueryParsedBill",
    description="Answers questions based on a previously parsed bill's structured data.",
)
async def query_parsed_bill_tool(bill_id: str, question: str, mcp_ctx: MCPContext) -> QueryResponse:
    if bill_id not in app_state.bills_store:
        logger.warning(f"Query attempt for non-existent bill ID: {bill_id}")
        mcp_ctx.info(f"Bill ID {bill_id} not found for querying.")
        raise ValueError(f"Bill with ID '{bill_id}' not found.")
    
    bill_data = app_state.bills_store[bill_id]
    # Use model_dump_json for Pydantic v2
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
        logger.error(f"MCP Tool 'QueryParsedBill' failed for bill {bill_id}: {e}", exc_info=True)
        mcp_ctx.info(f"Error during bill query for {bill_id}: {str(e)}")
        raise ValueError(f"Query processing failed: {str(e)}")

@mcp.tool(
    name="GeneralChat",
    description="Engages in general chit-chat with the user.",

)
async def general_chat_tool(user_message: str, mcp_ctx: MCPContext) -> ChatResponse:
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
        logger.error(f"MCP Tool 'GeneralChat' failed: {e}", exc_info=True)
        mcp_ctx.info(f"Error during chit-chat: {str(e)}")
        raise ValueError(f"Chit-chat processing failed: {str(e)}")

@mcp.tool(
    name="ExportBillData",
    description="Saves a parsed bill by its ID to a JSON file on the server.",
 
)
async def export_bill_data_tool(bill_id: str,  mcp_ctx: MCPContext, filename: Optional[str] = None) -> Dict[str, str]:
    if bill_id not in app_state.bills_store:
        logger.warning(f"Export attempt for non-existent bill ID: {bill_id}")
        raise ValueError(f"Bill with ID '{bill_id}' not found for export.")
    
    bill_data = app_state.bills_store[bill_id]
    
    export_dir = "exported_bills_mcp" # Server-side directory
    os.makedirs(export_dir, exist_ok=True)
    
    actual_filename = filename or f"bill_{bill_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(export_dir, actual_filename)
    
    try:
        with open(file_path, 'w') as f:
            # Use model_dump for Pydantic v2
            json.dump(bill_data.model_dump(), f, indent=2)
        logger.info(f"Bill {bill_id} exported to {file_path} by MCP tool.")
        mcp_ctx.info(f"Bill {bill_id} exported to {file_path}.")
        return {"message": "Bill exported successfully on the server.", "server_path": file_path}
    except Exception as e:
        logger.error(f"MCP Tool 'ExportBillData' failed for bill {bill_id}: {e}", exc_info=True)
        mcp_ctx.info(f"Error during bill export for {bill_id}: {str(e)}")
        raise ValueError(f"Failed to save bill on server: {str(e)}")

@mcp.resource(
    name="GetBillById",
    uri="bills://{bill_id}",
    description="Retrieves a parsed bill by its ID.",

)
async def get_bill_by_id_resource( bill_id: str,) -> Optional[BillData]:
    logger.info(f"MCP Resource 'GetBillById' called for ID: {bill_id}")
    return app_state.bills_store.get(bill_id)

@mcp.resource(
    name="ListAllBills",
    uri="bills://all", # You can define any URI scheme you like
    description="Lists all parsed bills currently in memory.",
)
async def list_all_bills_resource() -> List[BillData]:
    logger.info("MCP Resource 'ListAllBills' called.")
    # mcp_ctx.info("Accessing resource to list all bills.")
    return list(app_state.bills_store.values())

# TODO: Implement evaluation methods (e.g., as a separate script or tool)
# This would involve comparing LLM extracted fields against human-annotated ground truth.

if __name__ == "__main__":
    # This makes the server runnable with `python your_server_file.py`
    # It will use the default MCP transport (stdio, SSE, or Streamable HTTP depending on MCP defaults)
    # To run with MCP developer tools: `mcp dev your_server_file.py`
    
    # Example of how you might select a transport if needed, e.g., for Streamable HTTP
    # mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
    # For default behavior (often stdio for `mcp dev` or direct run):
    logger.info("Starting pure MCP Bill Processing Server...")
    mcp.run()