import os
import json
import logging
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import pymupdf as fitz
import asyncio
from mcp.server.fastmcp import FastMCP

from assets.base_classes import AppState, BillData, ChatResponse, QAPair, QueryResponse
from text_extractors import extract_text_from_image, extract_text_from_pdf
from utils import ASSETS_BASE_DIR, DEFAULT_MODEL, UPLOAD_DIR, _internal_process_and_store_bill, save_bill_json, setup_directories, aclient




# Setup logging
log_file_path = Path("logs/document_management.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# Remove default handlers from root logger if any
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("document_management.api")
app_state = AppState()

# ------------- Directory Setup ---------------



setup_directories() # Call at import time


# ------------- FastMCP Setup ---------------
mcp = FastMCP("Bill Processing MCP", dependencies=[
    "pymupdf", "openai", "pydantic", "asyncio", "python-dotenv" 
])

# ------------- Resources (URL-style endpoints) ---------------
@mcp.resource("bill://{bill_id}")
async def get_bill_by_id_resource(bill_id: str) -> BillData:
    """Get a specific bill by its internal ID"""
    logger.info(f"Getting bill with internal ID: {bill_id}")
    bill = await app_state.find_bill(bill_id) # find_bill can also find by original_filename if ID isn't matched
    if not bill:
        raise ValueError(f"Bill with ID '{bill_id}' not found.")
    return bill

# ------------- Tools (Function-style endpoints) ---------------

@mcp.tool()
async def discover_available_bills() -> List[Dict[str, str]]:
    """
    Scans the designated uploads directory and lists available bill files.
    This is the first step for the user to see what can be processed.
    """
    logger.info(f"Scanning for bills in: {UPLOAD_DIR}")
    available_files = []
    if not UPLOAD_DIR.exists():
        logger.warning(f"Upload directory {UPLOAD_DIR} does not exist.")
        return [{"message": f"Upload directory {UPLOAD_DIR} not found."}]

    for item in UPLOAD_DIR.iterdir():
        if item.is_file() and item.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            # Check if already processed (in memory) to provide more info, but list all files
            is_processed_in_session = await app_state.find_bill(item.name) is not None
            status = " (processed in session)" if is_processed_in_session else ""
            available_files.append({
                "filename": item.name,
                "path": str(item),
                "status": status
            })
    
    if not available_files:
        return [{"message": "No bill files found in the uploads directory."}]
    logger.info(f"Found {len(available_files)} potential bill files in {UPLOAD_DIR}.")
    return available_files

@mcp.tool()
async def process_selected_bill_and_ask(original_filename: str, question: str) -> QueryResponse:
    """
    Processes a selected bill file from the uploads directory (if not already processed in this session)
    and then answers a question about it. Stores/updates the bill data.
    'original_filename' is the name of the file as listed by 'discover_available_bills'.
    """
    logger.info(f"Request to process '{original_filename}' from uploads and answer question: '{question}'")
    global app_state
    # 1. Check if bill is already processed and in memory (AppState)
    bill = await app_state.find_bill(original_filename) 

    if not bill:
        logger.info(f"Bill '{original_filename}' not found in memory. Attempting to load and process from {UPLOAD_DIR}.")
        file_path_in_uploads = UPLOAD_DIR / original_filename
        
        if not file_path_in_uploads.exists() or not file_path_in_uploads.is_file():
            logger.error(f"File '{original_filename}' not found in upload directory: {file_path_in_uploads}")
            raise ValueError(f"File '{original_filename}' not found in the designated uploads directory.")

        try:
            with open(file_path_in_uploads, "rb") as f:
                file_content = f.read()
        except Exception as e:
            logger.error(f"Error reading file '{file_path_in_uploads}': {e}", exc_info=True)
            raise ValueError(f"Could not read file '{original_filename}': {str(e)}")

        file_ext = Path(original_filename).suffix.lower()
        
        if file_ext == ".pdf":
            bill, app_state = await _internal_process_and_store_bill(
                file_content=file_content,
                original_doc_filename=original_filename,
                text_extraction_func=extract_text_from_pdf,
                file_type_for_log="PDF",
                assets_copy_subdir="pdfs",
                app_state=app_state,
            )
        elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"]: # Added gif
            bill, app_state = await _internal_process_and_store_bill(
                file_content=file_content,
                original_doc_filename=original_filename,
                text_extraction_func=extract_text_from_image,
                file_type_for_log="Image",
                assets_copy_subdir="images",
                image_format=file_ext.strip('.'),
                app_state=app_state,
            )
        else:
            logger.error(f"Unsupported file type for '{original_filename}': {file_ext}")
            raise ValueError(f"Unsupported file type: {file_ext}. Cannot process '{original_filename}'.")
        
        logger.info(f"Newly processed '{original_filename}' assigned bill ID: {bill.id}")
    else:
        logger.info(f"Bill '{original_filename}' (ID: {bill.id}) found in memory cache.")

    # 2. Now that we have the bill (either newly processed or from cache), ask the question.
    # The ask_question_and_save tool handles the LLM call for Q&A, and saving it.
    logger.info(f"Proceeding to answer question for bill '{original_filename}' (ID: {bill.id}).")
    return await ask_question_and_save(bill_identifier=bill.id, question=question)


@mcp.tool()
async def list_processed_bills() -> List[Dict[str, Optional[str]]]:
    """
    List all bills currently processed and stored in memory for this session.
    """
    logger.info("Listing all in-memory processed bills.")
    if not app_state.bills_store:
        return [{"message": "No bills processed in this session yet."}]
    return [
        {"id": bill.id, "original_filename": bill.original_filename, "store_name": bill.store_name, "date": bill.date}
        for bill in app_state.bills_store.values()
    ]

# These tools below can remain for direct content parsing if ever needed,
# but the main flow now uses discover_available_bills and process_selected_bill_and_ask.
@mcp.tool()
async def parse_pdf_bill(pdf_content: bytes, filename: str) -> BillData:
    """
    (Advanced Use) Parse a PDF file from bytes to extract bill data, store it, and save the PDF.
    'filename' is a descriptive name for this content.
    """
    logger.info(f"Direct PDF content parsing requested for filename: {filename}")
    try:
        # This will save a copy to assets/bills/pdfs/filename
        bill, app_state = await _internal_process_and_store_bill(
            file_content=pdf_content,
            original_doc_filename=filename, # Treat passed filename as the key identifier
            text_extraction_func=extract_text_from_pdf,
            file_type_for_log="PDF (direct content)",
            assets_copy_subdir="pdfs",
            app_state=app_state,
        )
        return bill
    except Exception as e:
        logger.error(f"Direct PDF parsing tool failed for '{filename}': {e}", exc_info=True)
        raise ValueError(f"PDF processing from content failed for '{filename}': {str(e)}")

@mcp.tool()
async def parse_image_bill(image_content: bytes, filename: str, image_format: str = "png") -> BillData:
    """
    (Advanced Use) Parse an image file from bytes (e.g., PNG, JPG) to extract bill data.
    'filename' is a descriptive name. 'image_format' (e.g., 'png') is crucial.
    """
    logger.info(f"Direct image content parsing requested for filename: {filename}, format: {image_format}")
    supported_formats = ["png", "jpeg", "jpg", "bmp", "tiff", "gif"]
    global app_state
    if image_format.lower() not in supported_formats:
        raise ValueError(f"Unsupported image format: {image_format}. Supported: {', '.join(supported_formats)}")
    try:
         # This will save a copy to assets/bills/images/filename
        bill, app_state = await _internal_process_and_store_bill(
            file_content=image_content,
            original_doc_filename=filename, # Treat passed filename as the key identifier
            text_extraction_func=extract_text_from_image,
            file_type_for_log="Image (direct content)",
            assets_copy_subdir="images",
            image_format=image_format,
            app_state=app_state,
        )
        return bill
    except Exception as e:
        logger.error(f"Direct image parsing tool failed for '{filename}': {e}", exc_info=True)
        raise ValueError(f"Image processing from content failed for '{filename}': {str(e)}")

@mcp.tool()
async def ask_question_and_save(bill_identifier: str, question: str) -> QueryResponse:
    """
    Ask a question about a specific bill (identified by its ID or original_filename from uploads).
    The question and answer will be saved to the bill's data and its JSON file.
    """
    logger.info(f"Received question: '{question}' for bill identifier: '{bill_identifier}'")

    bill = await app_state.find_bill(bill_identifier)
    if not bill:
        logger.warning(f"Query attempt for non-existent bill identifier: {bill_identifier}")
        raise ValueError(f"Bill with identifier '{bill_identifier}' not found. Ensure it has been processed first.")

    bill_context_for_llm = f"""
    You are answering a question about a specific bill. Here is the extracted raw text from the bill:
    ---BEGIN RAW TEXT---
    {bill.raw_text if bill.raw_text and bill.raw_text.strip() else "No raw text available for this bill."}
    ---END RAW TEXT---

    Here is the structured data already parsed from the bill (if available):
    ---BEGIN STRUCTURED DATA---
    Store Name: {bill.store_name}
    Date: {bill.date}
    Total: {bill.total}
    GST: {bill.gst}
    Items: {json.dumps([item.model_dump() for item in bill.items], indent=2)}
    ---END STRUCTURED DATA---

    Based on all the information above (prioritizing raw text if there are discrepancies), please answer the following question.
    Be concise and directly answer the question.
    """
    try:
        completion = await aclient.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": bill_context_for_llm},
                {"role": "user", "content": f"Question: {question}"}
            ]
        )
        answer = completion.choices[0].message.content
        logger.info(f"LLM generated answer for bill {bill.id}: '{answer}'")

        bill.q_and_a.append(QAPair(question=question, answer=answer))
        app_state.bills_store[bill.id] = bill # Update in-memory store
        await save_bill_json(bill) # Re-save the updated bill data to JSON (uses ASSETS_BASE_DIR)

        return QueryResponse(answer=answer, bill_id=bill.id)
    except Exception as e:
        logger.error(f"Question processing failed for bill {bill.id} (identifier: {bill_identifier}): {e}", exc_info=True)
        raise ValueError(f"Query processing failed: {str(e)}")

@mcp.tool()
async def export_bill_data(bill_identifier: str, export_filename: Optional[str] = None) -> Dict[str, str]:
    """Export a bill's data (including Q&A) to a JSON file in the 'exports' directory."""
    bill = await app_state.find_bill(bill_identifier)
    if not bill:
        raise ValueError(f"Bill with identifier '{bill_identifier}' not found for export.")

    export_dir = ASSETS_BASE_DIR / "exports"
    actual_filename_stem = Path(bill.original_filename or 'export').stem
    actual_filename = export_filename or f"{bill.id}_{actual_filename_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = export_dir / actual_filename

    try:
        with open(file_path, 'w') as f:
            json.dump(bill.model_dump(), f, indent=2)
        logger.info(f"Bill {bill.id} (from {bill.original_filename}) exported to {file_path}")
        return {"message": "Bill exported successfully", "path": str(file_path)}
    except Exception as e:
        logger.error(f"Bill export failed for {bill.id}: {e}", exc_info=True)
        raise ValueError(f"Export failed for bill {bill.id}: {str(e)}")

@mcp.tool()
async def general_chat(message: str) -> ChatResponse:
    """Handle general conversation with the user, not specific to any bill."""
    try:
        logger.info(f"Processing general chat message: '{message}'")
        completion = await aclient.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a friendly conversational assistant for a bill management application."},
                {"role": "user", "content": message}
            ]
        )
        reply = completion.choices[0].message.content
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"General chat failed: {e}", exc_info=True)
        raise ValueError(f"Chat processing failed: {str(e)}")

# Example of how to run (for testing, typically MCP server handles this)
async def main_test_new_flow():
    print("MCP Bill Management System (New Flow Demo)")
    setup_directories() # Ensure directories are ready

    # --- Create dummy files in UPLOAD_DIR for demonstration ---
    dummy_pdf_filename = "sample_bill_1.pdf"
    dummy_img_filename = "sample_receipt_A.png"
    
    # Dummy PDF content
    dummy_pdf_content_str = """
    Store: My Corner Shop
    Date: 2024-01-15
    Items:
    - Bread  1   2.50
    - Milk   2   1.75
    Subtotal: 6.00
    Tax (GST 10%): 0.60
    Total: 6.60
    Thank you!
    """
    # Create a simple text PDF for testing (PyMuPDF can generate one)
    try:
        pdf_doc = fitz.open()
        page = pdf_doc.new_page()
        page.insert_text((72, 72), dummy_pdf_content_str, fontsize=11)
        pdf_bytes = pdf_doc.tobytes()
        with open(UPLOAD_DIR / dummy_pdf_filename, "wb") as f:
            f.write(pdf_bytes)
        print(f"Created dummy PDF: {UPLOAD_DIR / dummy_pdf_filename}")
        pdf_doc.close()
    except Exception as e:
        print(f"Could not create dummy PDF: {e}")

    # Dummy Image content (base64 of a tiny PNG) + some text appended for LLM to find
    # For real test, use an actual image with text. This is a placeholder.
    dummy_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    dummy_image_content_bytes = base64.b64decode(dummy_png_b64)
    # PyMuPDF can't easily add text to an arbitrary image like this for the *source file*,
    # so we'll rely on LLM's ability to process what it gets or the fact that `extract_text_from_image` might return ""
    # For the purpose of this test, the LLM prompt for extract_bill_data_with_llm will get "No text extracted..." if image text extraction fails.
    # To make the test more useful for the LLM *parsing* stage, we could manually set raw_text later if needed,
    # but the flow should handle empty raw_text.
    try:
        with open(UPLOAD_DIR / dummy_img_filename, "wb") as f:
            f.write(dummy_image_content_bytes)
        print(f"Created dummy Image: {UPLOAD_DIR / dummy_img_filename}")
    except Exception as e:
        print(f"Could not create dummy Image: {e}")


    # Step 1: User asks "what are available bills present?"
    print("\n--- Step 1: Discovering available bills ---")
    available_bills = await discover_available_bills()
    if "message" in available_bills[0]:
        print(f"Discovery message: {available_bills[0]['message']}")
    else:
        print("Available bill files in uploads directory:")
        for bill_file_info in available_bills:
            print(f"  - {bill_file_info['filename']} {bill_file_info.get('status','')}")
    
    if not available_bills or "message" in available_bills[0]:
        print("No bills to process. Exiting test.")
        return

    # --- Simulate processing the first available PDF bill ---
    selected_pdf_bill_filename = None
    for bill_info in available_bills:
        if bill_info['filename'].endswith(".pdf"):
            selected_pdf_bill_filename = bill_info['filename']
            break
    
    if selected_pdf_bill_filename:
        print(f"\n--- Step 2: User selects '{selected_pdf_bill_filename}' and asks a question ---")
        first_question_pdf = "What is the total amount?"
        print(f"User asks: '{first_question_pdf}' for bill '{selected_pdf_bill_filename}'")
        try:
            response1 = await process_selected_bill_and_ask(
                original_filename=selected_pdf_bill_filename,
                question=first_question_pdf
            )
            print(f"  System Answer: {response1.answer} (Bill ID: {response1.bill_id})")
            processed_bill_id_pdf = response1.bill_id

            # Step 3: User asks a follow-up question about the same PDF bill
            print("\n--- Step 3: User asks a follow-up question for the PDF bill ---")
            second_question_pdf = "What items are listed?"
            print(f"User asks: '{second_question_pdf}' for bill ID '{processed_bill_id_pdf}'")
            response2 = await ask_question_and_save(
                bill_identifier=processed_bill_id_pdf, 
                question=second_question_pdf
            )
            print(f"  System Answer: {response2.answer}")

            # Verify Q&A in memory
            final_bill_data_pdf = await app_state.find_bill(processed_bill_id_pdf)
            if final_bill_data_pdf:
                print(f"  Stored Q&A for bill {final_bill_data_pdf.id}:")
                for qa in final_bill_data_pdf.q_and_a:
                    print(f"    Q: {qa.question} -> A: {qa.answer}")
            
            # Step 4: Export the PDF bill data
            print("\n--- Step 4: Exporting the processed PDF bill data ---")
            export_info = await export_bill_data(bill_identifier=processed_bill_id_pdf)
            print(f"  Export result: {export_info['message']} at {export_info['path']}")

        except ValueError as e:
            print(f"  Error during PDF bill processing or Q&A: {e}")
        except Exception as e:
            print(f"  Unexpected error: {e}", exc_info=True)
    else:
        print("\nNo PDF bill found in uploads to test with.")

    # --- Optionally, clean up dummy files ---
    # try:
    #     if selected_pdf_bill_filename: os.remove(UPLOAD_DIR / selected_pdf_bill_filename)
    #     os.remove(UPLOAD_DIR / dummy_img_filename) # If you also tested image
    #     print("\nCleaned up dummy files from uploads.")
    # except OSError as e:
    #     print(f"Error cleaning up dummy files: {e}")


if __name__ == "__main__":
    # Make sure to set your OPENAI_API_KEY in a .env file or as an environment variable
    # e.g., create a .env file in the same directory as this script with:
    # OPENAI_API_KEY="sk-your-actual-key"
    
    # The main_test_new_flow() is for local testing.
    # In a real FastMCP deployment, the MCP server would run and expose tools.
    pass
    #asyncio.run(main_test_new_flow())

    # To run the MCP server (example, not directly from this script usually):
    # from mcp import McpAsyncRunner
    # runner = McpAsyncRunner(mcp_instance=mcp)
    # asyncio.run(runner.run_server_async(port=8000))