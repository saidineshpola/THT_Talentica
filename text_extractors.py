from enum import Enum
from abc import ABC, abstractmethod
import logging
import os
import pymupdf as fitz

logger = logging.getLogger("document_management.text_extractors")



class TextExtractionBackend(Enum):
    PYMUPDF = "pymupdf"
    DOCLING = "docling"
    AWS_TEXTRACT = "aws_textract"

class TextExtractor(ABC):
    @abstractmethod
    async def extract_text(self, pdf_bytes: bytes) -> str:
        pass

class PyMuPDFExtractor(TextExtractor):
    async def extract_text(self, pdf_bytes: bytes) -> str:
        raw_text = ""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                raw_text += page.get_text("text") + "\n"
            doc.close()
            return raw_text
        except Exception as e:
            raise ValueError(f"PyMuPDF text extraction failed: {str(e)}")

class DoclingExtractor(TextExtractor):
    async def extract_text(self, pdf_bytes: bytes) -> str:
        try:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError:
                raise ImportError("Docling package not installed. Install with: pip install docling")
            import tempfile
            # Save bytes to temporary file since Docling expects a path
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name

            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file
            return result.document.export_to_markdown()
        except ImportError:
            raise ImportError("Docling package not installed. Install with: pip install docling")
        except Exception as e:
            raise ValueError(f"Docling text extraction failed: {str(e)}")

class AWSTextractExtractor(TextExtractor):
    async def extract_text(self, pdf_bytes: bytes) -> str:
        raise NotImplementedError("AWS Textract extraction not implemented. Please implement this method.")
        # Placeholder for AWS Textract extraction logic

class TextExtractorFactory:
    @staticmethod
    def create_extractor(backend: TextExtractionBackend) -> TextExtractor:
        if backend == TextExtractionBackend.PYMUPDF:
            return PyMuPDFExtractor()
        elif backend == TextExtractionBackend.DOCLING:
            return DoclingExtractor()
        elif backend == TextExtractionBackend.AWS_TEXTRACT:
            return AWSTextractExtractor()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

async def extract_text_from_pdf(pdf_bytes: bytes, backend: TextExtractionBackend = TextExtractionBackend.PYMUPDF) -> str:
    """
    Extract text from PDF using the specified backend.
    
    Args:
        pdf_bytes: PDF content as bytes
        backend: TextExtractionBackend enum specifying which extractor to use
        
    Returns:
        str: Extracted text
        
    Raises:
        ValueError: If text extraction fails
    """
    raw_text = ""
    try:
        extractor = TextExtractorFactory.create_extractor(backend)
        raw_text = await extractor.extract_text(pdf_bytes)
        
        if not raw_text.strip():
            logger.warning("No text could be extracted from the provided PDF.")
        logger.info(f"Extracted raw text length from PDF using {backend.value}: {len(raw_text)}")
        return raw_text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF using {backend.value}: {e}", exc_info=True)
        raise ValueError(f"Text extraction from PDF failed: {str(e)}")
    
async def extract_text_from_image(image_bytes: bytes, image_format: str) -> str:
    raw_text = ""
    try:
        doc = fitz.open(stream=image_bytes, filetype=image_format)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_from_page = page.get_text("text")
            raw_text += text_from_page + "\n"
        doc.close()
        if not raw_text.strip():
            logger.warning(f"No text could be extracted from image (format: {image_format}). May require OCR for scanned images.")
        logger.info(f"Extracted raw text length from image: {len(raw_text)}")
        return raw_text
    except Exception as e:
        logger.error(f"Failed to extract text from image: {e}", exc_info=True)
        return "" # Allow LLM to attempt interpretation or note lack of text
