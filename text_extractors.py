from enum import Enum
from abc import ABC, abstractmethod
import os
import pymupdf as fitz




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
            from docling.document_converter import DocumentConverter
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
