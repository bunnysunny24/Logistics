import os
import logging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import tempfile
import chardet
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List
import traceback

class PDFExtractor:
    """
    Enhanced PDF extraction with PyMuPDF and OCR capabilities
    """
    
    def __init__(self, ocr_enabled=True, ocr_language='eng'):
        """
        Initialize the PDF extractor with OCR capabilities
        
        Args:
            ocr_enabled (bool): Whether to use OCR for images and when text extraction fails
            ocr_language (str): Language for OCR (default: 'eng')
        """
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.data_dir = os.environ.get("DATA_DIR", "./data")
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Check if Tesseract is available
        if ocr_enabled:
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR is available")
            except Exception as e:
                logger.warning(f"Tesseract OCR not properly configured: {e}")
                self.ocr_enabled = False
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF with fallback to OCR
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return ""
            
        try:
            # Try standard text extraction first
            text = self._extract_with_pymupdf(pdf_path)
            
            # If little or no text was extracted and OCR is enabled, try OCR
            if len(text.strip()) < 100 and self.ocr_enabled:
                logger.info(f"Minimal text extracted from {pdf_path}. Trying OCR...")
                text = self._extract_with_ocr(pdf_path)
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return f"Error extracting text from {os.path.basename(pdf_path)}: {str(e)}"
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF with encoding detection"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                # Get text with encoding detection
                page_text = page.get_text()
                
                # If text is empty or corrupted, try different encoding
                if not page_text or page_text.isspace():
                    # Try to get raw bytes and detect encoding
                    try:
                        byte_content = page.get_text("rawdict").encode("utf-8", errors="replace")
                        detected = chardet.detect(byte_content)
                        if detected['confidence'] > 0.7:
                            try:
                                page_text = byte_content.decode(detected['encoding'])
                            except:
                                pass
                    except:
                        pass
                
                text += page_text + "\n\n"
            return text
        except Exception as e:
            logger.error(f"PyMuPDF extraction error: {e}")
            raise
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR by rendering PDF pages as images"""
        text = ""
        temp_dir = tempfile.mkdtemp()
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                # Render page to an image at higher resolution for better OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                pix.save(image_path)
                
                # Perform OCR on the image
                img = Image.open(image_path)
                page_text = pytesseract.image_to_string(img, lang=self.ocr_language)
                text += page_text + "\n\n"
                
                # Clean up temp file
                os.remove(image_path)
                
            return text
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            raise
        finally:
            # Clean up temp directory
            try:
                os.rmdir(temp_dir)
            except:
                pass
    
    def extract_structured_data(self, pdf_path: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract structured data from PDF files based on document type
        
        Args:
            pdf_path (str): Path to the PDF file
            doc_type (str): Type of document (invoice, shipment, policy)
            
        Returns:
            Dict[str, Any]: Structured data extracted from the PDF
        """
        # Extract text first
        text = self.extract_text(pdf_path)
        
        # Process according to document type
        if doc_type == "invoice":
            return self._extract_invoice_data(text, pdf_path)
        elif doc_type == "shipment":
            return self._extract_shipment_data(text, pdf_path)
        else:
            # Return basic metadata for other types
            return {
                "document_id": os.path.basename(pdf_path).replace(".pdf", ""),
                "content": text,
                "page_count": self._get_page_count(pdf_path)
            }
    
    def _extract_invoice_data(self, text: str, pdf_path: str) -> Dict[str, Any]:
        """Extract invoice data from extracted text"""
        # In a real implementation, use NLP or regex patterns to extract structured data
        # For now, return basic info with the full text for further processing
        filename = os.path.basename(pdf_path).replace(".pdf", "")
        
        # Basic implementation - In production, use NER or regex to extract fields
        invoice_data = {
            "invoice_id": filename,
            "supplier": self._extract_pattern(text, r"Supplier:\s*([^\n]+)") or "Unknown",
            "amount": self._extract_amount(text) or 0.0,
            "currency": self._extract_pattern(text, r"\$|€|£|USD|EUR|GBP") or "USD",
            "issue_date": self._extract_date(text, "issue date", "date") or "Unknown",
            "due_date": self._extract_date(text, "due date", "payment") or "Unknown",
            "full_text": text  # Store full text for RAG
        }
        
        return invoice_data
    
    def _extract_shipment_data(self, text: str, pdf_path: str) -> Dict[str, Any]:
        """Extract shipment data from extracted text"""
        filename = os.path.basename(pdf_path).replace(".pdf", "")
        
        # Basic implementation - In production, use NER or regex to extract fields
        shipment_data = {
            "shipment_id": filename,
            "origin": self._extract_pattern(text, r"Origin:\s*([^\n]+)") or 
                     self._extract_pattern(text, r"From:\s*([^\n]+)") or "Unknown",
            "destination": self._extract_pattern(text, r"Destination:\s*([^\n]+)") or 
                          self._extract_pattern(text, r"To:\s*([^\n]+)") or "Unknown",
            "carrier": self._extract_pattern(text, r"Carrier:\s*([^\n]+)") or "Unknown",
            "status": self._extract_pattern(text, r"Status:\s*([^\n]+)") or "Unknown",
            "full_text": text  # Store full text for RAG
        }
        
        return shipment_data
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF"""
        try:
            doc = fitz.open(pdf_path)
            return len(doc)
        except Exception:
            return 0
    
    def _extract_pattern(self, text: str, pattern: str) -> Optional[str]:
        """Extract text using regex pattern"""
        import re
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
        return None
    
    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract amount from text"""
        import re
        # Look for currency amounts like $1,234.56
        match = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?!\d)', text)
        if match:
            try:
                # Remove commas and convert to float
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass
        return None
    
    def _extract_date(self, text: str, *keywords) -> Optional[str]:
        """Extract date from text based on keywords"""
        import re
        for keyword in keywords:
            # Look for dates in various formats near keywords
            pattern = rf'{keyword}.*?(\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        # Try to find any date-like strings
        date_pattern = r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})'
        match = re.search(date_pattern, text)
        if match:
            return match.group(1)
        return None