import pandas as pd
import pdfplumber
import pymupdf
import re
from typing import Dict, Any, List, Optional
from loguru import logger
import os
import json

class InvoiceExtractor:
    """
    Utility class for extracting data from invoice documents
    """
    
    def __init__(self):
        pass
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract invoice data from PDF file
        """
        try:
            # First try with PyMuPDF for better performance
            return self._extract_with_pymupdf(pdf_path)
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}, trying pdfplumber")
            try:
                # Fall back to p