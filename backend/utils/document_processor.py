import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Set up logging
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles extraction of structured data from PDF documents.
    Supports invoices, shipments, and policy documents.
    """
    
    def __init__(self):
        try:
            # Import PDF processing libraries
            import pdfplumber
            import pytesseract
            from PIL import Image
            
            self.pdfplumber = pdfplumber
            self.pytesseract = pytesseract
            self.Image = Image
            self.ocr_enabled = True
        except ImportError as e:
            logger.warning(f"PDF OCR capabilities limited due to missing dependencies: {e}")
            self.ocr_enabled = False
    
    def process_pdf(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract data from PDF file based on document type
        
        Args:
            file_path: Path to the PDF file
            doc_type: Type of document (invoice, shipment, policy)
            
        Returns:
            Dictionary of extracted data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        if doc_type.lower() == "invoice":
            return self._extract_invoice_data(file_path)
        elif doc_type.lower() == "shipment":
            return self._extract_shipment_data(file_path)
        elif doc_type.lower() == "policy":
            return self._extract_policy_data(file_path)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text content from PDF"""
        try:
            text_content = []
            with self.pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    text_content.append(text)
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def _extract_tables_from_pdf(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables from PDF"""
        try:
            tables = []
            with self.pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table_data in page_tables:
                            # Convert to DataFrame
                            if table_data and len(table_data) > 1:  # Has header and data
                                headers = table_data[0]
                                data = table_data[1:]
                                df = pd.DataFrame(data, columns=headers)
                                tables.append(df)
            
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables from PDF {pdf_path}: {e}")
            return []
    
    def _extract_invoice_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured data from invoice PDF"""
        # Extract text content
        text_content = self._extract_text_from_pdf(pdf_path)
        
        # Extract tables
        tables = self._extract_tables_from_pdf(pdf_path)
        
        # Initialize with basic metadata
        invoice_data = {
            "invoice_id": self._extract_invoice_id(text_content),
            "extracted_from": os.path.basename(pdf_path),
            "processed_at": datetime.now().isoformat()
        }
        
        # Extract invoice details using regex patterns
        invoice_data.update({
            "supplier": self._extract_pattern(text_content, r"(?:Supplier|Vendor|From):\s*([A-Za-z0-9\s.,]+)"),
            "amount": self._extract_amount(text_content),
            "currency": self._extract_pattern(text_content, r"(?:Currency|USD|EUR|GBP):\s*([A-Za-z]{3})") or "USD",
            "issue_date": self._extract_date(text_content, r"(?:Issue Date|Invoice Date|Date):\s*(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})"),
            "due_date": self._extract_date(text_content, r"(?:Due Date|Payment Due|Pay Before):\s*(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})"),
            "payment_terms": self._extract_payment_terms(text_content),
            "early_discount": self._extract_early_discount(text_content),
            "shipment_id": self._extract_pattern(text_content, r"(?:Shipment|Delivery|Order)\s(?:ID|Number|#):\s*([A-Za-z0-9-]+)")
        })
        
        # Extract line items if available
        if tables:
            line_items = self._extract_line_items(tables)
            if line_items:
                invoice_data["line_items"] = line_items
        
        return invoice_data
    
    def _extract_shipment_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured data from shipment PDF"""
        # Extract text content
        text_content = self._extract_text_from_pdf(pdf_path)
        
        # Extract tables
        tables = self._extract_tables_from_pdf(pdf_path)
        
        # Initialize with basic metadata
        shipment_data = {
            "shipment_id": self._extract_pattern(text_content, r"(?:Shipment|Tracking|Waybill)\s(?:ID|Number|#):\s*([A-Za-z0-9-]+)"),
            "extracted_from": os.path.basename(pdf_path),
            "processed_at": datetime.now().isoformat()
        }
        
        # Extract shipment details using regex patterns
        shipment_data.update({
            "origin": self._extract_pattern(text_content, r"(?:Origin|From|Pickup):\s*([A-Za-z0-9\s.,]+)"),
            "destination": self._extract_pattern(text_content, r"(?:Destination|To|Delivery):\s*([A-Za-z0-9\s.,]+)"),
            "carrier": self._extract_pattern(text_content, r"(?:Carrier|Shipper|Provider):\s*([A-Za-z0-9\s.,]+)"),
            "departure_date": self._extract_date(text_content, r"(?:Departure|Ship|Pickup) Date:\s*(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})"),
            "estimated_arrival": self._extract_date(text_content, r"(?:Estimated|Expected|Planned) (?:Arrival|Delivery) Date:\s*(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})"),
            "status": self._extract_shipment_status(text_content),
            "driver_id": self._extract_pattern(text_content, r"(?:Driver|Operator)\s(?:ID|Number):\s*([A-Za-z0-9-]+)")
        })
        
        # Extract items if available
        if tables:
            items = self._extract_shipment_items(tables)
            if items:
                shipment_data["items"] = items
        
        return shipment_data
    
    def _extract_policy_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured data from policy PDF"""
        # For policy documents, we primarily care about the full text content
        text_content = self._extract_text_from_pdf(pdf_path)
        
        # Initialize with basic metadata
        policy_data = {
            "policy_id": f"policy_{int(datetime.now().timestamp())}",
            "policy_file": os.path.basename(pdf_path),
            "extracted_at": datetime.now().isoformat(),
            "content": text_content
        }
        
        # Try to extract policy version
        version_match = re.search(r"(?:Version|v)[\s.:]*([0-9.]+)", text_content)
        if version_match:
            policy_data["version"] = version_match.group(1)
        
        # Try to extract policy type
        if "payment" in text_content.lower():
            policy_data["policy_type"] = "payment"
        elif "shipment" in text_content.lower():
            policy_data["policy_type"] = "shipment"
        elif "driver" in text_content.lower():
            policy_data["policy_type"] = "driver"
        else:
            policy_data["policy_type"] = "general"
        
        # Extract policy sections
        policy_data["sections"] = self._extract_policy_sections(text_content)
        
        return policy_data
    
    def _extract_invoice_id(self, text: str) -> str:
        """Extract invoice ID from text"""
        patterns = [
            r"Invoice\s(?:ID|Number|#):\s*([A-Za-z0-9-]+)",
            r"Invoice\s*#\s*([A-Za-z0-9-]+)",
            r"INV[-:]?(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # If no match found, generate a timestamp-based ID
        return f"INV-{int(datetime.now().timestamp())}"
    
    def _extract_amount(self, text: str) -> float:
        """Extract invoice amount from text"""
        patterns = [
            r"(?:Total|Amount|Sum):\s*[$€£]?\s*([0-9,]+\.[0-9]{2})",
            r"[$€£]\s*([0-9,]+\.[0-9]{2})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Remove commas and convert to float
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    pass
        
        return 0.0
    
    def _extract_date(self, text: str, pattern: str) -> Optional[str]:
        """Extract and standardize date from text"""
        match = re.search(pattern, text)
        if not match:
            return None
            
        date_str = match.group(1)
        
        # Try different date formats
        for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%m-%d-%Y', '%m/%d/%Y'):
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')  # Standardize to ISO format
            except ValueError:
                continue
                
        return date_str  # Return as-is if parsing fails
    
    def _extract_payment_terms(self, text: str) -> str:
        """Extract payment terms from text"""
        patterns = [
            r"(?:Payment Terms|Terms):\s*([A-Za-z0-9\s]+)",
            r"NET\s?(\d+)",
            r"(\d+)\s?days"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                terms = match.group(1).strip()
                # Standardize common terms
                if re.match(r'\d+', terms):
                    return f"NET{terms}"
                return terms
        
        return "NET30"  # Default if not found
    
    def _extract_early_discount(self, text: str) -> float:
        """Extract early payment discount percentage"""
        patterns = [
            r"(?:Early Payment|Prompt Payment|Early Discount):\s*(\d+(?:\.\d+)?)%",
            r"(\d+(?:\.\d+)?)%\s+(?:discount|off)\s+(?:if paid|for payment)\s+within"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1)) / 100.0  # Convert percentage to decimal
                except ValueError:
                    pass
        
        return 0.0  # Default if not found
    
    def _extract_line_items(self, tables: List[pd.DataFrame]) -> List[Dict]:
        """Extract line items from invoice tables"""
        line_items = []
        
        # Try each table
        for df in tables:
            # Check if this looks like a line items table
            if set(['item', 'description', 'quantity', 'price', 'amount']).issubset(set(map(str.lower, df.columns))) or \
               any(col.lower() in ['item', 'description', 'product'] for col in df.columns):
                
                # Process each row
                for _, row in df.iterrows():
                    item = {}
                    
                    # Map common column names
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['item', 'description', 'product']:
                            item['description'] = str(row[col])
                        elif col_lower in ['quantity', 'qty']:
                            item['quantity'] = float(row[col]) if pd.notna(row[col]) else 0
                        elif col_lower in ['price', 'unit price', 'rate']:
                            item['unit_price'] = float(row[col]) if pd.notna(row[col]) else 0
                        elif col_lower in ['amount', 'total', 'line total']:
                            item['amount'] = float(row[col]) if pd.notna(row[col]) else 0
                    
                    if item:
                        line_items.append(item)
        
        return line_items
    
    def _extract_shipment_status(self, text: str) -> str:
        """Extract shipment status from text"""
        status_patterns = {
            'delivered': r'(?:Delivered|Completed|Received)',
            'in_transit': r'(?:In Transit|On Route|In Progress)',
            'pending': r'(?:Pending|Waiting|Scheduled)',
            'delayed': r'(?:Delayed|Late|Behind Schedule)',
            'exception': r'(?:Exception|Problem|Issue|Missing)'
        }
        
        for status, pattern in status_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return status
        
        return 'unknown'  # Default if no status found
    
    def _extract_shipment_items(self, tables: List[pd.DataFrame]) -> List[Dict]:
        """Extract items from shipment tables"""
        items = []
        
        # Try each table
        for df in tables:
            # Check if this looks like a shipment items table
            if any(col.lower() in ['item', 'product', 'goods'] for col in df.columns):
                
                # Process each row
                for _, row in df.iterrows():
                    item = {}
                    
                    # Map common column names
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['item', 'product', 'goods', 'description']:
                            item['item'] = str(row[col])
                        elif col_lower in ['quantity', 'qty', 'count']:
                            item['quantity'] = float(row[col]) if pd.notna(row[col]) else 0
                        elif col_lower in ['value', 'price', 'worth']:
                            item['value'] = float(row[col]) if pd.notna(row[col]) else 0
                        elif col_lower in ['weight', 'kg', 'lb']:
                            item['weight'] = float(row[col]) if pd.notna(row[col]) else 0
                        elif col_lower in ['dimensions', 'size', 'measurements']:
                            item['dimensions'] = str(row[col])
                    
                    if item and 'item' in item:
                        items.append(item)
        
        return items
    
    def _extract_pattern(self, text: str, pattern: str) -> Optional[str]:
        """Extract text matching a regex pattern"""
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None
    
    def _extract_policy_sections(self, text: str) -> Dict[str, str]:
        """Extract policy sections based on headers"""
        sections = {}
        
        # Split by common section headers
        headers = re.findall(r'^#+\s+(.*?)$|^([A-Z][A-Za-z\s]+):$', text, re.MULTILINE)
        
        if headers:
            # Extract header titles
            header_titles = [h[0] or h[1] for h in headers if h[0] or h[1]]
            
            # Get content between headers
            content_blocks = re.split(r'^#+\s+.*?$|^[A-Z][A-Za-z\s]+:$', text, flags=re.MULTILINE)
            
            # Skip the first block if it's empty (before first header)
            if content_blocks and not content_blocks[0].strip():
                content_blocks = content_blocks[1:]
                
            # Map headers to content
            for i, title in enumerate(header_titles):
                if i < len(content_blocks):
                    sections[title.strip()] = content_blocks[i].strip()
        
        # If no sections found, include full text as single section
        if not sections:
            sections["Full Content"] = text
            
        return sections