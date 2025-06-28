import os
import shutil
import pandas as pd
import time
from typing import Dict, Any, Optional, List
from loguru import logger
from datetime import datetime
import json

class DocumentProcessor:
    """
    Utility class for processing different document types
    """
    
    def __init__(self):
        self.data_dir = os.environ.get("DATA_DIR", "./data")
        
        # Create necessary directories
        os.makedirs(f"{self.data_dir}/invoices", exist_ok=True)
        os.makedirs(f"{self.data_dir}/shipments", exist_ok=True)
        os.makedirs(f"{self.data_dir}/policies", exist_ok=True)
    
    def process_document(self, document_path: str, doc_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a document and store it in the appropriate directory
        Returns the path to the processed document
        """
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        filename = os.path.basename(document_path)
        extension = os.path.splitext(filename)[1].lower()
        
        # Determine target directory based on document type
        if doc_type == "invoice":
            target_dir = f"{self.data_dir}/invoices"
        elif doc_type == "shipment":
            target_dir = f"{self.data_dir}/shipments"
        elif doc_type == "policy":
            target_dir = f"{self.data_dir}/policies"
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
        
        # Generate destination path with timestamp to avoid overwrites
        timestamp = int(time.time())
        dest_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{extension}"
        dest_path = os.path.join(target_dir, dest_filename)
        
        # Process based on file type
        if extension == ".pdf":
            return self._process_pdf(document_path, dest_path, doc_type, metadata)
        elif extension == ".csv":
            return self._process_csv(document_path, dest_path, doc_type, metadata)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    
    def _process_pdf(self, source_path: str, dest_path: str, doc_type: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Process PDF document"""
        try:
            # In a real implementation, we would extract text and data from the PDF
            # For now, we'll just copy the file
            shutil.copy2(source_path, dest_path)
            
            # Save metadata alongside the file
            if metadata:
                metadata_path = f"{os.path.splitext(dest_path)[0]}.meta.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "doc_type": doc_type,
                        "original_path": source_path,
                        "processed_at": datetime.now().isoformat(),
                        **metadata
                    }, f, indent=2)
            
            logger.info(f"Processed PDF document: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"Error processing PDF {source_path}: {e}")
            raise
    
    def _process_csv(self, source_path: str, dest_path: str, doc_type: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Process CSV document"""
        try:
            # Read CSV file
            df = pd.read_csv(source_path)
            
            # In a real implementation, we would normalize and validate the data
            # For now, we'll just save it as-is
            df.to_csv(dest_path, index=False)
            
            # Save metadata alongside the file
            if metadata:
                metadata_path = f"{os.path.splitext(dest_path)[0]}.meta.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "doc_type": doc_type,
                        "original_path": source_path,
                        "processed_at": datetime.now().isoformat(),
                        "row_count": len(df),
                        "columns": df.columns.tolist(),
                        **metadata
                    }, f, indent=2)
            
            logger.info(f"Processed CSV document: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"Error processing CSV {source_path}: {e}")
            raise
    
    def extract_invoice_data(self, invoice_path: str) -> Dict[str, Any]:
        """
        Extract structured data from an invoice file
        """
        extension = os.path.splitext(invoice_path)[1].lower()
        
        if extension == ".csv":
            return self._extract_invoice_data_csv(invoice_path)
        elif extension == ".pdf":
            return self._extract_invoice_data_pdf(invoice_path)
        else:
            raise ValueError(f"Unsupported file extension for invoice: {extension}")
    
    def _extract_invoice_data_csv(self, invoice_path: str) -> Dict[str, Any]:
        """Extract data from CSV invoice"""
        try:
            df = pd.read_csv(invoice_path)
            
            # Extract basic invoice information
            # This is a simplified implementation
            invoice_data = {
                "invoice_id": df.get("invoice_id", [None])[0] or os.path.basename(invoice_path),
                "supplier": df.get("supplier", [None])[0] or "Unknown",
                "amount": df.get("amount", [0.0])[0] or 0.0,
                "currency": df.get("currency", ["USD"])[0] or "USD",
                "issue_date": df.get("issue_date", [None])[0] or "Unknown",
                "due_date": df.get("due_date", [None])[0] or "Unknown",
                "line_items": []
            }
            
            # Extract line items if available
            if all(col in df.columns for col in ["item", "quantity", "unit_price"]):
                for _, row in df.iterrows():
                    line_item = {
                        "item": row["item"],
                        "quantity": row["quantity"],
                        "unit_price": row["unit_price"],
                        "total": row.get("total", row["quantity"] * row["unit_price"])
                    }
                    invoice_data["line_items"].append(line_item)
            
            return invoice_data
        except Exception as e:
            logger.error(f"Error extracting data from CSV invoice {invoice_path}: {e}")
            return {
                "invoice_id": os.path.basename(invoice_path),
                "error": str(e)
            }
    
    def _extract_invoice_data_pdf(self, invoice_path: str) -> Dict[str, Any]:
        """Extract data from PDF invoice"""
        # In a real implementation, this would use OCR or PDF parsing
        # For now, return placeholder data
        return {
            "invoice_id": os.path.basename(invoice_path).replace(".pdf", ""),
            "supplier": "PDF Supplier",
            "amount": 1000.0,
            "currency": "USD",
            "issue_date": datetime.now().strftime("%Y-%m-%d"),
            "due_date": datetime.now().strftime("%Y-%m-%d"),
            "line_items": [
                {
                    "item": "Sample Item 1",
                    "quantity": 2,
                    "unit_price": 250.0,
                    "total": 500.0
                },
                {
                    "item": "Sample Item 2",
                    "quantity": 1,
                    "unit_price": 500.0,
                    "total": 500.0
                }
            ]
        }
    
    def extract_shipment_data(self, shipment_path: str) -> Dict[str, Any]:
        """
        Extract structured data from a shipment file
        """
        extension = os.path.splitext(shipment_path)[1].lower()
        
        if extension == ".csv":
            return self._extract_shipment_data_csv(shipment_path)
        elif extension == ".pdf":
            return self._extract_shipment_data_pdf(shipment_path)
        else:
            raise ValueError(f"Unsupported file extension for shipment: {extension}")
    
    def _extract_shipment_data_csv(self, shipment_path: str) -> Dict[str, Any]:
        """Extract data from CSV shipment"""
        try:
            df = pd.read_csv(shipment_path)
            
            # Extract basic shipment information
            # This is a simplified implementation
            shipment_data = {
                "shipment_id": df.get("shipment_id", [None])[0] or os.path.basename(shipment_path),
                "origin": df.get("origin", [None])[0] or "Unknown",
                "destination": df.get("destination", [None])[0] or "Unknown",
                "carrier": df.get("carrier", [None])[0] or "Unknown",
                "departure_date": df.get("departure_date", [None])[0] or "Unknown",
                "arrival_date": df.get("arrival_date", [None])[0] or "Unknown",
                "status": df.get("status", [None])[0] or "Unknown",
                "items": []
            }
            
            # Extract items if available
            if all(col in df.columns for col in ["item", "quantity", "value"]):
                for _, row in df.iterrows():
                    item = {
                        "item": row["item"],
                        "quantity": row["quantity"],
                        "value": row["value"],
                        "weight": row.get("weight", 0.0),
                        "dimensions": row.get("dimensions", "Unknown")
                    }
                    shipment_data["items"].append(item)
            
            return shipment_data
        except Exception as e:
            logger.error(f"Error extracting data from CSV shipment {shipment_path}: {e}")
            return {
                "shipment_id": os.path.basename(shipment_path),
                "error": str(e)
            }
    
    def _extract_shipment_data_pdf(self, shipment_path: str) -> Dict[str, Any]:
        """Extract data from PDF shipment"""
        # In a real implementation, this would use OCR or PDF parsing
        # For now, return placeholder data
        return {
            "shipment_id": os.path.basename(shipment_path).replace(".pdf", ""),
            "origin": "New York, USA",
            "destination": "London, UK",
            "carrier": "Global Shipping Inc.",
            "departure_date": datetime.now().strftime("%Y-%m-%d"),
            "arrival_date": (datetime.now().replace(day=datetime.now().day + 7)).strftime("%Y-%m-%d"),
            "status": "In Transit",
            "items": [
                {
                    "item": "Electronics",
                    "quantity": 10,
                    "value": 5000.0,
                    "weight": 25.5,
                    "dimensions": "60x40x30 cm"
                },
                {
                    "item": "Clothing",
                    "quantity": 50,
                    "value": 2000.0,
                    "weight": 15.0,
                    "dimensions": "70x50x40 cm"
                }
            ]
        }