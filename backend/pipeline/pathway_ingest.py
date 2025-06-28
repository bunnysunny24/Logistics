import pathway as pw
from pathway.stdlib.ml.index import KNNIndex
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import json
from datetime import datetime

class PathwayIngestPipeline:
    """
    Pathway pipeline for ingesting documents and detecting changes
    """
    
    def __init__(
        self,
        watch_dir: str,
        embeddings_model: str,
        vector_dimensions: int = 1536,  # Default for OpenAI embeddings
        index_refresh_rate_s: int = 10
    ):
        self.watch_dir = watch_dir
        self.embeddings_model = embeddings_model
        self.vector_dimensions = vector_dimensions
        self.index_refresh_rate_s = index_refresh_rate_s
        self.document_stats = {}
        self.anomalies = []
        
    def build_pipeline(self):
        """
        Build the Pathway data processing pipeline
        """
        # Create a schema for the document chunks
        class DocumentSchema(pw.Schema):
            id: str
            content: str
            metadata: Dict[str, Any]
            embedding: Optional[List[float]]
            timestamp: float
            
        # Create a schema for anomaly detection
        class AnomalySchema(pw.Schema):
            id: str
            document_id: str
            anomaly_type: str
            risk_score: float
            description: str
            timestamp: float
            metadata: Dict[str, Any]
        
        # Set up document monitoring
        csv_table = pw.io.fs.read(
            self.watch_dir + "/**/*.csv",
            format="csv",
            mode="streaming",
            with_metadata=True
        )
        
        pdf_table = pw.io.fs.read(
            self.watch_dir + "/**/*.pdf",
            format="binary",
            mode="streaming",
            with_metadata=True
        )
        
        # Process and extract text from PDFs
        pdf_docs = pdf_table.select(
            id=pw.this["_pw_metadata"]["path"],
            content=self._extract_pdf_text(pw.this["_pw_raw"]),
            metadata=self._extract_pdf_metadata(pw.this["_pw_metadata"]),
            timestamp=pw.this["_pw_metadata"]["mtime"]
        )
        
        # Process CSV files
        csv_docs = csv_table.select(
            id=pw.this["_pw_metadata"]["path"],
            content=self._convert_csv_to_text(pw.this),
            metadata=self._extract_csv_metadata(pw.this["_pw_metadata"]),
            timestamp=pw.this["_pw_metadata"]["mtime"]
        )
        
        # Combine document sources
        all_docs = pw.concat(pdf_docs, csv_docs)
        
        # Chunk documents into smaller pieces
        chunked_docs = all_docs.select(
            chunks=self._chunk_document(pw.this.content, pw.this.id, pw.this.metadata, pw.this.timestamp)
        ).flatten(pw.this.chunks)
        
        # Create embeddings for the chunks
        embedded_chunks = chunked_docs.select(
            id=pw.this.id,
            content=pw.this.content,
            metadata=pw.this.metadata,
            timestamp=pw.this.timestamp,
            embedding=self._compute_embedding(pw.this.content)
        )
        
        # Create vector index
        index = KNNIndex(
            embedded_chunks,
            vector_column_name="embedding",
            dimensions=self.vector_dimensions,
            distance_type="cosine"
        )
        
        # Export the index
        pw.io.jsonl.write(
            embedded_chunks,
            self.watch_dir + "/index/chunks.jsonl"
        )
        
        # Run anomaly detection on the documents
        anomalies = self._detect_anomalies(all_docs)
        
        # Export anomalies
        pw.io.jsonl.write(
            anomalies,
            self.watch_dir + "/anomalies/detected.jsonl"
        )
        
        # Compute document statistics
        doc_stats = all_docs.groupby(doc_type=pw.this.metadata["doc_type"]).reduce(
            count=pw.reducers.count(),
            latest_update=pw.reducers.max(pw.this.timestamp)
        )
        
        # Export statistics
        pw.io.jsonl.write(
            doc_stats,
            self.watch_dir + "/stats/document_stats.jsonl"
        )
        
        return index
    
    def _extract_pdf_text(self, pdf_data) -> str:
        """Extract text from PDF binary data"""
        # This would use PyMuPDF or pdfplumber in a real implementation
        # For now, we'll just return a placeholder
        return f"PDF text extracted at {time.time()}"
    
    def _extract_pdf_metadata(self, metadata) -> Dict[str, Any]:
        """Extract metadata from PDF file"""
        path = metadata["path"]
        filename = os.path.basename(path)
        
        result = {
            "doc_type": "unknown",
            "filename": filename,
            "path": path,
            "created_at": metadata.get("ctime", time.time()),
            "modified_at": metadata.get("mtime", time.time()),
        }
        
        # Determine document type based on filename patterns
        if "invoice" in filename.lower():
            result["doc_type"] = "invoice"
        elif "payout" in filename.lower() or "rules" in filename.lower():
            result["doc_type"] = "policy"
        elif "shipment" in filename.lower():
            result["doc_type"] = "shipment"
            
        return result
    
    def _convert_csv_to_text(self, csv_data) -> str:
        """Convert CSV data to text representation"""
        # In a real implementation, this would parse the CSV properly
        rows = []
        for key, value in csv_data.items():
            if not key.startswith("_pw_"):
                rows.append(f"{key}: {value}")
        
        return "\n".join(rows)
    
    def _extract_csv_metadata(self, metadata) -> Dict[str, Any]:
        """Extract metadata from CSV file"""
        path = metadata["path"]
        filename = os.path.basename(path)
        
        result = {
            "doc_type": "unknown",
            "filename": filename,
            "path": path,
            "created_at": metadata.get("ctime", time.time()),
            "modified_at": metadata.get("mtime", time.time()),
        }
        
        # Determine document type based on filename patterns
        if "invoice" in filename.lower():
            result["doc_type"] = "invoice"
        elif "shipment" in filename.lower():
            result["doc_type"] = "shipment"
            
        return result
    
    def _chunk_document(self, text, doc_id, metadata, timestamp) -> List[Dict[str, Any]]:
        """Split document into chunks for embedding"""
        # Simplified chunking implementation
        # In a real implementation, we would use a more sophisticated chunking strategy
        chunk_size = 1000
        overlap = 200
        
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            chunk_id = f"{doc_id}_{i}"
            
            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": chunk_id,
                    "parent_id": doc_id,
                    "chunk_index": i // (chunk_size - overlap)
                },
                "timestamp": timestamp
            })
            
        return chunks
    
    def _compute_embedding(self, text) -> List[float]:
        """Compute embedding for text using selected model"""
        # In a real implementation, this would call an embedding API
        # For now, return random values as placeholder
        import random
        return [random.random() for _ in range(self.vector_dimensions)]
    
    def _detect_anomalies(self, docs):
        """Detect anomalies in documents"""
        # Simplified anomaly detection
        # For invoices: check for unusual amounts or payment terms
        # For shipments: check for route deviations or unusual quantities
        
        # Filter documents by type
        invoices = docs.filter(pw.this.metadata["doc_type"] == "invoice")
        shipments = docs.filter(pw.this.metadata["doc_type"] == "shipment")
        
        # Invoice anomalies
        invoice_anomalies = invoices.select(
            id=f"anomaly_{pw.this.id}_{pw.this.timestamp}",
            document_id=pw.this.id,
            anomaly_type="invoice_amount_unusual",
            risk_score=0.8,  # Placeholder - would be calculated based on actual data
            description="Invoice amount significantly deviates from historical average",
            timestamp=pw.this.timestamp,
            metadata=pw.this.metadata
        ).filter(pw.this.risk_score > 0.7)  # Only keep high-risk anomalies
        
        # Shipment anomalies
        shipment_anomalies = shipments.select(
            id=f"anomaly_{pw.this.id}_{pw.this.timestamp}",
            document_id=pw.this.id,
            anomaly_type="shipment_route_deviation",
            risk_score=0.9,  # Placeholder
            description="Shipment route deviates significantly from expected path",
            timestamp=pw.this.timestamp,
            metadata=pw.this.metadata
        ).filter(pw.this.risk_score > 0.7)
        
        # Combine all anomalies
        all_anomalies = pw.concat(invoice_anomalies, shipment_anomalies)
        
        return all_anomalies