from typing import List, Dict, Any, Optional
import os
import time
from datetime import datetime
import json
from loguru import logger
import threading

from pipeline.pathway_ingest import PathwayIngestPipeline
from utils.document_processor import DocumentProcessor

class PathwayManager:
    """
    Manager class for the Pathway pipeline
    Handles starting/stopping the pipeline and providing access to results
    """
    
    def __init__(self):
        self.pipeline = None
        self.watch_dir = os.environ.get("WATCH_DIR", "./data")
        self.embeddings_model = os.environ.get("EMBEDDINGS_MODEL", "openai")
        self.pipeline_thread = None
        self.running = False
        self.doc_processor = DocumentProcessor()
        
        # Create necessary directories
        os.makedirs(f"{self.watch_dir}/index", exist_ok=True)
        os.makedirs(f"{self.watch_dir}/anomalies", exist_ok=True)
        os.makedirs(f"{self.watch_dir}/stats", exist_ok=True)
    
    def start_pipeline(self):
        """Start the Pathway pipeline in a separate thread"""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        self.running = True
        self.pipeline = PathwayIngestPipeline(
            watch_dir=self.watch_dir,
            embeddings_model=self.embeddings_model
        )
        
        def run_pipeline():
            try:
                import pathway as pw
                self.pipeline.build_pipeline()
                pw.run()
            except Exception as e:
                logger.error(f"Error in Pathway pipeline: {e}")
                self.running = False
        
        self.pipeline_thread = threading.Thread(target=run_pipeline)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()
        logger.info("Pathway pipeline started in background thread")
    
    def stop_pipeline(self):
        """Stop the Pathway pipeline"""
        if not self.running:
            logger.warning("Pipeline not running")
            return
        
        # In a real implementation, we would gracefully stop the Pathway pipeline
        self.running = False
        # Wait for thread to complete
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=5)
        logger.info("Pathway pipeline stopped")
    
    def process_document(self, document_path: str, doc_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Process a document and add it to the watch directory
        """
        try:
            # Process the document based on its type
            processed_path = self.doc_processor.process_document(
                document_path, doc_type, metadata
            )
            logger.info(f"Document processed: {processed_path}")
            return processed_path
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {e}")
            raise
    
    def get_anomalies(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        min_risk_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get anomalies detected by the pipeline
        Optionally filter by date range and minimum risk score
        """
        anomalies_path = f"{self.watch_dir}/anomalies/detected.jsonl"
        
        if not os.path.exists(anomalies_path):
            return []
        
        anomalies = []
        try:
            with open(anomalies_path, 'r') as f:
                for line in f:
                    anomaly = json.loads(line)
                    
                    # Apply filters
                    if min_risk_score and anomaly["risk_score"] < min_risk_score:
                        continue
                    
                    # Convert timestamp to datetime for date filtering
                    anomaly_date = datetime.fromtimestamp(anomaly["timestamp"])
                    
                    if start_date:
                        start = datetime.fromisoformat(start_date)
                        if anomaly_date < start:
                            continue
                    
                    if end_date:
                        end = datetime.fromisoformat(end_date)
                        if anomaly_date > end:
                            continue
                    
                    anomalies.append(anomaly)
                    
            return anomalies
        except Exception as e:
            logger.error(f"Error retrieving anomalies: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the Pathway pipeline
        """
        return {
            "running": self.running,
            "watch_dir": self.watch_dir,
            "embeddings_model": self.embeddings_model,
            "anomalies_count": len(self.get_anomalies()),
            "last_update": self.get_last_update_time()
        }
    
    def get_document_count(self) -> Dict[str, int]:
        """
        Get the count of documents by type
        """
        stats_path = f"{self.watch_dir}/stats/document_stats.jsonl"
        
        if not os.path.exists(stats_path):
            return {"total": 0}
        
        try:
            counts = {"total": 0}
            with open(stats_path, 'r') as f:
                for line in f:
                    stat = json.loads(line)
                    doc_type = stat.get("doc_type", "unknown")
                    count = stat.get("count", 0)
                    counts[doc_type] = count
                    counts["total"] += count
            
            return counts
        except Exception as e:
            logger.error(f"Error retrieving document counts: {e}")
            return {"total": 0}
    
    def get_last_update_time(self) -> Optional[str]:
        """
        Get the timestamp of the most recent document update
        """
        stats_path = f"{self.watch_dir}/stats/document_stats.jsonl"
        
        if not os.path.exists(stats_path):
            return None
        
        try:
            latest_update = 0
            with open(stats_path, 'r') as f:
                for line in f:
                    stat = json.loads(line)
                    update_time = stat.get("latest_update", 0)
                    latest_update = max(latest_update, update_time)
            
            if latest_update > 0:
                return datetime.fromtimestamp(latest_update).isoformat()
            return None
        except Exception as e:
            logger.error(f"Error retrieving last update time: {e}")
            return None