import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ..utils.document_processor import DocumentProcessor
from ..pipeline.enhanced_anomaly_detector import EnhancedAnomalyDetector
from ..models.rag_model import LogisticsPulseRAG
import logging

logger = logging.getLogger(__name__)

class DocumentChangeHandler(FileSystemEventHandler):
    def __init__(self, document_processor, anomaly_detector, rag_model, causal_engine):
        self.document_processor = document_processor
        self.anomaly_detector = anomaly_detector
        self.rag_model = rag_model
        self.causal_engine = causal_engine
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        logger.info(f"New file detected: {event.src_path}")
        self._process_file(event.src_path)
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        logger.info(f"File modified: {event.src_path}")
        self._process_file(event.src_path)
        
    def _process_file(self, file_path):
        """Process new or modified files"""
        # Determine file type based on directory or extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # Process based on file type
            if 'invoice' in file_path.lower() or '/invoices/' in file_path:
                logger.info(f"Processing invoice: {file_path}")
                data = self.document_processor.process_invoice(file_path)
                
                # Check for anomalies
                anomalies = self.anomaly_detector.detect_invoice_anomalies(data)
                if anomalies:
                    logger.info(f"Detected {len(anomalies)} invoice anomalies")
                    
                    # Update causal trace engine with new anomalies
                    for anomaly in anomalies:
                        self.causal_engine.register_event(
                            event_type="invoice_anomaly",
                            entity_id=data.get("invoice_id", "unknown"),
                            data=anomaly,
                            timestamp=time.time()
                        )
                    
                # Update RAG index
                self.rag_model.update_invoice_index(data)
                
                # Register event in causal engine
                self.causal_engine.register_event(
                    event_type="invoice_update",
                    entity_id=data.get("invoice_id", "unknown"),
                    data=data,
                    timestamp=time.time()
                )
                
            elif 'shipment' in file_path.lower() or '/shipments/' in file_path:
                logger.info(f"Processing shipment: {file_path}")
                data = self.document_processor.process_shipment(file_path)
                
                # Check for anomalies
                anomalies = self.anomaly_detector.detect_shipment_anomalies(data)
                if anomalies:
                    logger.info(f"Detected {len(anomalies)} shipment anomalies")
                    
                    # Update causal trace engine with new anomalies
                    for anomaly in anomalies:
                        self.causal_engine.register_event(
                            event_type="shipment_anomaly",
                            entity_id=data.get("shipment_id", "unknown"),
                            data=anomaly,
                            timestamp=time.time()
                        )
                
                # Update RAG index
                self.rag_model.update_shipment_index(data)
                
                # Register event in causal engine
                self.causal_engine.register_event(
                    event_type="shipment_update",
                    entity_id=data.get("shipment_id", "unknown"),
                    data=data,
                    timestamp=time.time()
                )
                
            elif 'policy' in file_path.lower() or '/policies/' in file_path:
                logger.info(f"Processing policy document: {file_path}")
                data = self.document_processor.process_policy(file_path)
                
                # Update RAG index
                self.rag_model.update_policy_index(data)
                
                # Register policy update in causal engine
                self.causal_engine.register_event(
                    event_type="policy_update",
                    entity_id=os.path.basename(file_path),
                    data=data,
                    timestamp=time.time()
                )
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            
class DataWatcher:
    def __init__(self, data_dir, document_processor, anomaly_detector, rag_model, causal_engine):
        self.data_dir = data_dir
        self.observer = Observer()
        self.handler = DocumentChangeHandler(document_processor, anomaly_detector, rag_model, causal_engine)
        
    def start(self):
        """Start watching data directories"""
        # Watch invoices directory
        invoices_dir = os.path.join(self.data_dir, "invoices")
        if os.path.exists(invoices_dir):
            self.observer.schedule(self.handler, invoices_dir, recursive=False)
            logger.info(f"Watching invoices directory: {invoices_dir}")
            
        # Watch shipments directory
        shipments_dir = os.path.join(self.data_dir, "shipments")
        if os.path.exists(shipments_dir):
            self.observer.schedule(self.handler, shipments_dir, recursive=False)
            logger.info(f"Watching shipments directory: {shipments_dir}")
            
        # Watch policies directory
        policies_dir = os.path.join(self.data_dir, "policies")
        if os.path.exists(policies_dir):
            self.observer.schedule(self.handler, policies_dir, recursive=False)
            logger.info(f"Watching policies directory: {policies_dir}")
            
        # Start the observer
        self.observer.start()
        logger.info("Data watcher started")
        
    def stop(self):
        """Stop watching data directories"""
        self.observer.stop()
        self.observer.join()
        logger.info("Data watcher stopped")