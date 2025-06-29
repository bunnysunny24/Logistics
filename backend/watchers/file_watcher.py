import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class DocumentChangeHandler(FileSystemEventHandler):
    def __init__(self, document_processor, anomaly_detector, rag_model, causal_engine):
        self.document_processor = document_processor
        self.anomaly_detector = anomaly_detector
        self.rag_model = rag_model
        self.causal_engine = causal_engine
        self.processing_queue = set()  # Track files being processed
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        # Check file extension
        _, ext = os.path.splitext(event.src_path)
        if ext.lower() not in ['.csv', '.pdf', '.xlsx', '.json', '.md']:
            return
            
        logger.info(f"New file detected: {event.src_path}")
        self._process_file(event.src_path)
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Check file extension
        _, ext = os.path.splitext(event.src_path)
        if ext.lower() not in ['.csv', '.pdf', '.xlsx', '.json', '.md']:
            return
            
        # Avoid processing the same file multiple times
        if event.src_path in self.processing_queue:
            return
            
        logger.info(f"File modified: {event.src_path}")
        self._process_file(event.src_path)
        
    def _process_file(self, file_path):
        """Process new or modified files"""
        try:
            # Add to processing queue
            self.processing_queue.add(file_path)
            
            # Determine file type based on directory and extension
            file_name = os.path.basename(file_path)
            file_dir = os.path.dirname(file_path)
            _, ext = os.path.splitext(file_path)
            
            # Determine document type
            doc_type = None
            if '/invoices/' in file_path or 'invoice' in file_name.lower():
                doc_type = 'invoice'
            elif '/shipments/' in file_path or 'shipment' in file_name.lower():
                doc_type = 'shipment'
            elif '/policies/' in file_path or 'policy' in file_name.lower():
                doc_type = 'policy'
            elif 'driver' in file_name.lower() and 'risk' in file_name.lower():
                doc_type = 'driver_risk'
            else:
                # Try to infer from content for common types
                if ext.lower() == '.csv':
                    doc_type = self._infer_document_type_from_csv(file_path)
                elif ext.lower() == '.pdf':
                    doc_type = self._infer_document_type_from_pdf(file_path)
            
            # Process based on document type
            if doc_type == 'invoice':
                self._process_invoice(file_path)
            elif doc_type == 'shipment':
                self._process_shipment(file_path)
            elif doc_type == 'policy':
                self._process_policy(file_path)
            elif doc_type == 'driver_risk':
                self._process_driver_risk(file_path)
            else:
                logger.warning(f"Could not determine document type for {file_path}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        finally:
            # Remove from processing queue
            self.processing_queue.discard(file_path)
    
    def _process_invoice(self, file_path):
        """Process invoice file"""
        logger.info(f"Processing invoice: {file_path}")
        
        try:
            # Extract data using document processor
            data = self.document_processor.process_invoice(file_path)
            
            if not data:
                logger.warning(f"No data extracted from invoice {file_path}")
                return
                
            # Register event in causal engine
            self.causal_engine.register_event(
                event_type="invoice_update",
                entity_id=data.get("invoice_id", os.path.basename(file_path)),
                data=data,
                timestamp=time.time()
            )
            
            # Check for anomalies
            anomalies = self.anomaly_detector.detect_invoice_anomalies(data)
            
            if anomalies:
                logger.info(f"Detected {len(anomalies)} invoice anomalies")
                
                # Register anomalies in causal engine
                for anomaly in anomalies:
                    self.causal_engine.register_event(
                        event_type="invoice_anomaly",
                        entity_id=data.get("invoice_id", os.path.basename(file_path)),
                        data=anomaly,
                        timestamp=time.time()
                    )
            
            # Update RAG index
            self.rag_model.update_invoice_index(data)
            
            logger.info(f"Successfully processed invoice: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing invoice {file_path}: {e}")
    
    def _process_shipment(self, file_path):
        """Process shipment file"""
        logger.info(f"Processing shipment: {file_path}")
        
        try:
            # Extract data using document processor
            data = self.document_processor.process_shipment(file_path)
            
            if not data:
                logger.warning(f"No data extracted from shipment {file_path}")
                return
                
            # Register event in causal engine
            self.causal_engine.register_event(
                event_type="shipment_update",
                entity_id=data.get("shipment_id", os.path.basename(file_path)),
                data=data,
                timestamp=time.time()
            )
            
            # Check for anomalies
            anomalies = self.anomaly_detector.detect_shipment_anomalies(data)
            
            if anomalies:
                logger.info(f"Detected {len(anomalies)} shipment anomalies")
                
                # Register anomalies in causal engine
                for anomaly in anomalies:
                    self.causal_engine.register_event(
                        event_type="shipment_anomaly",
                        entity_id=data.get("shipment_id", os.path.basename(file_path)),
                        data=anomaly,
                        timestamp=time.time()
                    )
            
            # Update RAG index
            self.rag_model.update_shipment_index(data)
            
            logger.info(f"Successfully processed shipment: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing shipment {file_path}: {e}")
    
    def _process_policy(self, file_path):
        """Process policy file"""
        logger.info(f"Processing policy: {file_path}")
        
        try:
            # Extract data using document processor
            data = self.document_processor.process_policy(file_path)
            
            if not data:
                logger.warning(f"No data extracted from policy {file_path}")
                return
                
            # Register event in causal engine
            self.causal_engine.register_event(
                event_type="policy_update",
                entity_id=data.get("policy_id", os.path.basename(file_path)),
                data=data,
                timestamp=time.time()
            )
            
            # Update RAG index
            self.rag_model.update_policy_index(data)
            
            logger.info(f"Successfully processed policy: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing policy {file_path}: {e}")
    
    def _process_driver_risk(self, file_path):
        """Process driver risk update file"""
        logger.info(f"Processing driver risk update: {file_path}")
        
        try:
            # Extract data using document processor
            import json
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                logger.warning(f"No data extracted from driver risk update {file_path}")
                return
                
            # Register event in causal engine
            self.causal_engine.register_event(
                event_type="driver_risk_update",
                entity_id=data.get("driver_id", os.path.basename(file_path)),
                data=data,
                timestamp=time.time()
            )
            
            logger.info(f"Successfully processed driver risk update: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing driver risk update {file_path}: {e}")
    
    def _infer_document_type_from_csv(self, file_path):
        """Try to infer document type from CSV headers"""
        try:
            import pandas as pd
            
            df = pd.read_csv(file_path, nrows=1)
            columns = [col.lower() for col in df.columns]
            
            if any(col in columns for col in ['invoice', 'amount', 'supplier']):
                return 'invoice'
            elif any(col in columns for col in ['shipment', 'origin', 'destination', 'carrier']):
                return 'shipment'
            elif any(col in columns for col in ['driver', 'risk']):
                return 'driver_risk'
            
        except Exception:
            pass
            
        return None
    
    def _infer_document_type_from_pdf(self, file_path):
        """Try to infer document type from PDF content"""
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) == 0:
                    return None
                    
                text = pdf.pages[0].extract_text().lower()
                
                if any(word in text for word in ['invoice', 'bill', 'payment']):
                    return 'invoice'
                elif any(word in text for word in ['shipment', 'shipping', 'transport', 'delivery']):
                    return 'shipment'
                elif any(word in text for word in ['policy', 'guideline', 'procedure']):
                    return 'policy'
                
        except Exception:
            pass
            
        return None

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
            
        # Also watch root data directory for driver risk updates
        self.observer.schedule(self.handler, self.data_dir, recursive=False)
        logger.info(f"Watching data directory: {self.data_dir}")
            
        # Start the observer
        self.observer.start()
        logger.info("Data watcher started")
        
    def stop(self):
        """Stop watching data directories"""
        self.observer.stop()
        self.observer.join()
        logger.info("Data watcher stopped")