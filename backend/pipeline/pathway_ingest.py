import pathway as pw
import os
import time
from loguru import logger
from datetime import datetime
import json

class PathwayIngestPipeline:
    """
    Pathway pipeline for ingesting documents and detecting changes
    """
    
    def __init__(self, watch_dir, embeddings_model):
        self.watch_dir = watch_dir
        self.embeddings_model = embeddings_model
        self.input_dirs = {
            "invoice": f"{watch_dir}/invoices",
            "shipment": f"{watch_dir}/shipments",
            "policy": f"{watch_dir}/policies"
        }
        self.output_dir = f"{watch_dir}/index"
        self.anomalies_dir = f"{watch_dir}/anomalies"
        self.poll_interval_seconds = 5  # Check for changes every 5 seconds
    
    def build_pipeline(self):
        """Build the Pathway pipeline"""
        # Create input connectors for each document type
        input_connectors = {}
        for doc_type, dir_path in self.input_dirs.items():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            input_connectors[doc_type] = pw.io.fs.read(
                dir_path,
                format="csv",
                mode="streaming",
                poll_interval_seconds=self.poll_interval_seconds,
                with_metadata=True
            )
        
        # Special case for policy documents
        policy_connector = pw.io.fs.read(
            self.input_dirs["policy"],
            format="plaintext",
            mode="streaming",
            poll_interval_seconds=self.poll_interval_seconds,
            with_metadata=True
        )
        
        # Process each document type
        processed_tables = {}
        for doc_type, connector in input_connectors.items():
            processed = self._process_document_table(connector, doc_type)
            processed_tables[doc_type] = processed
        
        # Process policy documents
        policy_processed = self._process_policy_table(policy_connector)
        processed_tables["policy"] = policy_processed
        
        # Combine all processed documents
        all_docs = pw.Table.concat(*list(processed_tables.values()))
        
        # Index documents for retrieval
        self._index_documents(all_docs)
        
        # Run anomaly detection on relevant document types
        if "invoice" in processed_tables:
            self._detect_invoice_anomalies(processed_tables["invoice"])
        if "shipment" in processed_tables:
            self._detect_shipment_anomalies(processed_tables["shipment"])
    
    def _process_document_table(self, table, doc_type):
        """Process a document table based on its type"""
        # Add document type metadata
        table = table.select(
            pw.this.data,
            doc_type=pw.lit(doc_type),
            **pw.this.metadata
        )
        
        # Handle CSV data
        return table.select(
            # Extract content for embedding
            content=self._extract_content_for_doc_type(pw.this.data, doc_type),
            metadata=pw.declare_type(dict, pw.dict(
                doc_type=pw.this.doc_type,
                filename=pw.this.metadata.filename,
                last_modified=pw.this.metadata.last_modified,
                **pw.this.data
            ))
        )
    
    def _process_policy_table(self, table):
        """Process policy documents (markdown/text files)"""
        # Split documents into chunks
        return table.select(
            content=pw.this.data,
            metadata=pw.declare_type(dict, pw.dict(
                doc_type=pw.lit("policy"),
                filename=pw.this.metadata.filename,
                last_modified=pw.this.metadata.last_modified
            ))
        )
    
    def _extract_content_for_doc_type(self, data, doc_type):
        """Extract content from document data based on type"""
        if doc_type == "invoice":
            # For invoices, include key fields in the content
            return pw.apply(
                lambda x: f"Invoice {x.get('invoice_id', 'Unknown')}\n"
                         f"Supplier: {x.get('supplier', 'Unknown')}\n"
                         f"Amount: {x.get('amount', 'Unknown')}\n"
                         f"Issue Date: {x.get('issue_date', 'Unknown')}\n"
                         f"Due Date: {x.get('due_date', 'Unknown')}\n"
                         f"Payment Terms: {x.get('payment_terms', 'Unknown')}\n",
                data
            )
        elif doc_type == "shipment":
            # For shipments, include key fields in the content
            return pw.apply(
                lambda x: f"Shipment {x.get('shipment_id', 'Unknown')}\n"
                         f"Origin: {x.get('origin', 'Unknown')}\n"
                         f"Destination: {x.get('destination', 'Unknown')}\n"
                         f"Carrier: {x.get('carrier', 'Unknown')}\n"
                         f"Status: {x.get('status', 'Unknown')}\n",
                data
            )
        else:
            # Default extraction
            return pw.apply(str, data)
    
    def _index_documents(self, documents):
        """Index documents for retrieval"""
        # Write documents to the index directory
        documents.select(
            content=pw.this.content,
            metadata=pw.this.metadata
        ).with_universe_id().select(
            pw.this.universe_id,
            content=pw.this.content,
            metadata=pw.this.metadata
        ).write(
            pw.io.jsonlines.write(
                f"{self.output_dir}/chunks.jsonl",
                append=False  # Overwrite existing file
            )
        )
        
        # Update stats
        documents.groupby(pw.this.metadata.doc_type).reduce(
            doc_type=pw.this.metadata.doc_type,
            count=pw.reducers.count(),
            latest_update=pw.reducers.max(pw.apply(
                lambda x: int(time.time()),
                pw.this.metadata
            ))
        ).write(
            pw.io.jsonlines.write(
                f"{self.watch_dir}/stats/document_stats.jsonl",
                append=False  # Overwrite existing file
            )
        )
    
    def _detect_invoice_anomalies(self, invoices):
        """Detect anomalies in invoices"""
        # Implement anomaly detection logic
        # Example: Detect late payment issues
        anomalies = invoices.select(
            pw.this.content,
            data=pw.this.metadata,
        ).filter(
            lambda x: self._check_invoice_compliance(x.data)
        ).select(
            id=pw.apply(lambda x: f"ANM-INV-{int(time.time())}", pw.this.data),
            document_id=pw.this.data.invoice_id,
            anomaly_type=pw.lit("payment_terms_noncompliance"),
            description=pw.apply(self._generate_invoice_anomaly_description, pw.this.data),
            risk_score=pw.apply(lambda x: 0.8, pw.this.data),  # High risk for non-compliance
            timestamp=pw.apply(lambda x: int(time.time()), pw.this.data),
            metadata=pw.this.data
        )
        
        # Write anomalies to output
        self._write_anomalies(anomalies)
    
    def _detect_shipment_anomalies(self, shipments):
        """Detect anomalies in shipments"""
        # Implement anomaly detection logic
        # Example: Detect route deviations
        anomalies = shipments.select(
            pw.this.content,
            data=pw.this.metadata,
        ).filter(
            lambda x: self._check_shipment_anomalies(x.data)
        ).select(
            id=pw.apply(lambda x: f"ANM-SHP-{int(time.time())}", pw.this.data),
            document_id=pw.this.data.shipment_id,
            anomaly_type=pw.apply(self._determine_anomaly_type, pw.this.data),
            description=pw.apply(self._generate_shipment_anomaly_description, pw.this.data),
            risk_score=pw.apply(self._calculate_risk_score, pw.this.data),
            timestamp=pw.apply(lambda x: int(time.time()), pw.this.data),
            metadata=pw.this.data
        )
        
        # Write anomalies to output
        self._write_anomalies(anomalies)
    
    def _write_anomalies(self, anomalies):
        """Write anomalies to output file"""
        anomalies.write(
            pw.io.jsonlines.write(
                f"{self.anomalies_dir}/detected.jsonl",
                append=True  # Append to existing file
            )
        )
    
    # Helper methods for anomaly detection
    def _check_invoice_compliance(self, invoice_data):
        """Check if an invoice is compliant with payment terms"""
        try:
            payment_terms = invoice_data.get("payment_terms", "")
            if not payment_terms.startswith("NET"):
                return False
            
            expected_days = int(payment_terms.replace("NET", ""))
            issue_date = invoice_data.get("issue_date")
            due_date = invoice_data.get("due_date")
            
            if not (issue_date and due_date):
                return False
            
            # Calculate actual days
            from datetime import datetime
            issue_dt = datetime.fromisoformat(issue_date)
            due_dt = datetime.fromisoformat(due_date)
            actual_days = (due_dt - issue_dt).days
            
            # Check if there's a significant discrepancy
            return abs(actual_days - expected_days) > 2
        except:
            return False
    
    def _check_shipment_anomalies(self, shipment_data):
        """Check if a shipment has anomalies"""
        # Check for status-based anomalies
        if shipment_data.get("status") == "Delayed":
            return True
        
        # Check for risk score
        risk_score = shipment_data.get("risk_score")
        if risk_score and float(risk_score) > 0.7:
            return True
        
        # Check for anomaly type
        anomaly_type = shipment_data.get("anomaly_type")
        if anomaly_type and anomaly_type != "none":
            return True
        
        return False
    
    def _determine_anomaly_type(self, shipment_data):
        """Determine the type of shipment anomaly"""
        anomaly_type = shipment_data.get("anomaly_type")
        if anomaly_type and anomaly_type != "none":
            return anomaly_type
        
        if shipment_data.get("status") == "Delayed":
            return "timeline_delay"
        
        return "unknown_anomaly"
    
    def _generate_invoice_anomaly_description(self, invoice_data):
        """Generate a description for an invoice anomaly"""
        payment_terms = invoice_data.get("payment_terms", "Unknown")
        supplier = invoice_data.get("supplier", "Unknown")
        amount = invoice_data.get("amount", "Unknown")
        
        return f"Invoice from {supplier} for {amount} does not comply with {payment_terms} payment terms."
    
    def _generate_shipment_anomaly_description(self, shipment_data):
        """Generate a description for a shipment anomaly"""
        anomaly_type = self._determine_anomaly_type(shipment_data)
        shipment_id = shipment_data.get("shipment_id", "Unknown")
        origin = shipment_data.get("origin", "Unknown")
        destination = shipment_data.get("destination", "Unknown")
        
        if anomaly_type == "route_deviation":
            return f"Shipment {shipment_id} shows deviation from expected route between {origin} and {destination}."
        elif anomaly_type == "timeline_delay":
            return f"Shipment {shipment_id} from {origin} to {destination} is experiencing delays."
        elif anomaly_type == "value_discrepancy":
            return f"Shipment {shipment_id} has unusual value compared to historical data for this route."
        else:
            return f"Anomaly detected in shipment {shipment_id} from {origin} to {destination}."
    
    def _calculate_risk_score(self, shipment_data):
        """Calculate a risk score for a shipment anomaly"""
        # Use existing risk score if available
        if "risk_score" in shipment_data:
            try:
                return float(shipment_data["risk_score"])
            except:
                pass
        
        # Default scores based on anomaly type
        anomaly_type = self._determine_anomaly_type(shipment_data)
        if anomaly_type == "route_deviation":
            return 0.85
        elif anomaly_type == "timeline_delay":
            return 0.6
        elif anomaly_type == "value_discrepancy":
            return 0.9
        else:
            return 0.7