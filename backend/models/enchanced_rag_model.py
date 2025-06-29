# Import your existing rag_model.py code
from .rag_model import LogisticsPulseRAG
import logging

logger = logging.getLogger(__name__)

class CausalRAGModel(LogisticsPulseRAG):
    """
    Enhanced RAG model with causal reasoning capabilities
    """
    
    def __init__(self, data_dir=None, causal_engine=None, **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self.causal_engine = causal_engine
        
    def process_query(self, query, context=None):
        """Enhanced query processing with causal reasoning"""
        # Check if this is a causal query
        if self._is_causal_query(query):
            return self._process_causal_query(query, context)
        
        # Otherwise use standard RAG processing
        return super().process_query(query, context)
        
    def _is_causal_query(self, query):
        """Determine if a query is asking for causal reasoning"""
        causal_terms = [
            "why", "cause", "reason", "because", "due to", "resulted in",
            "led to", "root cause", "trigger", "caused by", "explain why",
            "what caused", "how did", "what led to"
        ]
        
        query_lower = query.lower()
        return any(term in query_lower for term in causal_terms)
        
    def _process_causal_query(self, query, context=None):
        """Process a query requiring causal reasoning"""
        if not self.causal_engine:
            # Fallback to standard processing if no causal engine
            return super().process_query(query, context)
            
        start_time = time.time()
        
        # Extract entity IDs from the query
        entity_ids = self._extract_entity_ids(query)
        
        if not entity_ids:
            # No specific entities mentioned, fallback to standard RAG
            return super().process_query(query, context)
            
        # Analyze each entity using causal engine
        causal_analyses = []
        for entity_id in entity_ids:
            analysis = self.causal_engine.analyze_entity(entity_id)
            if analysis["found"]:
                causal_analyses.append(analysis)
                
        if not causal_analyses:
            # No causal analyses found, fallback to standard RAG
            return super().process_query(query, context)
            
        # Combine causal analyses with standard RAG
        # First get standard response
        standard_response = super().process_query(query, context)
        
        # Then enhance with causal information
        enhanced_answer = self._create_causal_answer(query, causal_analyses, standard_response)
        
        return {
            "answer": enhanced_answer,
            "sources": standard_response.get("sources", []),
            "confidence": standard_response.get("confidence", 0.8),
            "metadata": {
                **standard_response.get("metadata", {}),
                "causal_reasoning": True,
                "entities_analyzed": len(causal_analyses),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        }
        
    def _extract_entity_ids(self, query):
        """Extract entity IDs (invoice numbers, shipment IDs) from query"""
        entity_ids = []
        
        # Look for invoice patterns (e.g., INV-2025-001, #234)
        invoice_patterns = [
            r'INV-\d{4}-\d{3,}',  # INV-2025-001
            r'invoice #?(\d+)',   # invoice #234 or invoice 234
            r'invoice (\w+-\w+-\w+)'  # invoice ABC-123-XYZ
        ]
        
        for pattern in invoice_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entity_ids.extend(matches)
                
        # Look for shipment patterns (e.g., SHP-2025-001, #12345)
        shipment_patterns = [
            r'SHP-\d{4}-\d{3,}',  # SHP-2025-001
            r'shipment #?(\d+)',  # shipment #12345 or shipment 12345
            r'shipment (\w+-\w+-\w+)'  # shipment ABC-123-XYZ
        ]
        
        for pattern in shipment_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entity_ids.extend(matches)
                
        return entity_ids
        
    def _create_causal_answer(self, query, causal_analyses, standard_response):
        """Create an enhanced answer incorporating causal reasoning"""
        standard_answer = standard_response.get("answer", "")
        
        causal_insights = []
        for analysis in causal_analyses:
            if analysis.get("has_anomaly", False) and analysis.get("narrative"):
                # Use the LLM-generated narrative
                causal_insights.append(analysis["narrative"])
            elif analysis.get("has_anomaly", False) and analysis.get("potential_causes"):
                # Create a structured explanation
                anomaly = analysis["anomaly"]
                causes = analysis["potential_causes"]
                
                insight = f"The {anomaly['data'].get('anomaly_type', 'anomaly')} for {anomaly['entity_id']} was likely caused by:\n"
                
                for i, cause in enumerate(causes[:2]):  # Top 2 causes
                    cause_event = cause["event"]
                    insight += f"- {cause_event['event_type']} ({cause['temporal_proximity']} before): {cause_event['data'].get('description', '')}\n"
                    
                causal_insights.append(insight)
                
        if causal_insights:
            # Combine standard answer with causal insights
            combined_answer = standard_answer + "\n\n### Root Cause Analysis\n\n" + "\n\n".join(causal_insights)
            return combined_answer
        else:
            return standard_answer
            
    def update_invoice_index(self, invoice_data):
        """Update the invoice index with new data"""
        try:
            # Create document text
            doc_text = self._format_invoice_document(invoice_data)
            
            # Create document with metadata
            doc = Document(
                page_content=doc_text,
                metadata={
                    "source": f"invoice_{invoice_data.get('invoice_id', 'unknown')}.csv",
                    "doc_type": "invoice",
                    "invoice_id": invoice_data.get("invoice_id", "unknown"),
                    "supplier": invoice_data.get("supplier", "unknown"),
                    "amount": invoice_data.get("amount", 0.0),
                    "timestamp": time.time()
                }
            )
            
            # Add to vector store
            if "invoice" in self.vector_stores:
                self.vector_stores["invoice"].add_documents([doc])
                logger.info(f"Updated invoice index with {invoice_data.get('invoice_id', 'unknown')}")
            else:
                # Create new vector store with this document
                self.vector_stores["invoice"] = FAISS.from_documents([doc], self.embeddings)
                logger.info(f"Created new invoice vector store with {invoice_data.get('invoice_id', 'unknown')}")
                
            # Update last checked time
            self.last_checked["invoice"] = time.time()
            
        except Exception as e:
            logger.error(f"Error updating invoice index: {e}")
            
    def update_shipment_index(self, shipment_data):
        """Update the shipment index with new data"""
        try:
            # Create document text
            doc_text = self._format_shipment_document(shipment_data)
            
            # Create document with metadata
            doc = Document(
                page_content=doc_text,
                metadata={
                    "source": f"shipment_{shipment_data.get('shipment_id', 'unknown')}.csv",
                    "doc_type": "shipment",
                    "shipment_id": shipment_data.get("shipment_id", "unknown"),
                    "origin": shipment_data.get("origin", "unknown"),
                    "destination": shipment_data.get("destination", "unknown"),
                    "carrier": shipment_data.get("carrier", "unknown"),
                    "status": shipment_data.get("status", "unknown"),
                    "risk_score": shipment_data.get("risk_score", 0.0),
                    "timestamp": time.time()
                }
            )
            
            # Add to vector store
            if "shipment" in self.vector_stores:
                self.vector_stores["shipment"].add_documents([doc])
                logger.info(f"Updated shipment index with {shipment_data.get('shipment_id', 'unknown')}")
            else:
                # Create new vector store with this document
                self.vector_stores["shipment"] = FAISS.from_documents([doc], self.embeddings)
                logger.info(f"Created new shipment vector store with {shipment_data.get('shipment_id', 'unknown')}")
                
            # Update last checked time
            self.last_checked["shipment"] = time.time()
            
        except Exception as e:
            logger.error(f"Error updating shipment index: {e}")
            
    def update_policy_index(self, policy_data):
        """Update the policy index with new data"""
        try:
            # Create document text
            if isinstance(policy_data, dict):
                doc_text = self._format_policy_document(policy_data)
            else:
                doc_text = str(policy_data)
                
            # Create document with metadata
            doc = Document(
                page_content=doc_text,
                metadata={
                    "source": f"policy_{int(time.time())}.pdf",
                    "doc_type": "policy",
                    "policy_id": policy_data.get("policy_id", f"policy_{int(time.time())}"),
                    "policy_type": policy_data.get("policy_type", "unknown"),
                    "timestamp": time.time()
                }
            )
            
            # Add to vector store
            if "policy" in self.vector_stores:
                self.vector_stores["policy"].add_documents([doc])
                logger.info(f"Updated policy index with {policy_data.get('policy_id', 'unknown')}")
            else:
                # Create new vector store with this document
                self.vector_stores["policy"] = FAISS.from_documents([doc], self.embeddings)
                logger.info(f"Created new policy vector store with {policy_data.get('policy_id', 'unknown')}")
                
            # Update last checked time
            self.last_checked["policy"] = time.time()
            
        except Exception as e:
            logger.error(f"Error updating policy index: {e}")