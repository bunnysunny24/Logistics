import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CausalQueryHandler:
    """
    Integrates causal reasoning into the RAG model's query processing
    """
    
    def __init__(self, rag_model, causal_engine):
        self.rag_model = rag_model
        self.causal_engine = causal_engine
        
        # Initialize risk-based hold system
        self.risk_thresholds = {
            "driver_risk": 0.7,
            "shipment_anomaly": 0.8,
            "invoice_anomaly": 0.75,
            "combined_risk": 0.6
        }
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a query with causal reasoning if applicable
        
        Args:
            query: The user's query
            context: Optional context information
            
        Returns:
            Response dictionary
        """
        # Check if this is a causal query
        if self._is_causal_query(query):
            # Extract entity IDs from the query
            entity_ids = self._extract_entity_ids(query)
            
            if entity_ids:
                # Process with causal reasoning
                return self._process_causal_query(query, entity_ids, context)
        
        # Default to standard RAG processing
        return self.rag_model.process_query(query, context)
    
    def _is_causal_query(self, query: str) -> bool:
        """
        Determine if a query is asking for causal reasoning
        
        Args:
            query: The user's query
            
        Returns:
            True if the query requires causal reasoning
        """
        causal_terms = [
            "why", "cause", "reason", "because", "due to", "resulted in",
            "led to", "root cause", "trigger", "caused by", "explain why",
            "what caused", "how did", "what led to"
        ]
        
        query_lower = query.lower()
        return any(term in query_lower for term in causal_terms)
    
    def _extract_entity_ids(self, query: str) -> List[str]:
        """
        Extract entity IDs (invoice numbers, shipment IDs) from query
        
        Args:
            query: The user's query
            
        Returns:
            List of entity IDs
        """
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
    
    def _process_causal_query(self, query: str, entity_ids: List[str], context: Optional[Dict] = None) -> Dict:
        """
        Process a query requiring causal reasoning
        
        Args:
            query: The user's query
            entity_ids: List of entity IDs extracted from the query
            context: Optional context information
            
        Returns:
            Response dictionary
        """
        # Get standard RAG response
        standard_response = self.rag_model.process_query(query, context)
        
        # Analyze each entity using causal engine
        causal_analyses = []
        for entity_id in entity_ids:
            analysis = self.causal_engine.analyze_entity(entity_id)
            if analysis["found"]:
                causal_analyses.append(analysis)
        
        if not causal_analyses:
            # No causal analyses found, return standard response
            return standard_response
        
        # Enhance the answer with causal information
        enhanced_answer = self._create_causal_answer(query, causal_analyses, standard_response)
        
        # Extract causal data for frontend display
        causal_chains = []
        risk_holds_data = []
        
        for analysis in causal_analyses:
            if analysis.get("has_anomaly", False):
                # Build causal chain data
                chain = {
                    "entity_id": analysis.get("entity_id"),
                    "anomaly_type": analysis.get("anomaly", {}).get("data", {}).get("anomaly_type", ""),
                    "risk_score": analysis.get("anomaly", {}).get("data", {}).get("risk_score", 0),
                    "causes": []
                }
                
                for cause in analysis.get("potential_causes", [])[:3]:
                    chain["causes"].append({
                        "event_type": cause["event"]["event_type"],
                        "entity_id": cause["event"]["entity_id"],
                        "timestamp": cause["event"]["timestamp"],
                        "description": cause["event"]["data"].get("description", ""),
                        "temporal_proximity": cause["temporal_proximity"]
                    })
                
                causal_chains.append(chain)
                
                # Check for risk holds
                hold_info = self._check_risk_based_holds(analysis)
                if hold_info:
                    risk_holds_data.append({
                        "entity_id": analysis.get("entity_id"),
                        "hold_type": "risk_based",
                        "reason": hold_info,
                        "risk_score": analysis.get("anomaly", {}).get("data", {}).get("risk_score", 0),
                        "timestamp": analysis.get("anomaly", {}).get("timestamp", 0)
                    })

        return {
            "answer": enhanced_answer,
            "sources": standard_response.get("sources", []),
            "confidence": standard_response.get("confidence", 0.8),
            "metadata": {
                **standard_response.get("metadata", {}),
                "causal_reasoning": True,
                "entities_analyzed": len(causal_analyses)
            },
            "causal_analysis": {
                "chains": causal_chains,
                "risk_holds": risk_holds_data,
                "has_causal_data": len(causal_chains) > 0
            }
        }
    
    def _create_causal_answer(self, query: str, causal_analyses: List[Dict], standard_response: Dict) -> str:
        """
        Create an enhanced answer incorporating causal reasoning
        
        Args:
            query: The user's query
            causal_analyses: List of causal analyses
            standard_response: Standard RAG response
            
        Returns:
            Enhanced answer incorporating causal reasoning
        """
        standard_answer = standard_response.get("answer", "")
        
        causal_insights = []
        risk_holds = []
        
        for analysis in causal_analyses:
            # Check for risk-based holds
            hold_info = self._check_risk_based_holds(analysis)
            if hold_info:
                risk_holds.append(hold_info)
            
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
        
        # Build enhanced response
        enhanced_parts = [standard_answer]
        
        if risk_holds:
            enhanced_parts.append("\n### 🚨 Risk-Based Holds Applied\n\n" + "\n\n".join(risk_holds))
        
        if causal_insights:
            enhanced_parts.append("\n### 🧠 Root Cause Analysis\n\n" + "\n\n".join(causal_insights))
        
        return "\n".join(enhanced_parts)
    
    def _check_risk_based_holds(self, analysis: Dict) -> Optional[str]:
        """
        Check if a risk-based hold should be applied based on causal analysis
        
        Args:
            analysis: The causal analysis result
            
        Returns:
            Hold information text if a hold should be applied, None otherwise
        """
        if not analysis.get("has_anomaly", False):
            return None
            
        entity_id = analysis.get("entity_id")
        anomaly = analysis.get("anomaly", {})
        anomaly_type = anomaly.get("data", {}).get("anomaly_type", "")
        
        # Extract risk scores
        risk_score = float(anomaly.get("data", {}).get("risk_score", 0))
        combined_risk = risk_score
        
        # Check related events for driver risk
        has_driver_risk = False
        driver_risk_score = 0
        driver_name = ""
        
        # Look for driver risk events in potential causes
        for cause in analysis.get("potential_causes", []):
            cause_event = cause.get("event", {})
            if cause_event.get("event_type") == "driver_risk_update":
                has_driver_risk = True
                driver_risk_score = float(cause_event.get("data", {}).get("risk_score", 0))
                driver_name = cause_event.get("data", {}).get("driver_name", "Unknown driver")
                combined_risk = max(combined_risk, driver_risk_score * 0.9)
                
        # Calculate final risk score with causal weighting
        if len(analysis.get("potential_causes", [])) > 0:
            # If we have causal connections, increase the risk
            combined_risk *= 1.2
            
        # Apply hold if threshold exceeded
        if "invoice" in anomaly_type.lower() and combined_risk >= self.risk_thresholds["invoice_anomaly"]:
            return f"PAYMENT HOLD on {entity_id}: Invoice flagged for manual review due to high risk score ({combined_risk:.2f})."
            
        elif "shipment" in anomaly_type.lower() and combined_risk >= self.risk_thresholds["shipment_anomaly"]:
            hold_reason = f"SHIPMENT HOLD on {entity_id}: Shipment flagged due to anomaly detection ({combined_risk:.2f})."
            if has_driver_risk:
                hold_reason += f" Driver {driver_name} has elevated risk score ({driver_risk_score:.2f})."
            return hold_reason
            
        elif has_driver_risk and driver_risk_score >= self.risk_thresholds["driver_risk"]:
            return f"DRIVER RESTRICTION: {driver_name} has high risk score ({driver_risk_score:.2f}). Recommend limiting sensitive shipments."
            
        elif combined_risk >= self.risk_thresholds["combined_risk"]:
            return f"ADVISORY HOLD on {entity_id}: Combined risk factors ({combined_risk:.2f}) suggest caution."
            
        return None