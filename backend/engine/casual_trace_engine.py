import time
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class CausalTraceEngine:
    """
    The Causal Trace Engine connects events across different entities and domains,
    identifying causal relationships between them based on temporal patterns and domain rules.
    """
    
    def __init__(self, llm=None):
        # Event log storing all registered events
        self.event_log = []
        
        # Entity relationships mapping
        self.entity_relationships = defaultdict(list)
        
        # Cache of recent causal analyses
        self.causal_analysis_cache = {}
        
        # LLM for advanced causal reasoning (optional)
        self.llm = llm
        
        # Domain-specific rules for causality
        self.causal_rules = self._initialize_causal_rules()
        
    def _initialize_causal_rules(self) -> Dict:
        """Initialize domain-specific rules for causal relationships"""
        return {
            # Invoice rules
            "invoice_delay": {
                "potential_causes": ["shipment_delay", "payment_policy_change", "driver_risk_increase"],
                "time_window": 86400 * 7,  # 7 days
                "strength_threshold": 0.6
            },
            
            # Shipment rules
            "shipment_anomaly": {
                "potential_causes": ["weather_event", "driver_risk_increase", "route_change"],
                "time_window": 86400 * 3,  # 3 days
                "strength_threshold": 0.7
            },
            
            # Payment rules
            "payment_term_violation": {
                "potential_causes": ["policy_update", "invoice_anomaly"],
                "time_window": 86400 * 5,  # 5 days
                "strength_threshold": 0.65
            }
        }
        
    def register_event(self, event_type: str, entity_id: str, data: Dict, timestamp: Optional[float] = None):
        """
        Register a new event in the system
        
        Args:
            event_type: Type of event (e.g., 'invoice_update', 'shipment_anomaly')
            entity_id: ID of the entity (e.g., invoice_id, shipment_id)
            data: Dictionary containing event data
            timestamp: Event timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Create the event record
        event = {
            "event_type": event_type,
            "entity_id": entity_id,
            "data": data,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat()
        }
        
        # Add to event log
        self.event_log.append(event)
        logger.info(f"Registered event: {event_type} for entity {entity_id}")
        
        # Update entity relationships based on event data
        self._update_entity_relationships(event)
        
        # Invalidate cached analyses for this entity
        self._invalidate_cache(entity_id)
        
        # Generate causal hypotheses if this is an anomaly event
        if "anomaly" in event_type:
            self._generate_causal_hypotheses(event)
            
    def _update_entity_relationships(self, event: Dict):
        """Update entity relationships based on event data"""
        entity_id = event["entity_id"]
        data = event["data"]
        
        # Extract relationships from data
        if event["event_type"] == "invoice_update":
            # Invoice is related to shipments mentioned in it
            if "shipment_id" in data:
                self._add_relationship(entity_id, data["shipment_id"], "invoice_to_shipment")
                
            # Invoice is related to supplier
            if "supplier" in data:
                self._add_relationship(entity_id, data["supplier"], "invoice_to_supplier")
                
        elif event["event_type"] == "shipment_update":
            # Shipment is related to carrier
            if "carrier" in data:
                self._add_relationship(entity_id, data["carrier"], "shipment_to_carrier")
                
            # Shipment is related to driver
            if "driver_id" in data:
                self._add_relationship(entity_id, data["driver_id"], "shipment_to_driver")
                
        elif event["event_type"] == "policy_update":
            # Policy might affect certain suppliers or carriers
            if "affected_entities" in data:
                for affected in data.get("affected_entities", []):
                    self._add_relationship(entity_id, affected, "policy_to_entity")
                    
    def _add_relationship(self, source_id: str, target_id: str, relationship_type: str):
        """Add a relationship between two entities"""
        self.entity_relationships[source_id].append({
            "target_id": target_id,
            "relationship_type": relationship_type,
            "created_at": time.time()
        })
        
        # Also add the reverse relationship for easier querying
        self.entity_relationships[target_id].append({
            "target_id": source_id,
            "relationship_type": f"{relationship_type}_reverse",
            "created_at": time.time()
        })
        
    def _invalidate_cache(self, entity_id: str):
        """Invalidate cached analyses for an entity and its related entities"""
        # Clear direct cache for this entity
        if entity_id in self.causal_analysis_cache:
            del self.causal_analysis_cache[entity_id]
            
        # Clear cache for related entities
        for related_entity in self.entity_relationships.get(entity_id, []):
            target_id = related_entity["target_id"]
            if target_id in self.causal_analysis_cache:
                del self.causal_analysis_cache[target_id]
                
    def _generate_causal_hypotheses(self, anomaly_event: Dict):
        """Generate causal hypotheses for an anomaly event"""
        event_type = anomaly_event["event_type"]
        entity_id = anomaly_event["entity_id"]
        
        # Extract anomaly type
        if "invoice_anomaly" in event_type:
            anomaly_type = anomaly_event["data"].get("anomaly_type", "unknown")
            base_type = "invoice_anomaly"
        elif "shipment_anomaly" in event_type:
            anomaly_type = anomaly_event["data"].get("anomaly_type", "unknown")
            base_type = "shipment_anomaly"
        else:
            return
            
        # Look for events that could have caused this anomaly
        timestamp = anomaly_event["timestamp"]
        potential_causes = self._find_potential_causes(entity_id, base_type, anomaly_type, timestamp)
        
        # Store the hypotheses
        if potential_causes:
            logger.info(f"Generated {len(potential_causes)} causal hypotheses for {entity_id}")
            
            # Store in cache
            self.causal_analysis_cache[entity_id] = {
                "anomaly": anomaly_event,
                "potential_causes": potential_causes,
                "generated_at": time.time()
            }
            
    def _find_potential_causes(self, entity_id: str, base_type: str, anomaly_type: str, 
                              timestamp: float) -> List[Dict]:
        """Find potential causes for an anomaly"""
        potential_causes = []
        
        # Get relevant time window from rules
        rule_key = None
        if base_type == "invoice_anomaly" and "delay" in anomaly_type:
            rule_key = "invoice_delay"
        elif base_type == "shipment_anomaly":
            rule_key = "shipment_anomaly"
        elif base_type == "invoice_anomaly" and "payment" in anomaly_type:
            rule_key = "payment_term_violation"
            
        if not rule_key or rule_key not in self.causal_rules:
            return potential_causes
            
        rule = self.causal_rules[rule_key]
        time_window = rule["time_window"]
        min_time = timestamp - time_window
        
        # Find directly related entities
        related_entities = [rel["target_id"] for rel in self.entity_relationships.get(entity_id, [])]
        
        # Search event log for potential causes
        for event in self.event_log:
            # Skip events that happened after the anomaly
            if event["timestamp"] > timestamp:
                continue
                
            # Skip events too old to be relevant
            if event["timestamp"] < min_time:
                continue
                
            # Skip the anomaly event itself
            if event["entity_id"] == entity_id and event["event_type"] == base_type:
                continue
                
            # Check if this event matches potential causes from rules
            event_matches_rule = False
            for potential_cause in rule["potential_causes"]:
                if potential_cause.lower() in event["event_type"].lower():
                    event_matches_rule = True
                    break
                    
            # If event matches rule or involves a related entity, it's a potential cause
            is_related_entity = event["entity_id"] in related_entities
            
            if event_matches_rule or is_related_entity:
                # Calculate causal strength based on temporal proximity and relationship
                temporal_score = 1.0 - ((timestamp - event["timestamp"]) / time_window)
                relationship_score = 1.0 if is_related_entity else 0.5
                causal_strength = (temporal_score + relationship_score) / 2
                
                # Only include if above threshold
                if causal_strength >= rule["strength_threshold"]:
                    potential_causes.append({
                        "event": event,
                        "causal_strength": causal_strength,
                        "temporal_proximity": f"{int((timestamp - event['timestamp']) / 60)} minutes",
                        "is_related_entity": is_related_entity
                    })
                    
        # Sort by causal strength
        potential_causes.sort(key=lambda x: x["causal_strength"], reverse=True)
        return potential_causes
        
    def analyze_entity(self, entity_id: str, entity_type: str = None) -> Dict:
        """
        Perform causal analysis for an entity
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity (invoice, shipment, etc.)
            
        Returns:
            Dictionary with analysis results
        """
        # Check cache first
        if entity_id in self.causal_analysis_cache:
            cached = self.causal_analysis_cache[entity_id]
            # Only use cache if fresh (less than 5 minutes old)
            if time.time() - cached["generated_at"] < 300:
                return cached
                
        # Find events related to this entity
        entity_events = [e for e in self.event_log if e["entity_id"] == entity_id]
        
        if not entity_events:
            return {
                "entity_id": entity_id,
                "found": False,
                "message": f"No events found for entity {entity_id}"
            }
            
        # Find the most recent anomaly for this entity
        anomalies = [e for e in entity_events if "anomaly" in e["event_type"]]
        
        if anomalies:
            # Sort by timestamp (newest first)
            anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
            latest_anomaly = anomalies[0]
            
            # Find potential causes
            anomaly_type = latest_anomaly["data"].get("anomaly_type", "unknown")
            base_type = "invoice_anomaly" if "invoice" in latest_anomaly["event_type"] else "shipment_anomaly"
            
            potential_causes = self._find_potential_causes(
                entity_id, 
                base_type, 
                anomaly_type, 
                latest_anomaly["timestamp"]
            )
            
            # Use LLM to generate narrative explanation if available
            narrative = self._generate_causal_narrative(latest_anomaly, potential_causes) if self.llm else None
            
            analysis = {
                "entity_id": entity_id,
                "found": True,
                "has_anomaly": True,
                "anomaly": latest_anomaly,
                "potential_causes": potential_causes,
                "narrative": narrative,
                "generated_at": time.time()
            }
            
            # Cache the result
            self.causal_analysis_cache[entity_id] = analysis
            return analysis
            
        else:
            # No anomalies, but entity exists
            return {
                "entity_id": entity_id,
                "found": True,
                "has_anomaly": False,
                "message": f"Entity {entity_id} exists but has no detected anomalies"
            }
            
    def _generate_causal_narrative(self, anomaly: Dict, causes: List[Dict]) -> str:
        """Generate a natural language explanation of causal relationships using LLM"""
        if not self.llm or not causes:
            return None
            
        try:
            # Create a prompt for the LLM
            entity_id = anomaly["entity_id"]
            anomaly_type = anomaly["data"].get("anomaly_type", "unknown")
            anomaly_desc = anomaly["data"].get("description", f"An {anomaly_type} anomaly")
            
            prompt = f"""
You are analyzing the root cause of a logistics anomaly.

ANOMALY DETAILS:
- Entity ID: {entity_id}
- Type: {anomaly_type}
- Description: {anomaly_desc}
- Detected at: {anomaly["datetime"]}

POTENTIAL CAUSES (in order of likelihood):
"""
            
            # Add each potential cause to the prompt
            for i, cause in enumerate(causes[:3]):  # Use top 3 causes
                cause_event = cause["event"]
                prompt += f"""
{i+1}. {cause_event["event_type"]} for {cause_event["entity_id"]} 
   - Occurred at: {cause_event["datetime"]}
   - Causal strength: {cause["causal_strength"]:.2f}
   - Temporal proximity: {cause["temporal_proximity"]}
   - Details: {str(cause_event["data"]).replace("{", "").replace("}", "")}
"""
            
            prompt += """
Please provide a clear, concise explanation of how these events likely caused the anomaly.
Focus on the temporal sequence and logical connections between events.
Start with "This anomaly was likely caused by..." and be specific about the causal chain.
"""

            # Call LLM for narrative generation
            response = self.llm.invoke(prompt)
            
            # Extract the narrative from response
            if hasattr(response, 'content'):
                narrative = response.content
            else:
                narrative = str(response)
                
            return narrative
            
        except Exception as e:
            logger.error(f"Error generating causal narrative: {e}")
            return f"Unable to generate narrative explanation due to error: {e}"
            
    def get_all_anomalies(self, limit: int = 10, entity_type: str = None) -> List[Dict]:
        """Get all anomalies, optionally filtered by entity type"""
        # Filter anomaly events
        anomalies = [e for e in self.event_log if "anomaly" in e["event_type"]]
        
        # Apply entity type filter if specified
        if entity_type:
            if entity_type == "invoice":
                anomalies = [e for e in anomalies if "invoice" in e["event_type"]]
            elif entity_type == "shipment":
                anomalies = [e for e in anomalies if "shipment" in e["event_type"]]
                
        # Sort by timestamp (newest first)
        anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Return limited number of results
        return anomalies[:limit]
        
    def get_related_events(self, entity_id: str, time_window: float = 86400) -> List[Dict]:
        """Get events related to an entity within a time window"""
        # Find all events for this entity
        entity_events = [e for e in self.event_log if e["entity_id"] == entity_id]
        
        if not entity_events:
            return []
            
        # Get related entity IDs
        related_ids = [rel["target_id"] for rel in self.entity_relationships.get(entity_id, [])]
        
        # Find events for related entities
        related_events = [
            e for e in self.event_log 
            if e["entity_id"] in related_ids
        ]
        
        # Combine and sort by timestamp
        all_events = entity_events + related_events
        all_events.sort(key=lambda x: x["timestamp"])
        
        return all_events