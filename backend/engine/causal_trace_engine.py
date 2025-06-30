import time
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json
import os
import re
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CausalLink:
    """A causal link between two events"""
    cause_id: str
    effect_id: str
    cause_type: str
    effect_type: str
    cause_description: str
    effect_description: str
    confidence: float
    evidence: List[str]
    timestamp: float

@dataclass
class CausalChain:
    """A chain of causal links forming a cause-and-effect pathway"""
    chain_id: str
    links: List[CausalLink]
    root_cause: str
    final_effect: str
    overall_confidence: float
    timestamp: float
    
@dataclass
class RiskBasedHold:
    """A hold placed on an entity based on risk analysis"""
    hold_id: str
    entity_id: str
    entity_type: str
    risk_score: float
    reason: str
    causal_chain_id: Optional[str]
    status: str  # "active", "pending_approval", "released", "rejected"
    created_at: float
    updated_at: float
    approved_by: Optional[str] = None

class CausalTraceEngine:
    """
    The Causal Trace Engine connects events across different entities and domains,
    identifying causal relationships between them based on temporal patterns and domain rules.
    """
    
    def __init__(self, llm=None, data_dir="./data"):
        # Event log storing all registered events
        self.event_log = []
        
        # Entity relationships mapping
        self.entity_relationships = defaultdict(list)
        
        # Cache of recent causal analyses
        self.causal_analysis_cache = {}
        
        # Active risk-based holds
        self.risk_holds = []
        
        # Data directory for persistence
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, "causal"), exist_ok=True)
        
        # LLM for advanced causal reasoning (optional)
        self.llm = llm
        
        # Domain-specific rules for causality
        self.causal_rules = self._initialize_causal_rules()
        
        # Load existing data
        self._load_persisted_data()
        
    def _initialize_causal_rules(self) -> Dict:
        """Initialize domain-specific rules for causal relationships"""
        return {
            
            # Shipment rules
            "shipment_delay": {
                "potential_causes": ["driver_risk_increase", "weather_event", "loading_issue"],
                "time_window": 86400 * 3,  # 3 days
                "strength_threshold": 0.7
            },
            
            # Shipment anomaly rules
            "shipment_anomaly": {
                "potential_causes": ["weather_event", "driver_risk_increase", "route_change"],
                "time_window": 86400 * 3,  # 3 days
                "strength_threshold": 0.7
            },
            
            # Payment rules
            "payment_policy_change": {
                "potential_effects": ["invoice_delay", "payment_hold"],
                "time_window": 86400 * 5,  # 5 days
                "strength_threshold": 0.8
            },
            
            # Payment term violation rules
            "payment_term_violation": {
                "potential_causes": ["policy_update", "invoice_anomaly"],
                "time_window": 86400 * 5,  # 5 days
                "strength_threshold": 0.65
            },
            
            # Driver risk rules
            "driver_risk_increase": {
                "potential_effects": ["shipment_delay", "invoice_delay", "risk_based_hold"],
                "time_window": 86400 * 2,  # 2 days
                "strength_threshold": 0.75
            },
            
            # Missing package rules
            "package_missing": {
                "potential_causes": ["loading_issue", "theft_event", "labeling_error"],
                "potential_effects": ["shipment_delay", "invoice_dispute"],
                "time_window": 86400 * 4,  # 4 days
                "strength_threshold": 0.65
            }
        }
    
    def _load_persisted_data(self):
        """Load persisted causal data"""
        try:
            # Load event log
            event_log_path = os.path.join(self.data_dir, "causal", "event_log.json")
            if os.path.exists(event_log_path):
                with open(event_log_path, 'r') as f:
                    self.event_log = json.load(f)
                    
            # Load risk holds
            risk_holds_path = os.path.join(self.data_dir, "causal", "risk_holds.json")
            if os.path.exists(risk_holds_path):
                with open(risk_holds_path, 'r') as f:
                    holds_data = json.load(f)
                    self.risk_holds = [RiskBasedHold(**h) for h in holds_data]
                    
            # Load entity relationships
            relationships_path = os.path.join(self.data_dir, "causal", "entity_relationships.json")
            if os.path.exists(relationships_path):
                with open(relationships_path, 'r') as f:
                    self.entity_relationships = defaultdict(list, json.load(f))
                    
            logger.info(f"Loaded {len(self.event_log)} events, {len(self.risk_holds)} risk holds")
        except Exception as e:
            logger.error(f"Error loading persisted causal data: {e}")
    
    def _persist_data(self):
        """Persist causal data to disk"""
        try:
            # Save event log
            event_log_path = os.path.join(self.data_dir, "causal", "event_log.json")
            with open(event_log_path, 'w') as f:
                json.dump(self.event_log, f)
                
            # Save risk holds
            risk_holds_path = os.path.join(self.data_dir, "causal", "risk_holds.json")
            with open(risk_holds_path, 'w') as f:
                json.dump([asdict(h) for h in self.risk_holds], f)
                
            # Save entity relationships
            relationships_path = os.path.join(self.data_dir, "causal", "entity_relationships.json")
            with open(relationships_path, 'w') as f:
                json.dump(dict(self.entity_relationships), f)
        except Exception as e:
            logger.error(f"Error persisting causal data: {e}")
            
    def register_event(self, event_type: str, entity_id: str, entity_type: str, 
                      description: str, metadata: Dict[str, Any] = None) -> str:
        """
        Register a new event in the system
        
        Args:
            event_type: Type of event (e.g., "shipment_delay", "driver_risk_increase")
            entity_id: ID of the entity associated with the event
            entity_type: Type of entity (e.g., "shipment", "driver", "invoice")
            description: Human-readable description of the event
            metadata: Additional data about the event
            
        Returns:
            event_id: Unique ID for the registered event
        """
        # Create event record
        event_id = f"{entity_type}_{entity_id}_{event_type}_{int(time.time())}"
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "description": description,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        # Add to event log
        self.event_log.append(event)
        
        # Add to entity relationships
        self.entity_relationships[f"{entity_type}:{entity_id}"].append(event_id)
        
        # Process event for potential causal relationships
        self._process_event_causality(event)
        
        # Persist data
        self._persist_data()
        
        return event_id
        
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
    
    def _process_event_causality(self, event: Dict[str, Any]):
        """Process a new event for potential causal relationships"""
        event_type = event["event_type"]
        event_time = event["timestamp"]
        
        # Check if this event could be an effect of previous events
        if event_type in self.causal_rules and "potential_causes" in self.causal_rules[event_type]:
            potential_causes = self.causal_rules[event_type]["potential_causes"]
            time_window = self.causal_rules[event_type]["time_window"]
            threshold = self.causal_rules[event_type]["strength_threshold"]
            
            # Find potential cause events within the time window
            for past_event in self.event_log:
                if past_event["event_id"] == event["event_id"]:
                    continue  # Skip the current event
                    
                # Check if this past event is a potential cause
                if past_event["event_type"] in potential_causes:
                    # Check if it's within the time window
                    if event_time - past_event["timestamp"] <= time_window:
                        # Calculate causality strength
                        strength = self._calculate_causality_strength(past_event, event)
                        
                        if strength >= threshold:
                            # Create causal link
                            self._create_causal_link(past_event, event, strength)
        
        # Check if this event could be a cause of previous events
        if event_type in self.causal_rules and "potential_effects" in self.causal_rules[event_type]:
            potential_effects = self.causal_rules[event_type]["potential_effects"]
            time_window = self.causal_rules[event_type]["time_window"]
            threshold = self.causal_rules[event_type]["strength_threshold"]
            
            # Find potential effect events within the time window
            for past_event in self.event_log:
                if past_event["event_id"] == event["event_id"]:
                    continue  # Skip the current event
                    
                # Check if this past event is a potential effect
                if past_event["event_type"] in potential_effects:
                    # Check if it's within the time window
                    if past_event["timestamp"] - event_time <= time_window:
                        # Calculate causality strength
                        strength = self._calculate_causality_strength(event, past_event)
                        
                        if strength >= threshold:
                            # Create causal link
                            self._create_causal_link(event, past_event, strength)
    
    def _calculate_causality_strength(self, cause_event: Dict[str, Any], effect_event: Dict[str, Any]) -> float:
        """
        Calculate the strength of a causal relationship between two events
        
        Factors:
        1. Temporal proximity - closer in time = stronger
        2. Entity relationships - same or related entities = stronger
        3. Domain-specific rules - e.g., driver risk increase -> shipment delay
        
        Returns:
            float: Causality strength score between 0-1
        """
        # Base score
        score = 0.5
        
        # Factor 1: Temporal proximity
        time_diff = effect_event["timestamp"] - cause_event["timestamp"]
        if time_diff <= 3600:  # Within an hour
            score += 0.3
        elif time_diff <= 86400:  # Within a day
            score += 0.2
        elif time_diff <= 86400 * 3:  # Within 3 days
            score += 0.1
            
        # Factor 2: Entity relationships
        if cause_event["entity_id"] == effect_event["entity_id"]:
            score += 0.2
        
        # Factor 3: Domain-specific boosts
        causal_pairs = {
            "driver_risk_increase": {
                "shipment_delay": 0.15,
                "risk_based_hold": 0.25
            },
            "payment_policy_change": {
                "invoice_delay": 0.2
            },
            "package_missing": {
                "shipment_delay": 0.25
            }
        }
        
        boost = causal_pairs.get(cause_event["event_type"], {}).get(effect_event["event_type"], 0)
        score += boost
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _create_causal_link(self, cause_event: Dict[str, Any], effect_event: Dict[str, Any], confidence: float):
        """Create a causal link between two events and add it to the system"""
        # Generate evidence
        evidence = [
            f"Temporal sequence: {cause_event['event_type']} occurred {int(effect_event['timestamp'] - cause_event['timestamp'])} seconds before {effect_event['event_type']}",
            f"Entity relationship: {cause_event['entity_type']} {cause_event['entity_id']} related to {effect_event['entity_type']} {effect_event['entity_id']}"
        ]
        
        # Add domain-specific evidence
        if cause_event["event_type"] == "driver_risk_increase" and effect_event["event_type"] == "shipment_delay":
            evidence.append(f"Domain rule: Driver risk increases are known to cause shipment delays (confidence: {confidence:.2f})")
        
        # Create the causal link
        causal_link = CausalLink(
            cause_id=cause_event["event_id"],
            effect_id=effect_event["event_id"],
            cause_type=cause_event["event_type"],
            effect_type=effect_event["event_type"],
            cause_description=cause_event["description"],
            effect_description=effect_event["description"],
            confidence=confidence,
            evidence=evidence,
            timestamp=time.time()
        )
        
        # Store the link in both events' metadata
        for event in self.event_log:
            if event["event_id"] == cause_event["event_id"]:
                if "causal_links" not in event["metadata"]:
                    event["metadata"]["causal_links"] = []
                event["metadata"]["causal_links"].append({
                    "link_type": "cause",
                    "related_event_id": effect_event["event_id"]
                })
                
            if event["event_id"] == effect_event["event_id"]:
                if "causal_links" not in event["metadata"]:
                    event["metadata"]["causal_links"] = []
                event["metadata"]["causal_links"].append({
                    "link_type": "effect",
                    "related_event_id": cause_event["event_id"]
                })
        
        # Check if this should trigger a risk-based hold
        self._check_for_risk_based_hold(causal_link)
        
        # Persist changes
        self._persist_data()
    
    def _check_for_risk_based_hold(self, causal_link: CausalLink):
        """Check if a causal link should trigger a risk-based hold"""
        # High-risk causal chains that should trigger holds
        high_risk_patterns = [
            ("driver_risk_increase", "shipment_delay", 0.8),  # Driver risk -> Shipment delay with high confidence
            ("package_missing", "invoice_dispute", 0.75),  # Missing package -> Invoice dispute
            ("payment_policy_change", "invoice_delay", 0.85)  # Policy change -> Invoice delay
        ]
        
        # Check against patterns
        for cause_type, effect_type, threshold in high_risk_patterns:
            if (causal_link.cause_type == cause_type and 
                causal_link.effect_type == effect_type and 
                causal_link.confidence >= threshold):
                
                # Extract entity info from the effect
                for event in self.event_log:
                    if event["event_id"] == causal_link.effect_id:
                        entity_id = event["entity_id"]
                        entity_type = event["entity_type"]
                        
                        # Create a risk-based hold
                        hold = RiskBasedHold(
                            hold_id=f"hold_{int(time.time())}_{entity_id}",
                            entity_id=entity_id,
                            entity_type=entity_type,
                            risk_score=causal_link.confidence,
                            reason=f"Automatic hold due to {causal_link.cause_type} causing {causal_link.effect_type}",
                            causal_chain_id=causal_link.cause_id,  # Reference the causal chain
                            status="pending_approval",
                            created_at=time.time(),
                            updated_at=time.time()
                        )
                        
                        # Add to holds
                        self.risk_holds.append(hold)
                        logger.info(f"Created risk-based hold: {hold.hold_id} for {entity_type} {entity_id}")
                        break
    
    def get_causal_chains(self, entity_id: Optional[str] = None, entity_type: Optional[str] = None, 
                       event_type: Optional[str] = None, limit: int = 10) -> List[CausalChain]:
        """
        Get causal chains matching the specified criteria
        
        Args:
            entity_id: Optional filter by entity ID
            entity_type: Optional filter by entity type
            event_type: Optional filter by event type
            limit: Maximum number of chains to return
            
        Returns:
            List of CausalChain objects
        """
        # Start by finding all events that match criteria
        matching_events = []
        for event in self.event_log:
            if entity_id and event["entity_id"] != entity_id:
                continue
                
            if entity_type and event["entity_type"] != entity_type:
                continue
                
            if event_type and event["event_type"] != event_type:
                continue
                
            matching_events.append(event)
        
        # Extract causal chains from matching events
        causal_chains = []
        processed_roots = set()
        
        for event in matching_events:
            # Skip if no causal links
            if "causal_links" not in event["metadata"]:
                continue
                
            # Find root causes (events that are causes but not effects)
            is_root = False
            for link in event["metadata"]["causal_links"]:
                if link["link_type"] == "cause":
                    # Check if this event is only a cause, not an effect
                    is_effect = False
                    for l in event["metadata"]["causal_links"]:
                        if l["link_type"] == "effect":
                            is_effect = True
                            break
                    
                    if not is_effect:
                        is_root = True
                        break
            
            if is_root:
                # Skip if we've already processed this root
                if event["event_id"] in processed_roots:
                    continue
                    
                processed_roots.add(event["event_id"])
                
                # Build chain from this root
                chain = self._build_causal_chain(event)
                if chain:
                    causal_chains.append(chain)
        
        # Sort by timestamp (newest first) and limit
        causal_chains.sort(key=lambda x: x.timestamp, reverse=True)
        return causal_chains[:limit]
    
    def _build_causal_chain(self, root_event: Dict[str, Any]) -> Optional[CausalChain]:
        """Build a causal chain starting from a root event"""
        # Find all events in the chain
        chain_events = [root_event]
        links = []
        
        # Track visited events to avoid cycles
        visited = {root_event["event_id"]}
        
        # Queue for BFS traversal
        queue = [(root_event, None)]  # (event, parent)
        
        while queue:
            current, parent = queue.pop(0)
            
            # Create link if there's a parent
            if parent:
                # Find the actual link data
                confidence = 0.7  # Default confidence
                evidence = []
                
                # Look for parent -> current link evidence
                for link_info in parent["metadata"].get("causal_links", []):
                    if link_info["link_type"] == "cause" and link_info["related_event_id"] == current["event_id"]:
                        # Found the link, extract details
                        # This is simplified - in a real system, link details would be stored
                        confidence = 0.7
                        evidence = [f"Causal relationship: {parent['event_type']} -> {current['event_type']}"]
                
                link = CausalLink(
                    cause_id=parent["event_id"],
                    effect_id=current["event_id"],
                    cause_type=parent["event_type"],
                    effect_type=current["event_type"],
                    cause_description=parent["description"],
                    effect_description=current["description"],
                    confidence=confidence,
                    evidence=evidence,
                    timestamp=current["timestamp"]
                )
                links.append(link)
            
            # Add children to queue
            if "causal_links" in current["metadata"]:
                for link_info in current["metadata"]["causal_links"]:
                    if link_info["link_type"] == "cause":
                        # This event is a cause for another event
                        effect_id = link_info["related_event_id"]
                        
                        if effect_id not in visited:
                            visited.add(effect_id)
                            
                            # Find the effect event
                            for event in self.event_log:
                                if event["event_id"] == effect_id:
                                    chain_events.append(event)
                                    queue.append((event, current))
                                    break
        
        if not links:
            return None  # No chain found
            
        # Calculate chain confidence (average of link confidences)
        overall_confidence = sum(link.confidence for link in links) / len(links)
        
        # Find the leaf event (no outgoing causal links)
        leaf_events = []
        for event in chain_events:
            is_leaf = True
            if "causal_links" in event["metadata"]:
                for link in event["metadata"]["causal_links"]:
                    if link["link_type"] == "cause":
                        is_leaf = False
                        break
            
            if is_leaf:
                leaf_events.append(event)
        
        # Use the most recent leaf if multiple
        final_event = max(leaf_events, key=lambda x: x["timestamp"]) if leaf_events else chain_events[-1]
        
        # Create the chain
        chain = CausalChain(
            chain_id=f"chain_{int(time.time())}_{root_event['event_id']}",
            links=links,
            root_cause=root_event["description"],
            final_effect=final_event["description"],
            overall_confidence=overall_confidence,
            timestamp=time.time()
        )
        
        return chain
    
    def get_risk_holds(self, entity_id: Optional[str] = None, entity_type: Optional[str] = None, 
                       status: Optional[str] = None) -> List[RiskBasedHold]:
        """
        Get risk-based holds matching criteria
        
        Args:
            entity_id: Optional filter by entity ID
            entity_type: Optional filter by entity type
            status: Optional filter by status
            
        Returns:
            List of RiskBasedHold objects
        """
        result = []
        
        for hold in self.risk_holds:
            if entity_id and hold.entity_id != entity_id:
                continue
                
            if entity_type and hold.entity_type != entity_type:
                continue
                
            if status and hold.status != status:
                continue
                
            result.append(hold)
            
        # Sort by creation time (newest first)
        result.sort(key=lambda x: x.created_at, reverse=True)
        return result
    
    def update_hold_status(self, hold_id: str, status: str, approved_by: Optional[str] = None) -> bool:
        """
        Update the status of a risk-based hold
        
        Args:
            hold_id: ID of the hold to update
            status: New status ("active", "released", "rejected")
            approved_by: User who approved the status change
            
        Returns:
            True if successful, False otherwise
        """
        for i, hold in enumerate(self.risk_holds):
            if hold.hold_id == hold_id:
                self.risk_holds[i].status = status
                self.risk_holds[i].updated_at = time.time()
                
                if approved_by:
                    self.risk_holds[i].approved_by = approved_by
                    
                # Persist changes
                self._persist_data()
                return True
                
        return False
    
    def find_related_events(self, entity_id: str, entity_type: str, 
                          max_time_window: int = 86400 * 7) -> List[Dict[str, Any]]:
        """
        Find events related to a specific entity within a time window
        
        Args:
            entity_id: Entity ID to find related events for
            entity_type: Type of entity
            max_time_window: Maximum time window to look back in seconds
            
        Returns:
            List of related events
        """
        # Get events for this entity
        entity_key = f"{entity_type}:{entity_id}"
        entity_event_ids = self.entity_relationships.get(entity_key, [])
        
        if not entity_event_ids:
            return []
            
        # Get all entity events
        entity_events = []
        for event in self.event_log:
            if event["event_id"] in entity_event_ids:
                entity_events.append(event)
                
        # Sort by timestamp (newest first)
        entity_events.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Use the most recent event as reference
        if not entity_events:
            return []
            
        reference_time = entity_events[0]["timestamp"]
        
        # Find related events within time window
        related_events = []
        
        for event in self.event_log:
            # Skip if it's the same entity
            if event["entity_id"] == entity_id and event["entity_type"] == entity_type:
                continue
                
            # Check time window
            if abs(event["timestamp"] - reference_time) <= max_time_window:
                related_events.append(event)
                
        return related_events
    
    # Dashboard-related methods
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the anomaly detection dashboard
        
        Returns:
            Dictionary with dashboard metrics including:
            - total_anomalies: Total count of anomalies
            - high_risk: Count of high risk anomalies
            - medium_risk: Count of medium risk anomalies
            - low_risk: Count of low risk anomalies
            - recent_anomalies: List of most recent anomalies
            - by_type: Breakdown of anomalies by type
            - last_updated: Timestamp of when metrics were calculated
        """
        # Get all anomalies
        all_anomalies = [e for e in self.event_log if "anomaly" in e["event_type"]]
        
        # Categorize anomalies by risk level
        high_risk = []
        medium_risk = []
        low_risk = []
        
        for anomaly in all_anomalies:
            risk_level = self._calculate_anomaly_risk(anomaly)
            if risk_level == "high":
                high_risk.append(anomaly)
            elif risk_level == "medium":
                medium_risk.append(anomaly)
            else:
                low_risk.append(anomaly)
        
        # Count anomalies by type
        by_type = defaultdict(int)
        for anomaly in all_anomalies:
            anomaly_type = anomaly["event_type"]
            by_type[anomaly_type] += 1
        
        # Get most recent anomalies
        all_anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
        recent_anomalies = all_anomalies[:10]  # Top 10 most recent
        
        # Calculate time-based metrics
        last_24h = time.time() - 86400
        anomalies_24h = [a for a in all_anomalies if a["timestamp"] > last_24h]
        
        return {
            "total_anomalies": len(all_anomalies),
            "high_risk": len(high_risk),
            "medium_risk": len(medium_risk),
            "low_risk": len(low_risk),
            "recent_anomalies": recent_anomalies,
            "by_type": dict(by_type),
            "last_24h": len(anomalies_24h),
            "last_updated": time.time()
        }
    
    def _calculate_anomaly_risk(self, anomaly: Dict[str, Any]) -> str:
        """
        Calculate the risk level of an anomaly
        
        Args:
            anomaly: The anomaly event dictionary
            
        Returns:
            Risk level as string: "high", "medium", or "low"
        """
        # Default to medium risk
        risk_level = "medium"
        
        # Extract anomaly type and entity type
        event_type = anomaly["event_type"]
        entity_type = anomaly.get("entity_type", "")
        
        # Check for high risk conditions
        high_risk_conditions = [
            # Driver risk conditions
            "driver_risk_increase" in event_type,
            # Invoice fraud conditions
            "invoice" in entity_type and "fraud" in event_type,
            # Shipment theft or loss
            "shipment" in entity_type and ("theft" in event_type or "missing" in event_type),
            # Policy violations
            "policy_violation" in event_type
        ]
        
        # Check for low risk conditions
        low_risk_conditions = [
            # Minor delays
            "delay" in event_type and anomaly.get("metadata", {}).get("delay_minutes", 120) < 30,
            # Small price discrepancies
            "price" in event_type and anomaly.get("metadata", {}).get("percentage_difference", 10) < 5
        ]
        
        # Determine risk level
        if any(high_risk_conditions):
            risk_level = "high"
        elif any(low_risk_conditions):
            risk_level = "low"
        
        # If we have causal links, adjust based on chain length/confidence
        if "causal_links" in anomaly.get("metadata", {}):
            causal_links = anomaly["metadata"]["causal_links"]
            
            # If many causal links, likely higher impact
            if len(causal_links) > 3:
                risk_level = "high"
            
            # Look for related high-risk events
            for link in causal_links:
                related_id = link.get("related_event_id")
                for event in self.event_log:
                    if event.get("event_id") == related_id and "driver_risk" in event.get("event_type", ""):
                        risk_level = "high"
                        break
        
        return risk_level
    
    def get_triggered_anomalies(self, since_timestamp: float) -> List[Dict[str, Any]]:
        """
        Get anomalies triggered since a specific timestamp
        
        Args:
            since_timestamp: Get anomalies detected after this timestamp
            
        Returns:
            List of anomaly events detected since the timestamp
        """
        # Filter anomalies by timestamp
        recent_anomalies = [
            e for e in self.event_log 
            if "anomaly" in e["event_type"] and e["timestamp"] > since_timestamp
        ]
        
        # Sort by timestamp (newest first)
        recent_anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return recent_anomalies
    
    def get_anomaly_trend(self, days: int = 7) -> Dict[str, List]:
        """
        Get anomaly trend data for visualization
        
        Args:
            days: Number of days to include in the trend
            
        Returns:
            Dictionary with trend data for visualization:
            - labels: List of date labels
            - total: Total anomalies per day
            - high_risk: High risk anomalies per day
            - medium_risk: Medium risk anomalies per day
            - low_risk: Low risk anomalies per day
        """
        # Calculate start timestamp (days ago from now)
        start_time = time.time() - (86400 * days)
        
        # Initialize data structure
        result = {
            "labels": [],
            "total": [],
            "high_risk": [],
            "medium_risk": [],
            "low_risk": []
        }
        
        # Create daily buckets
        for day in range(days):
            day_start = start_time + (86400 * day)
            day_end = day_start + 86400
            
            # Add label (date string)
            day_label = datetime.fromtimestamp(day_start).strftime('%m/%d')
            result["labels"].append(day_label)
            
            # Filter anomalies for this day
            day_anomalies = [
                e for e in self.event_log 
                if "anomaly" in e["event_type"] 
                and day_start <= e["timestamp"] < day_end
            ]
            
            # Count by risk level
            high = 0
            medium = 0
            low = 0
            
            for anomaly in day_anomalies:
                risk = self._calculate_anomaly_risk(anomaly)
                if risk == "high":
                    high += 1
                elif risk == "medium":
                    medium += 1
                else:
                    low += 1
            
            # Add counts to result
            result["total"].append(len(day_anomalies))
            result["high_risk"].append(high)
            result["medium_risk"].append(medium)
            result["low_risk"].append(low)
        
        return result

    def update_anomaly_risk(self, anomaly_id: str, new_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Update risk assessment for a specific anomaly with new data
        
        Args:
            anomaly_id: The ID of the anomaly to update
            new_data: Optional new data to add to the anomaly metadata
            
        Returns:
            Updated anomaly event or None if not found
        """
        # Find the anomaly
        for i, event in enumerate(self.event_log):
            if event.get("event_id") == anomaly_id and "anomaly" in event.get("event_type", ""):
                # Update with new data if provided
                if new_data:
                    self.event_log[i]["metadata"].update(new_data)
                
                # Recalculate risk based on updated information
                updated_risk = self._calculate_anomaly_risk(self.event_log[i])
                
                # Store risk level in metadata
                self.event_log[i]["metadata"]["risk_level"] = updated_risk
                self.event_log[i]["metadata"]["last_updated"] = time.time()
                
                # Persist changes
                self._persist_data()
                
                # Return updated anomaly
                return self.event_log[i]
        
        return None
    
    def simulate_anomaly_upload(self, anomaly_type: str, entity_type: str, count: int = 1, 
                            high_risk_ratio: float = 0.3) -> List[str]:
        """
        Simulate the upload of new anomaly data for demonstration purposes
        
        Args:
            anomaly_type: Type of anomaly to create (e.g., 'shipment_delay', 'invoice_fraud')
            entity_type: Type of entity for the anomaly (e.g., 'shipment', 'invoice')
            count: Number of anomalies to create
            high_risk_ratio: Ratio of anomalies that should be high risk
            
        Returns:
            List of created anomaly IDs
        """
        created_ids = []
        
        for i in range(count):
            # Generate random entity ID
            entity_id = f"{entity_type}_{int(time.time() % 10000) + i}"
            
            # Determine if this should be high risk
            is_high_risk = (i / count) < high_risk_ratio
            
            # Create metadata based on risk level
            if is_high_risk:
                if "delay" in anomaly_type:
                    metadata = {
                        "delay_minutes": 120 + (i * 10),  # Significant delay
                        "impact_level": "critical",
                        "affected_customers": 3 + (i % 5)
                    }
                    description = f"Critical {anomaly_type} affecting multiple customers"
                elif "fraud" in anomaly_type:
                    metadata = {
                        "fraud_indicators": ["unusual_pricing", "unauthorized_changes"],
                        "percentage_difference": 25 + (i % 15),
                        "previous_incidents": 2
                    }
                    description = f"Potential fraud detected with multiple indicators"
                else:
                    metadata = {
                        "severity": "high",
                        "impact": "significant",
                        "confidence": 0.85 + (i % 10) / 100
                    }
                    description = f"High severity {anomaly_type} detected"
            else:
                if "delay" in anomaly_type:
                    metadata = {
                        "delay_minutes": 15 + (i % 25),  # Minor delay
                        "impact_level": "low",
                        "affected_customers": 1
                    }
                    description = f"Minor {anomaly_type} with limited impact"
                elif "fraud" in anomaly_type:
                    metadata = {
                        "fraud_indicators": ["price_discrepancy"],
                        "percentage_difference": 3 + (i % 5),
                        "previous_incidents": 0
                    }
                    description = f"Potential anomaly detected for review"
                else:
                    metadata = {
                        "severity": "low",
                        "impact": "minimal",
                        "confidence": 0.6 + (i % 20) / 100
                    }
                    description = f"Low severity {anomaly_type} detected"
            
            # Register the event
            event_id = self.register_event(
                event_type=anomaly_type,
                entity_id=entity_id,
                entity_type=entity_type,
                description=description,
                metadata=metadata
            )
            
            created_ids.append(event_id)
            
            # Log the creation
            logger.info(f"Simulated anomaly created: {event_id}")
        
        return created_ids

    def reset_dashboard_data(self):
        """
        Reset dashboard data for demonstration purposes
        
        This method clears all anomaly events from the event log while preserving
        other event types and relationships.
        """
        # Filter out anomaly events
        self.event_log = [e for e in self.event_log if "anomaly" not in e["event_type"]]
        
        # Also clear risk holds since they're related to anomalies
        self.risk_holds = []
        
        # Clear the analysis cache
        self.causal_analysis_cache = {}
        
        # Persist the changes
        self._persist_data()
        
        logger.info(f"Dashboard data has been reset. Remaining events: {len(self.event_log)}")
        
        return {
            "status": "success",
            "remaining_events": len(self.event_log),
            "message": "Dashboard data has been reset successfully."
        }