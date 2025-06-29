import logging
from typing import Dict, Any, Optional
from .casual_rag_implementation import CausalQueryHandler

logger = logging.getLogger(__name__)

class CausalRAGWrapper:
    """
    Wrapper for existing RAG model to add causal reasoning capabilities
    """
    
    def __init__(self, rag_model, causal_engine):
        """
        Initialize the wrapper
        
        Args:
            rag_model: Existing RAG model instance
            causal_engine: Causal Trace Engine instance
        """
        self.rag_model = rag_model
        self.causal_engine = causal_engine
        self.causal_handler = CausalQueryHandler(rag_model, causal_engine)
        
        # Add reference to causal engine in RAG model for updates
        if hasattr(rag_model, 'update_invoice_index'):
            self._wrap_update_methods()
    
    def _wrap_update_methods(self):
        """Wrap the update methods to register events in causal engine"""
        original_update_invoice = self.rag_model.update_invoice_index
        original_update_shipment = getattr(self.rag_model, 'update_shipment_index', None)
        original_update_policy = getattr(self.rag_model, 'update_policy_index', None)
        
        def wrapped_update_invoice(invoice_data):
            # Call original method
            result = original_update_invoice(invoice_data)
            
            # Register event in causal engine
            self.causal_engine.register_event(
                event_type="invoice_update",
                entity_id=invoice_data.get("invoice_id", "unknown"),
                data=invoice_data,
                timestamp=invoice_data.get("timestamp", None)
            )
            
            return result
        
        # Replace the method
        self.rag_model.update_invoice_index = wrapped_update_invoice
        
        # Do the same for shipment and policy if they exist
        if original_update_shipment:
            def wrapped_update_shipment(shipment_data):
                # Call original method
                result = original_update_shipment(shipment_data)
                
                # Register event in causal engine
                self.causal_engine.register_event(
                    event_type="shipment_update",
                    entity_id=shipment_data.get("shipment_id", "unknown"),
                    data=shipment_data,
                    timestamp=shipment_data.get("timestamp", None)
                )
                
                return result
            
            self.rag_model.update_shipment_index = wrapped_update_shipment
        
        if original_update_policy:
            def wrapped_update_policy(policy_data):
                # Call original method
                result = original_update_policy(policy_data)
                
                # Register event in causal engine
                self.causal_engine.register_event(
                    event_type="policy_update",
                    entity_id=policy_data.get("policy_id", "unknown"),
                    data=policy_data,
                    timestamp=policy_data.get("timestamp", None)
                )
                
                return result
            
            self.rag_model.update_policy_index = wrapped_update_policy
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a query with causal reasoning capabilities
        
        Args:
            query: The user's query
            context: Optional context information
            
        Returns:
            Response dictionary
        """
        return self.causal_handler.process_query(query, context)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped RAG model"""
        return getattr(self.rag_model, name)