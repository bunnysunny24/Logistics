"""
CausalRAG Integration Module

This module integrates the CausalRAG components with the main application.
"""

import os
import logging
from typing import Optional

from .models.casual_rag_wrapper import CausalRAGWrapper
from .engine.casual_trace_engine import CausalTraceEngine
from .models.rag_model import LogisticsPulseRAG

logger = logging.getLogger(__name__)

def setup_causal_rag(rag_model: Optional[LogisticsPulseRAG] = None) -> Optional[CausalRAGWrapper]:
    """
    Set up the CausalRAG system by creating a wrapper around the existing RAG model.
    
    Args:
        rag_model: Existing RAG model instance or None to create a new one
        
    Returns:
        CausalRAGWrapper instance or None if setup fails
    """
    try:
        # Create a new RAG model if one isn't provided
        if rag_model is None:
            from .models.rag_model import LogisticsPulseRAG
            
            # Initialize the RAG model
            data_dir = os.getenv("DATA_DIR", "./data")
            index_dir = os.getenv("INDEX_DIR", "./data/index")
            rag_model = LogisticsPulseRAG(data_dir=data_dir, index_dir=index_dir)
        
        # Initialize the causal engine
        causal_engine = CausalTraceEngine(llm=rag_model.llm if hasattr(rag_model, 'llm') else None)
        
        # Create the causal wrapper
        causal_rag = CausalRAGWrapper(rag_model, causal_engine)
        
        logger.info("Successfully initialized Causal RAG system")
        return causal_rag
        
    except Exception as e:
        logger.error(f"Failed to initialize Causal RAG system: {e}")
        return None
