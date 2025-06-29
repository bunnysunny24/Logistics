import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from loguru import logger

class LogisticsPulseRAG:
    def __init__(self):
        # Load environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.index_dir = os.getenv("INDEX_DIR", "./data/index")
        
        # Initialize components
        self.llm = OpenAI(temperature=0.2, model_name=self.llm_model)
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        
        # Load prompts
        self.load_prompts()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="query",
            output_key="result",
            return_messages=True
        )
        
        # Initialize vector store cache with timestamps
        self.vector_stores = {}
        self.last_checked = {}
        self.refresh_interval = 30  # seconds
        
    def load_prompts(self):
        """Load prompt templates from files"""
        self.prompt_templates = {
            "invoice": {
                "system": open("prompts/invoice_compliance_prompt.py").read(),
                "query": open("prompts/invoice_compliance_prompt.py").read(),
            },
            "shipment": {
                "system": open("prompts/shipment_anomaly_prompt.py").read(),
                "query": open("prompts/shipment_anomaly_prompt.py").read(),
            },
            "general": {
                "system": "You are LogisticsPulse, an AI assistant specialized in logistics operations.",
                "query": "Based on the retrieved context, answer the following query: {query}\n\nContext: {context}"
            }
        }
    
    def get_vector_store(self, doc_type):
        """Get vector store for a document type, refreshing if needed"""
        current_time = time.time()
        
        # Check if we need to refresh the vector store
        if (doc_type not in self.vector_stores or 
            doc_type not in self.last_checked or
            current_time - self.last_checked[doc_type] > self.refresh_interval):
            
            # Path to the vector store
            store_path = os.path.join(self.index_dir, doc_type)
            
            # Check if the vector store exists
            if os.path.exists(store_path):
                try:
                    # Load the vector store
                    self.vector_stores[doc_type] = FAISS.load_local(
                        store_path, self.embeddings
                    )
                    logger.info(f"Refreshed vector store for {doc_type}")
                except Exception as e:
                    logger.error(f"Error loading vector store for {doc_type}: {e}")
                    return None
            else:
                logger.warning(f"Vector store for {doc_type} does not exist")
                return None
            
            # Update the last checked time
            self.last_checked[doc_type] = current_time
        
        return self.vector_stores.get(doc_type)
    
    def detect_doc_types_from_query(self, query):
        """Detect document types relevant to the query"""
        query_lower = query.lower()
        
        doc_types = []
        
        # Check for invoice-related terms
        if any(term in query_lower for term in ["invoice", "payment", "discount", "late fee", "compliance", "due date"]):
            doc_types.append("invoice")
        
        # Check for shipment-related terms
        if any(term in query_lower for term in ["shipment", "route", "delivery", "tracking", "anomaly", "freight"]):
            doc_types.append("shipment")
        
        # Add policy if we're asking about rules
        if any(term in query_lower for term in ["policy", "rule", "guideline", "procedure", "protocol"]):
            doc_types.append("policy")
        
        # If no specific types detected, search all
        if not doc_types:
            doc_types = ["invoice", "shipment", "policy"]
        
        return doc_types
    
    def process_query(self, query, context=None):
        """Process a natural language query and return answer with sources"""
        # Initialize context if None
        if context is None:
            context = {}
        
        # Detect relevant document types
        doc_types = self.detect_doc_types_from_query(query)
        logger.info(f"Detected document types: {doc_types}")
        
        # Get combined results from all relevant document types
        combined_docs = []
        for doc_type in doc_types:
            vector_store = self.get_vector_store(doc_type)
            if vector_store:
                # Search for relevant documents
                docs = vector_store.similarity_search_with_score(query, k=5)
                # Filter out low relevance documents (score > 0.6)
                docs = [doc for doc, score in docs if score < 0.6]
                combined_docs.extend(docs)
        
        # If no relevant documents found
        if not combined_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing or ask about a different topic.",
                "sources": [],
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
        
        # Determine the primary document type
        primary_type = doc_types[0] if doc_types else "general"
        
        # Create prompt
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template=self.prompt_templates[primary_type]["query"]
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={
                "prompt": prompt_template,
                "memory": self.memory
            },
            return_source_documents=True
        )
        
        # Execute the chain
        result = qa_chain({"query": query})
        
        # Format sources
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Return formatted response
        return {
            "answer": result["result"],
            "sources": sources,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "doc_types": doc_types,
                "query_time_ms": context.get("query_time_ms", 0)
            }
        }
    
    def get_status(self):
        """Get status of the RAG model"""
        return {
            "model": self.llm_model,
            "embedding_model": self.embedding_model,
            "vector_stores": list(self.vector_stores.keys()),
            "last_checked": {k: datetime.fromtimestamp(v).isoformat() for k, v in self.last_checked.items()},
            "conversation_turns": len(self.memory.chat_memory.messages) // 2  # Each turn is a user message and AI response
        }