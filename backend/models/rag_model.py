from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

import os
import json
from typing import Dict, Any, List, Optional
from loguru import logger
from datetime import datetime

class LogisticsPulseRAG:
    """
    RAG model for Logistics Pulse Copilot
    Uses LangChain to create a retrieval-based Q&A system
    """
    
    def __init__(self):
        # Initialize configuration from environment variables
        self.index_dir = os.environ.get("INDEX_DIR", "./data/index")
        self.prompts_dir = os.environ.get("PROMPTS_DIR", "./prompts")
        
        # Initialize embedding model
        self.embeddings = self._initialize_embeddings()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize vector store and retriever
        self.vector_store = self._initialize_vector_store()
        self.retriever = self._initialize_retriever()
        
        # Initialize QA chain
        self.qa_chain = self._initialize_qa_chain()
        
        # Track usage statistics
        self.query_count = 0
        self.last_query_time = None
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        # Use OpenAI embeddings by default
        embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not set. Using fake embeddings for demo purposes.")
            # In a real implementation, you might use a local embedding model
            # For now, we'll just return a simple embedding function
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=1536)
        
        return OpenAIEmbeddings(model=embedding_model)
    
    def _initialize_llm(self):
        """Initialize the language model"""
        # Use ChatGPT by default
        model_name = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not set. Using fake LLM for demo purposes.")
            # In a real implementation, you might use a local LLM
            # For now, we'll just return a simple LLM that echoes the prompt
            from langchain_community.llms import FakeListLLM
            return FakeListLLM(responses=["This is a demo response from the Logistics Pulse Copilot."])
        
        return ChatOpenAI(model_name=model_name, temperature=0.2)
    
    def _initialize_vector_store(self):
        """Initialize the vector store from saved index"""
        # Check if index exists
        if not os.path.exists(f"{self.index_dir}/chunks.jsonl"):
            logger.warning(f"Index file not found at {self.index_dir}/chunks.jsonl")
            # Create an empty vector store
            return FAISS.from_texts(["Placeholder document"], self.embeddings)
        
        # Load documents from the index
        documents = []
        try:
            with open(f"{self.index_dir}/chunks.jsonl", 'r') as f:
                for line in f:
                    data = json.loads(line)
                    doc = Document(
                        page_content=data.get("content", ""),
                        metadata=data.get("metadata", {})
                    )
                    documents.append(doc)
            
            # Create vector store from documents
            return FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Create an empty vector store
            return FAISS.from_texts(["Error loading documents"], self.embeddings)
    
    def _initialize_retriever(self):
        """Initialize the retriever with contextual compression"""
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Add contextual compression to improve relevance
        compressor = LLMChainExtractor.from_llm(self.llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    def _get_prompt_template(self, query_type=None):
        """Get the appropriate prompt template based on query type"""
        if not query_type:
            # Try to infer query type from query content
            query_type = "general"
        
        # Check if prompt template file exists
        template_path = f"{self.prompts_dir}/{query_type}_prompt.txt"
        if not os.path.exists(template_path):
            logger.warning(f"Prompt template not found: {template_path}")
            # Use default template
            template = """
            You are Logistics Pulse Copilot, an AI assistant for logistics and finance professionals.
            
            Context information is below:
            ----------------
            {context}
            ----------------
            
            Given the context information and not prior knowledge, answer the question: {question}
            
            Provide a detailed response that directly addresses the question. If the answer cannot be found in the context,
            say "I don't have enough information to answer this question." Do not make up information.
            
            For compliance questions, cite specific clauses or policies.
            For anomaly questions, explain the risk factors and reasons for flagging.
            """
            return PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        
        # Load template from file
        try:
            with open(template_path, 'r') as f:
                template = f.read()
            
            return PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            # Use simple template
            template = "{context}\n\nQuestion: {question}\n\nAnswer:"
            return PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
    
    def _initialize_qa_chain(self):
        """Initialize the QA chain"""
        prompt = self._get_prompt_template()
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            },
            return_source_documents=True
        )
    
    def _determine_query_type(self, query):
        """Determine the type of query to use the appropriate prompt template"""
        query_lower = query.lower()
        
        if "invoice" in query_lower or "payment" in query_lower or "fee" in query_lower or "due date" in query_lower:
            return "invoice"
        elif "shipment" in query_lower or "route" in query_lower or "delivery" in query_lower or "anomaly" in query_lower:
            return "shipment"
        else:
            return "general"
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a natural language query and return the answer with sources
        """
        try:
            # Update statistics
            self.query_count += 1
            self.last_query_time = datetime.now().isoformat()
            
            # Determine query type
            query_type = self._determine_query_type(query)
            
            # Get appropriate prompt template
            prompt = self._get_prompt_template(query_type)
            
            # Update chain with new prompt
            self.qa_chain.combine_documents_chain.llm_chain.prompt = prompt
            
            # Process the query
            result = self.qa_chain({"query": query})
            
            # Extract sources
            sources = []
            for doc in result.get("source_documents", []):
                source = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source)
            
            # Prepare response
            response = {
                "answer": result.get("result", "No answer found."),
                "sources": sources,
                "metadata": {
                    "query_type": query_type,
                    "timestamp": datetime.now().isoformat(),
                    "model": os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
                }
            }
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the RAG model
        """
        return {
            "query_count": self.query_count,
            "last_query_time": self.last_query_time,
            "vector_store_documents": self.vector_store._index.ntotal if hasattr(self.vector_store, "_index") else 0,
            "embeddings_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
            "llm_model": os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        }