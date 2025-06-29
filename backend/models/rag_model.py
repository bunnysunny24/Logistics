import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import pandas as pd
from loguru import logger

class LogisticsPulseRAG:
    def __init__(self):
        # Load environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.index_dir = os.getenv("INDEX_DIR", "./data/index")
        self.data_dir = os.getenv("DATA_DIR", "./data")
        
        # Initialize components with better models
        if self.api_key:
            self.llm = ChatOpenAI(
                model=self.llm_model,
                temperature=0.1,  # Lower temperature for more consistent responses
                max_tokens=2000,
                api_key=self.api_key
            )
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key
            )
        else:
            # Fallback to local models
            logger.warning("OpenAI API key not found, using local models")
            self.llm = None  # Will be handled in process_query
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Load and refresh data
        self.load_prompts()
        self.load_domain_knowledge()
        
        # Initialize conversation memory with window
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="query",
            output_key="result",
            return_messages=True,
            k=5  # Keep last 5 exchanges
        )
        
        # Initialize enhanced vector stores
        self.vector_stores = {}
        self.last_checked = {}
        self.refresh_interval = 60  # seconds
        self.initialize_vector_stores()
        
    def load_prompts(self):
        """Load enhanced prompt templates from files"""
        try:
            # Determine the base path for prompts
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_dir = os.path.join(current_dir, '..', 'prompts')
            
            # Invoice compliance prompts
            invoice_prompt_path = os.path.join(prompts_dir, 'invoice_qna_prompt.txt')
            if os.path.exists(invoice_prompt_path):
                with open(invoice_prompt_path, 'r') as f:
                    invoice_prompt = f.read()
            else:
                invoice_prompt = self._get_default_invoice_prompt()
            
            # Shipment anomaly prompts
            shipment_prompt_path = os.path.join(prompts_dir, 'shipment_qna_prompt.txt')
            if os.path.exists(shipment_prompt_path):
                with open(shipment_prompt_path, 'r') as f:
                    shipment_prompt = f.read()
            else:
                shipment_prompt = self._get_default_shipment_prompt()
                
        except Exception as e:
            logger.warning(f"Error loading prompt files: {e}")
            invoice_prompt = self._get_default_invoice_prompt()
            shipment_prompt = self._get_default_shipment_prompt()
        
        self.prompt_templates = {
            "invoice": {
                "system": invoice_prompt,
                "query": invoice_prompt,
            },
            "shipment": {
                "system": shipment_prompt,
                "query": shipment_prompt,
            },
            "general": {
                "system": """You are LogisticsPulse Copilot, an AI assistant specialized in logistics operations, 
                supply chain management, invoice processing, and fraud detection. You provide accurate, 
                actionable insights based on the provided data and context.""",
                "query": """Based on the retrieved context, provide a comprehensive answer to the following query.
                
Context: {context}

Query: {query}

Provide specific details, risk assessments, and actionable recommendations where applicable."""
            }
        }
    
    def _get_default_invoice_prompt(self):
        """Default invoice compliance prompt"""
        return """You are LogisticsPulse Copilot, an AI assistant specialized in invoice processing and payment compliance.

CONTEXT INFORMATION:
-----------------
{context}
-----------------

Given the context information, answer the question: {query}

Focus on invoice-related details such as:
1. Invoice numbers, amounts, and payment terms
2. Due dates and early payment discounts  
3. Late payment penalties and compliance issues
4. Supplier-specific payment terms
5. Risk assessments and anomaly detection
6. Financial compliance and regulatory requirements

If answering about compliance issues, clearly explain:
- Which specific clause or policy is relevant
- Why the invoice is compliant or non-compliant
- Risk score and severity assessment
- Recommended actions for handling the situation

Provide specific data points, calculations, and actionable insights.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

ANSWER:"""

    def _get_default_shipment_prompt(self):
        """Default shipment anomaly prompt"""
        return """You are LogisticsPulse Copilot, an AI assistant specialized in shipment tracking and anomaly detection.

CONTEXT INFORMATION:
-----------------
{context}
-----------------

Given the context information, answer the question: {query}

Focus on shipment-related details such as:
1. Shipment routes, carriers, and schedules
2. Origin and destination information
3. Anomaly detection and risk factors
4. Historical patterns and deviations
5. Fraud indicators and security concerns
6. Performance metrics and KPIs

If explaining an anomaly or flag, clearly state:
- What triggered the flag
- The risk score and its meaning
- Possible reasons for the deviation
- Impact assessment
- Recommended actions and next steps

Provide specific data analysis, trend identification, and actionable recommendations.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

ANSWER:"""
    
    def load_domain_knowledge(self):
        """Load domain-specific knowledge and rules"""
        self.domain_knowledge = {
            "invoice_compliance_rules": {
                "standard_payment_terms": 30,
                "early_payment_discount_threshold": 10,
                "late_payment_penalty_rate": 0.05,
                "high_risk_amount_threshold": 10000,
                "duplicate_invoice_window_days": 30
            },
            "shipment_anomaly_thresholds": {
                "route_deviation_km": 200,
                "delivery_delay_days": 2,
                "value_variance_percentage": 0.25,
                "carrier_change_risk_score": 0.7,
                "customs_delay_threshold_days": 5
            },
            "fraud_indicators": {
                "invoice": [
                    "duplicate_invoice_number",
                    "round_amount_pattern",
                    "weekend_processing",
                    "unusual_supplier_location",
                    "missing_required_fields"
                ],
                "shipment": [
                    "route_inconsistency",
                    "carrier_mismatch",
                    "value_discrepancy",
                    "document_tampering",
                    "timing_anomaly"
                ]
            }
        }
    
    def initialize_vector_stores(self):
        """Initialize vector stores from CSV data and existing documents"""
        try:
            # Load and process invoice data
            self._load_invoice_data()
            
            # Load and process shipment data  
            self._load_shipment_data()
            
            # Load policy documents
            self._load_policy_data()
            
            logger.info("Vector stores initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector stores: {e}")
    
    def _load_invoice_data(self):
        """Load and vectorize invoice data"""
        invoice_docs = []
        
        # Load CSV files
        invoice_files = [
            "comprehensive_invoices.csv",
            "invoice_001.csv", 
            "invoice_002.csv",
            "invoice_003.csv",
            "invoice_004_abnormal.csv"
        ]
        
        for file in invoice_files:
            file_path = f"{self.data_dir}/invoices/{file}"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    for _, row in df.iterrows():
                        # Create comprehensive document text
                        doc_text = self._format_invoice_document(row.to_dict())
                        
                        invoice_docs.append(Document(
                            page_content=doc_text,
                            metadata={
                                "source": file,
                                "doc_type": "invoice",
                                "invoice_id": row.get("invoice_id", "unknown"),
                                "supplier": row.get("supplier", "unknown"),
                                "amount": row.get("amount", 0.0),
                                "status": row.get("status", "unknown")
                            }
                        ))
                except Exception as e:
                    logger.error(f"Error loading invoice file {file}: {e}")
        
        # Create vector store for invoices
        if invoice_docs:
            self.vector_stores["invoice"] = FAISS.from_documents(invoice_docs, self.embeddings)
            logger.info(f"Created invoice vector store with {len(invoice_docs)} documents")
    
    def _load_shipment_data(self):
        """Load and vectorize shipment data"""
        shipment_docs = []
        
        # Load CSV files
        shipment_files = [
            "comprehensive_shipments.csv",
            "shipment_001.csv",
            "shipment_002.csv", 
            "shipment_003_abnormal.csv",
            "shipment_004_abnormal.csv"
        ]
        
        for file in shipment_files:
            file_path = f"{self.data_dir}/shipments/{file}"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    for _, row in df.iterrows():
                        # Create comprehensive document text
                        doc_text = self._format_shipment_document(row.to_dict())
                        
                        shipment_docs.append(Document(
                            page_content=doc_text,
                            metadata={
                                "source": file,
                                "doc_type": "shipment", 
                                "shipment_id": row.get("shipment_id", "unknown"),
                                "origin": row.get("origin", "unknown"),
                                "destination": row.get("destination", "unknown"),
                                "carrier": row.get("carrier", "unknown"),
                                "status": row.get("status", "unknown"),
                                "risk_score": row.get("risk_score", 0.0)
                            }
                        ))
                except Exception as e:
                    logger.error(f"Error loading shipment file {file}: {e}")
        
        # Create vector store for shipments
        if shipment_docs:
            self.vector_stores["shipment"] = FAISS.from_documents(shipment_docs, self.embeddings)
            logger.info(f"Created shipment vector store with {len(shipment_docs)} documents")
    
    def _load_policy_data(self):
        """Load and vectorize policy documents"""
        policy_docs = []
        
        # Load policy files
        policy_files = ["payout-rules-v3.md", "shipment-guidelines-v2.md"]
        
        for file in policy_files:
            file_path = f"{self.data_dir}/policies/{file}"
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split content into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", ". ", " "]
                    )
                    chunks = text_splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        policy_docs.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": file,
                                "doc_type": "policy",
                                "chunk_id": i,
                                "policy_type": "payout" if "payout" in file else "shipment"
                            }
                        ))
                except Exception as e:
                    logger.error(f"Error loading policy file {file}: {e}")
        
        # Create vector store for policies
        if policy_docs:
            self.vector_stores["policy"] = FAISS.from_documents(policy_docs, self.embeddings)
            logger.info(f"Created policy vector store with {len(policy_docs)} documents")
    
    def _format_invoice_document(self, invoice_data):
        """Format invoice data into searchable text"""
        # Safe number formatting
        try:
            amount = float(invoice_data.get('amount', 0.0))
            amount_str = f"${amount:,.2f}"
        except (ValueError, TypeError):
            amount_str = f"${invoice_data.get('amount', 'Unknown')}"
        
        try:
            discount = float(invoice_data.get('early_discount', 0.0))
            discount_str = f"{discount*100:.1f}%"
            discount_desc = f"{discount*100:.1f}%"
        except (ValueError, TypeError):
            discount_str = f"{invoice_data.get('early_discount', 'Unknown')}"
            discount_desc = f"{invoice_data.get('early_discount', 'Unknown')}"
        
        return f"""
Invoice ID: {invoice_data.get('invoice_id', 'Unknown')}
Supplier: {invoice_data.get('supplier', 'Unknown')}
Amount: {amount_str} {invoice_data.get('currency', 'USD')}
Issue Date: {invoice_data.get('issue_date', 'Unknown')}
Due Date: {invoice_data.get('due_date', 'Unknown')}
Payment Terms: {invoice_data.get('payment_terms', 'Unknown')}
Early Discount: {discount_str}
Status: {invoice_data.get('status', 'Unknown')}
Approver: {invoice_data.get('approver', 'Unknown')}

This invoice shows payment terms of {invoice_data.get('payment_terms', 'NET30')} with an early payment discount of {discount_desc}.
The invoice amount is {amount_str} and is currently {invoice_data.get('status', 'pending')}.
        """.strip()
    
    def _format_shipment_document(self, shipment_data):
        """Format shipment data into searchable text"""
        # Safe number formatting
        try:
            risk_score = float(shipment_data.get('risk_score', 0.0))
            risk_str = f"{risk_score:.2f}"
            risk_desc = f"{risk_score:.2f}"
        except (ValueError, TypeError):
            risk_str = f"{shipment_data.get('risk_score', 'Unknown')}"
            risk_desc = f"{shipment_data.get('risk_score', 'Unknown')}"
        
        return f"""
Shipment ID: {shipment_data.get('shipment_id', 'Unknown')}
Route: {shipment_data.get('origin', 'Unknown')} → {shipment_data.get('destination', 'Unknown')}
Carrier: {shipment_data.get('carrier', 'Unknown')}
Departure Date: {shipment_data.get('departure_date', 'Unknown')}
Estimated Arrival: {shipment_data.get('estimated_arrival', 'Unknown')}
Actual Arrival: {shipment_data.get('actual_arrival', 'TBD')}
Status: {shipment_data.get('status', 'Unknown')}
Risk Score: {risk_str}
Anomaly Type: {shipment_data.get('anomaly_type', 'none')}

This shipment is traveling from {shipment_data.get('origin', 'Unknown')} to {shipment_data.get('destination', 'Unknown')} 
via {shipment_data.get('carrier', 'Unknown')}. Current status is {shipment_data.get('status', 'Unknown')} 
with a risk score of {risk_desc}.
{f"Detected anomaly: {shipment_data.get('anomaly_type')}" if shipment_data.get('anomaly_type') != 'none' else "No anomalies detected."}
        """.strip()
    
    def get_vector_store(self, doc_type):
        """Get vector store for a document type, refreshing if needed"""
        current_time = time.time()
        
        # Check if we need to refresh the vector store
        if (doc_type not in self.vector_stores or 
            doc_type not in self.last_checked or
            current_time - self.last_checked[doc_type] > self.refresh_interval):
            
            # Refresh from data files
            if doc_type == "invoice":
                self._load_invoice_data()
            elif doc_type == "shipment":
                self._load_shipment_data()
            elif doc_type == "policy":
                self._load_policy_data()
            
            # Update the last checked time
            self.last_checked[doc_type] = current_time
        
        return self.vector_stores.get(doc_type)

    def detect_doc_types_from_query(self, query):
        """Enhanced document type detection with better keywords"""
        query_lower = query.lower()
        
        doc_types = []
        
        # Enhanced invoice-related terms
        invoice_terms = [
            "invoice", "payment", "discount", "late fee", "compliance", "due date",
            "supplier", "billing", "amount", "currency", "net30", "net45", 
            "early payment", "penalty", "payout", "financial", "accounting"
        ]
        
        # Enhanced shipment-related terms  
        shipment_terms = [
            "shipment", "route", "delivery", "tracking", "anomaly", "freight",
            "carrier", "origin", "destination", "transit", "logistics", 
            "cargo", "customs", "delay", "deviation", "fraud", "risk"
        ]
        
        # Policy-related terms
        policy_terms = [
            "policy", "rule", "guideline", "procedure", "protocol", 
            "compliance", "regulation", "standard", "requirement"
        ]
        
        # Check for matches
        if any(term in query_lower for term in invoice_terms):
            doc_types.append("invoice")
        
        if any(term in query_lower for term in shipment_terms):
            doc_types.append("shipment")
            
        if any(term in query_lower for term in policy_terms):
            doc_types.append("policy")
        
        # If no specific types detected, prioritize based on query intent
        if not doc_types:
            # Default to both main document types
            doc_types = ["invoice", "shipment"]
        
        return doc_types

    def process_query(self, query, context=None):
        """Enhanced query processing with better retrieval and response generation"""
        start_time = time.time()
        
        # Initialize context if None
        if context is None:
            context = {}
        
        # Handle case where LLM is not available
        if not self.llm:
            return self._generate_fallback_response(query)
        
        # Detect relevant document types
        doc_types = self.detect_doc_types_from_query(query)
        logger.info(f"Detected document types: {doc_types}")
        
        # Get combined results from all relevant document types
        combined_docs = []
        sources_info = []
        
        for doc_type in doc_types:
            vector_store = self.get_vector_store(doc_type)
            if vector_store:
                try:
                    # Search for relevant documents with similarity scores
                    docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
                    
                    # Filter by relevance score (lower is better for similarity)
                    relevant_docs = [doc for doc, score in docs_with_scores if score < 0.8]
                    
                    combined_docs.extend(relevant_docs)
                    
                    # Track sources
                    for doc in relevant_docs:
                        sources_info.append({
                            "type": doc_type,
                            "source": doc.metadata.get("source", "unknown"),
                            "content_preview": doc.page_content[:200] + "..."
                        })
                        
                except Exception as e:
                    logger.error(f"Error searching {doc_type} vector store: {e}")
        
        # If no relevant documents found
        if not combined_docs:
            return {
                "answer": "I couldn't find relevant information in the current dataset to answer your question. Please try rephrasing your query or provide more specific details about invoices, shipments, or policies.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_time_ms": (time.time() - start_time) * 1000,
                    "doc_types_searched": doc_types
                }
            }
        
        # Determine the primary document type for prompt selection
        primary_type = doc_types[0] if doc_types else "general"
        
        # Create enhanced context
        context_text = "\n\n".join([doc.page_content for doc in combined_docs[:10]])
        
        # Add domain knowledge context
        domain_context = self._get_domain_context(query, primary_type)
        if domain_context:
            context_text = f"{domain_context}\n\n{context_text}"
        
        # Create prompt with better formatting
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template=self.prompt_templates[primary_type]["query"]
        )
        
        # Create retrieval QA chain with compression
        try:
            # Create retriever with the best vector store
            best_vector_store = self.get_vector_store(primary_type) or list(self.vector_stores.values())[0]
            retriever = best_vector_store.as_retriever(search_kwargs={"k": 8})
            
            # Add compression to improve relevance
            if self.llm:
                compressor = LLMChainExtractor.from_llm(self.llm)
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retriever
                )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": prompt_template,
                    "verbose": False
                },
                return_source_documents=True
            )
            
            # Execute the chain
            result = qa_chain({"query": query})
            
            # Calculate confidence based on source relevance and completeness
            confidence = self._calculate_confidence(result, combined_docs, query)
            
            # Format sources with better information
            formatted_sources = []
            for i, source_info in enumerate(sources_info[:5]):  # Limit to top 5 sources
                formatted_sources.append(f"{source_info['type']}: {source_info['source']}")
            
            return {
                "answer": result["result"],
                "sources": formatted_sources,
                "confidence": confidence,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "doc_types": doc_types,
                    "query_time_ms": (time.time() - start_time) * 1000,
                    "documents_retrieved": len(combined_docs),
                    "primary_type": primary_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error in QA chain execution: {e}")
            return self._generate_error_response(query, str(e))
    
    def _get_domain_context(self, query, doc_type):
        """Get relevant domain knowledge for the query"""
        context_parts = []
        
        if doc_type == "invoice":
            rules = self.domain_knowledge["invoice_compliance_rules"]
            context_parts.append(f"Standard payment terms: {rules['standard_payment_terms']} days")
            context_parts.append(f"High-risk amount threshold: ${rules['high_risk_amount_threshold']:,}")
            
        elif doc_type == "shipment":
            thresholds = self.domain_knowledge["shipment_anomaly_thresholds"]
            context_parts.append(f"Route deviation threshold: {thresholds['route_deviation_km']} km")
            context_parts.append(f"Delivery delay threshold: {thresholds['delivery_delay_days']} days")
            
        # Add fraud indicators if query mentions fraud/risk
        if any(term in query.lower() for term in ["fraud", "risk", "anomaly", "suspicious"]):
            indicators = self.domain_knowledge["fraud_indicators"].get(doc_type, [])
            if indicators:
                context_parts.append(f"Fraud indicators to watch: {', '.join(indicators[:3])}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _calculate_confidence(self, result, docs, query):
        """Calculate confidence score based on result quality"""
        base_confidence = 0.7
        
        # Adjust based on number of source documents
        if len(docs) >= 3:
            base_confidence += 0.1
        elif len(docs) == 0:
            base_confidence = 0.2
            
        # Adjust based on answer length and completeness
        answer = result.get("result", "")
        if len(answer) > 100:
            base_confidence += 0.1
        if "I don't have enough information" in answer:
            base_confidence = 0.3
            
        # Cap at 0.95 to show some uncertainty
        return min(0.95, base_confidence)
    
    def _generate_fallback_response(self, query):
        """Generate a response when LLM is not available"""
        return {
            "answer": "I'm currently running in demo mode without access to the full AI model. Please configure your OpenAI API key to get comprehensive responses about your logistics data.",
            "sources": [],
            "confidence": 0.1,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "mode": "fallback"
            }
        }
    
    def _generate_error_response(self, query, error_msg):
        """Generate an error response"""
        return {
            "answer": f"I encountered an error while processing your query: {error_msg}. Please try rephrasing your question or contact support if the issue persists.",
            "sources": [],
            "confidence": 0.0,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }
        }
    
    def get_status(self):
        """Get enhanced status of the RAG model"""
        vector_store_stats = {}
        for doc_type, store in self.vector_stores.items():
            try:
                # Try to get document count
                doc_count = len(store.docstore._dict) if hasattr(store, 'docstore') else 0
                vector_store_stats[doc_type] = {
                    "documents": doc_count,
                    "last_refreshed": self.last_checked.get(doc_type, 0)
                }
            except:
                vector_store_stats[doc_type] = {"documents": "unknown", "last_refreshed": 0}
        
        return {
            "model": self.llm_model,
            "embedding_model": self.embedding_model,
            "api_key_configured": bool(self.api_key),
            "vector_stores": vector_store_stats,
            "conversation_turns": len(self.memory.chat_memory.messages) // 2,
            "domain_knowledge_loaded": bool(self.domain_knowledge),
            "refresh_interval_seconds": self.refresh_interval
        }
    
    def add_feedback(self, query, answer, rating, feedback_text=None):
        """Add user feedback to improve responses"""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "rating": rating,
            "feedback": feedback_text
        }
        
        # Save feedback for future model improvements
        feedback_file = f"{self.data_dir}/feedback.jsonl"
        try:
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(feedback_entry) + '\n')
            logger.info("User feedback saved successfully")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")