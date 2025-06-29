import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
import pandas as pd
from loguru import logger
from models.local_llm import LocalHuggingFaceLLM

class LogisticsPulseRAG:
    # In backend/models/rag_model.py - update the __init__ method

    def __init__(self):
        # Load environment variables - fully local setup
        self.llm_model = os.getenv("LLM_MODEL", "local")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.index_dir = os.getenv("INDEX_DIR", "./data/index")
        self.data_dir = os.getenv("DATA_DIR", "./data")
        
        # Real-time monitoring attributes
        self.last_policy_update = datetime.now()
        self.policy_cache = {}
        self.compliance_rules_cache = {}
        
        # Initialize with local models only
        logger.info("Initializing fully local RAG system - no external API dependencies")
        
        # Use local models exclusively
        try:
            # Try to load a smaller model by default
            self.llm = LocalHuggingFaceLLM(model_name="facebook/opt-1.3b")
            logger.info("Successfully initialized local LLM")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            self.llm = None  # Fall back to template responses
            
        # Local embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model
            )
            logger.info(f"Successfully initialized local embeddings: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embeddings model {self.embedding_model}: {e}")
            # Try a more basic model as fallback
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Successfully initialized fallback embeddings model")
            except Exception as e2:
                logger.error(f"Failed to load fallback embeddings: {e2}")
                self.embeddings = None
        
        # Load and refresh data
        self.load_prompts()
        self.load_domain_knowledge()
        self.load_compliance_rules()
        
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
        """Enhanced query processing with hybrid retrieval and response generation"""
        start_time = time.time()
        
        # Initialize context if None
        if context is None:
            context = {}
        
        # Handle case where LLM is not available
        if not self.llm:
            return self._generate_fallback_response(query)
        
        # Detect relevant document types with enhanced logic
        doc_types = self.detect_doc_types_from_query(query)
        logger.info(f"Detected document types: {doc_types}")
        
        # Perform hybrid retrieval: vector similarity + keyword matching
        combined_docs = []
        sources_info = []
        semantic_docs = []
        keyword_docs = []
        
        for doc_type in doc_types:
            vector_store = self.get_vector_store(doc_type)
            if vector_store:
                try:
                    # 1. Semantic search with improved scoring
                    docs_with_scores = vector_store.similarity_search_with_score(query, k=8)
                    
                    # Adaptive threshold based on query complexity
                    threshold = 0.7 if len(query.split()) > 5 else 0.8
                    semantic_relevant = [doc for doc, score in docs_with_scores if score < threshold]
                    
                    # 2. Keyword-based retrieval for specific terms
                    keyword_relevant = self._keyword_search(vector_store, query, doc_type)
                    
                    # 3. Combine and deduplicate results
                    all_relevant = self._combine_retrieval_results(semantic_relevant, keyword_relevant)
                    
                    combined_docs.extend(all_relevant)
                    
                    # Track sources with enhanced metadata
                    for doc in all_relevant:
                        sources_info.append({
                            "type": doc_type,
                            "source": doc.metadata.get("source", "unknown"),
                            "doc_id": doc.metadata.get("invoice_id", doc.metadata.get("shipment_id", "unknown")),
                            "relevance_score": self._calculate_doc_relevance(doc, query),
                            "content_preview": doc.page_content[:200] + "..."
                        })
                        
                except Exception as e:
                    logger.error(f"Error searching {doc_type} vector store: {e}")
        
        # If no relevant documents found, try relaxed search
        if not combined_docs:
            combined_docs = self._fallback_search(query, doc_types)
        
        # Still no results - return informative response
        if not combined_docs:
            return self._generate_no_results_response(query, doc_types, start_time)
        
        # Rank documents by relevance and recency
        ranked_docs = self._rank_documents(combined_docs, query)
        
        # Determine the primary document type for prompt selection
        primary_type = doc_types[0] if doc_types else "general"
        
        # Create enhanced context with better structure
        context_text = self._create_enhanced_context(ranked_docs[:10], query, primary_type)
        
        # Add domain knowledge and compliance rules
        domain_context = self._get_domain_context(query, primary_type)
        if domain_context:
            context_text = f"{domain_context}\n\n{context_text}"
        
        # Create dynamic prompt based on query type
        prompt_template = self._create_dynamic_prompt(query, primary_type)
        
        # Generate response with improved chain
        try:
    # Use direct LLM call for better control
            formatted_prompt = prompt_template.format(query=query, context=context_text)
    
    # Get response from LLM - with adjusted parameters for local models
            if is_local_llm:
        # Local models may need different parameters
                response = self.llm.invoke(formatted_prompt, max_new_tokens=256, temperature=0.2)
            else:
        # OpenAI API call
                response = self.llm.invoke(formatted_prompt)
    
    # Extract answer content
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
    
    # Post-process answer for consistency
            answer = self._post_process_answer(answer, query, primary_type)
    
    # Calculate enhanced confidence score
            confidence = self._calculate_enhanced_confidence(answer, ranked_docs, query, sources_info)
    
    # Format sources with better information
            formatted_sources = self._format_enhanced_sources(sources_info[:5])
    
            return {
                "answer": answer,
                "sources": formatted_sources,
                "confidence": confidence,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "doc_types": doc_types,
                    "query_time_ms": (time.time() - start_time) * 1000,
                    "documents_retrieved": len(combined_docs),
                    "documents_ranked": len(ranked_docs),
                    "primary_type": primary_type,
                    "retrieval_method": "hybrid",
                    "query_complexity": len(query.split()),
                    "model_type": "local" if is_local_llm else "openai"
                }
            }
    
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
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
    # Detect document types to determine appropriate response
        doc_types = self.detect_doc_types_from_query(query)
        primary_type = doc_types[0] if doc_types else "general"
    
    # Generate a more helpful response based on query type
        if any(word in query.lower() for word in ["list", "show", "find"]):
            return {
                "answer": f"I understand you want information about {primary_type}. Using local models, I can index your documents but provide limited natural language responses. You can use the search functionality in the application to find specific {primary_type} data.",
                "sources": [],
                "confidence": 0.5,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "mode": "local_only",
                    "query_type": "listing"
                }
            }
        elif any(word in query.lower() for word in ["why", "how", "explain"]):
            return {
                "answer": f"You're asking for an explanation related to {primary_type}. While using local models, detailed explanations are limited. Please check the documentation or data sources directly for this information.",
                "sources": [],
                "confidence": 0.5,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "mode": "local_only",
                    "query_type": "explanation"
                }
            }
        else:
            return {
                "answer": f"I've processed your query about {primary_type} using local models. Your documents have been indexed locally and the system can identify relevant content. For detailed analysis, consider using specific search terms or filters in the application interface.",
                "sources": [],
                "confidence": 0.5,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "mode": "local_only"
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
    
    def load_compliance_rules(self):
        """Load real-time compliance rules and policy information"""
        try:
            # Load current policy versions and rules
            policies_dir = os.path.join(self.data_dir, "policies")
            
            self.compliance_rules_cache = {
                "payout_rules_v3": {
                    "version": "3.0",
                    "effective_date": "2025-06-15",
                    "late_fee_rate": 0.015,  # 1.5% per month
                    "late_fee_threshold_days": 30,
                    "early_discount_rate": 0.02,  # 2% discount
                    "early_discount_window": 10,  # days
                    "approval_thresholds": {
                        "auto": 5000,
                        "manager": 15000,
                        "director": 50000,
                        "cfo": 100000
                    }
                },
                "shipment_guidelines_v2": {
                    "version": "2.0", 
                    "effective_date": "2025-06-01",
                    "route_deviation_threshold": 200,  # km
                    "delivery_delay_threshold": 2,  # days
                    "value_variance_threshold": 0.25,  # 25%
                    "carrier_risk_threshold": 0.7,
                    "approved_carriers": [
                        "Global Shipping Inc",
                        "Express Worldwide",
                        "Reliable Freight",
                        "International Logistics Co"
                    ]
                },
                "emergency_protocols": {
                    "weekend_processing_flag": True,
                    "holiday_restrictions": True,
                    "expedited_approval_limit": 25000,
                    "critical_route_monitoring": True
                }
            }
            
            # Load any updated rules from policy files
            if os.path.exists(policies_dir):
                for filename in os.listdir(policies_dir):
                    if filename.endswith('.md'):
                        policy_path = os.path.join(policies_dir, filename)
                        try:
                            with open(policy_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                self._parse_policy_updates(filename, content)
                        except Exception as e:
                            logger.error(f"Error reading policy file {filename}: {e}")
                            
            logger.info("Compliance rules loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading compliance rules: {e}")
    
    def _parse_policy_updates(self, filename, content):
        """Parse policy files for rule updates"""
        content_lower = content.lower()
        
        # Parse late fee updates
        if 'late fee' in content_lower:
            # Look for percentage patterns like "1.5%" or "2.5%"
            import re
            fee_match = re.search(r'(\d+\.?\d*)%.*per month', content_lower)
            if fee_match:
                new_rate = float(fee_match.group(1)) / 100
                self.compliance_rules_cache["payout_rules_v3"]["late_fee_rate"] = new_rate
                logger.info(f"Updated late fee rate to {new_rate*100}% from {filename}")
        
        # Parse approval threshold updates
        if 'approval' in content_lower and '$' in content:
            # Look for dollar amounts like "$20,000" or "$50,000"
            import re
            amounts = re.findall(r'\$(\d{1,3}(?:,\d{3})*)', content)
            if amounts:
                # Update thresholds based on context
                for amount_str in amounts:
                    amount = int(amount_str.replace(',', ''))
                    if 'cfo' in content_lower and amount >= 50000:
                        self.compliance_rules_cache["payout_rules_v3"]["approval_thresholds"]["cfo"] = amount
                    elif 'director' in content_lower and amount >= 20000:
                        self.compliance_rules_cache["payout_rules_v3"]["approval_thresholds"]["director"] = amount
                    elif 'manager' in content_lower and amount >= 10000:
                        self.compliance_rules_cache["payout_rules_v3"]["approval_thresholds"]["manager"] = amount
        
        # Update last policy change timestamp
        self.last_policy_update = datetime.now()

    def get_current_compliance_rules(self):
        """Get current compliance rules with real-time updates"""
        # Check for policy file updates
        try:
            policies_dir = os.path.join(self.data_dir, "policies")
            if os.path.exists(policies_dir):
                latest_mtime = 0
                for filename in os.listdir(policies_dir):
                    if filename.endswith('.md'):
                        policy_path = os.path.join(policies_dir, filename)
                        mtime = os.path.getmtime(policy_path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                
                # If policies have been updated, reload them
                if latest_mtime > self.last_policy_update.timestamp():
                    logger.info("Policy files updated, reloading compliance rules")
                    self.load_compliance_rules()
        except Exception as e:
            logger.error(f"Error checking for policy updates: {e}")
        
        return self.compliance_rules_cache

    def add_document_to_index(self, content: str, doc_type: str, metadata: dict = None):
        """Add a new document to the vector store index"""
        try:
            if not content or not doc_type:
                logger.warning("Cannot add empty content or missing document type to index")
                return False
        
            if not metadata:
                metadata = {}
        
        # Create text chunks for better indexing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        
        # Split text into chunks
            chunks = text_splitter.split_text(content)
            logger.info(f"Split document into {len(chunks)} chunks")
        
        # Create document objects
            docs = []
            for i, chunk in enumerate(chunks):
            # Skip empty chunks
                if not chunk.strip():
                    continue
                
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": metadata.get("source", "unknown"),
                        "doc_type": doc_type,
                        "chunk_id": i,
                        "timestamp": datetime.now().isoformat(),
                        **metadata
                    }
                ))
        
            if not docs:
                logger.warning("No valid content chunks to index")
                return False
        
            # Add to appropriate vector store
            if doc_type not in self.vector_stores:
                # Initialize new vector store if needed
                self.vector_stores[doc_type] = FAISS.from_documents(docs, self.embeddings)
                logger.info(f"Created new vector store for {doc_type} with {len(docs)} chunks")
            else:
                # Add to existing vector store
                self.vector_stores[doc_type].add_documents(docs)
                logger.info(f"Added {len(docs)} chunks to existing {doc_type} vector store")

            # Update last checked time
            self.last_checked[doc_type] = time.time()
        
            return True
    
        except Exception as e:
            logger.error(f"Error adding document to index: {e}")
            return False

    def calculate_late_fees(self, invoice_amount, days_overdue):
        """Calculate late fees based on current policy"""
        rules = self.get_current_compliance_rules()
        payout_rules = rules.get("payout_rules_v3", {})
        
        late_fee_rate = payout_rules.get("late_fee_rate", 0.015)
        threshold_days = payout_rules.get("late_fee_threshold_days", 30)
        
        if days_overdue <= threshold_days:
            return 0.0
        
        # Calculate compound monthly interest
        months_overdue = (days_overdue - threshold_days) / 30.0
        late_fee = invoice_amount * late_fee_rate * months_overdue
        
        return round(late_fee, 2)

    def get_approval_requirement(self, invoice_amount):
        """Determine approval requirement based on current policy"""
        rules = self.get_current_compliance_rules()
        thresholds = rules.get("payout_rules_v3", {}).get("approval_thresholds", {})
        
        if invoice_amount >= thresholds.get("cfo", 100000):
            return "CFO approval required"
        elif invoice_amount >= thresholds.get("director", 50000):
            return "Director approval required"
        elif invoice_amount >= thresholds.get("manager", 15000):
            return "Manager approval required"
        elif invoice_amount >= thresholds.get("auto", 5000):
            return "Automatic approval"
        else:
            return "Standard processing"

    def assess_shipment_risk(self, shipment_data):
        """Assess shipment risk based on current guidelines"""
        rules = self.get_current_compliance_rules()
        guidelines = rules.get("shipment_guidelines_v2", {})
        
        risk_factors = []
        risk_score = 0.0
        
        # Check carrier approval
        carrier = shipment_data.get('carrier', '')
        approved_carriers = guidelines.get('approved_carriers', [])
        if carrier not in approved_carriers:
            risk_factors.append(f"Non-approved carrier: {carrier}")
            risk_score += 0.3
        
        # Check value variance
        current_value = shipment_data.get('value', 0)
        historical_avg = shipment_data.get('historical_average', current_value)
        if historical_avg > 0:
            variance = abs(current_value - historical_avg) / historical_avg
            threshold = guidelines.get('value_variance_threshold', 0.25)
            if variance > threshold:
                risk_factors.append(f"Value variance: {variance:.1%} (threshold: {threshold:.1%})")
                risk_score += min(0.4, variance)
        
        # Determine severity
        if risk_score >= 0.9:
            severity = "Critical"
        elif risk_score >= 0.7:
            severity = "High"
        elif risk_score >= 0.4:
            severity = "Medium"
        else:
            severity = "Low"
        
        return {
            "risk_score": round(risk_score, 2),
            "severity": severity,
            "risk_factors": risk_factors,
            "guidelines_version": guidelines.get("version", "2.0")
        }
    
    def _keyword_search(self, vector_store, query, doc_type):
        """Perform keyword-based search to complement semantic search"""
        try:
            # Extract key terms from query
            key_terms = self._extract_key_terms(query, doc_type)
            
            # Get all documents from vector store
            all_docs = []
            try:
                # Similarity search with high k to get more docs for filtering
                docs_with_scores = vector_store.similarity_search_with_score(query, k=50)
                all_docs = [doc for doc, score in docs_with_scores]
            except:
                # Fallback to basic search
                all_docs = vector_store.similarity_search(query, k=20)
            
            # Filter by keyword matches
            keyword_matches = []
            for doc in all_docs:
                content_lower = doc.page_content.lower()
                match_score = 0
                
                for term in key_terms:
                    if term.lower() in content_lower:
                        match_score += 1
                
                if match_score > 0:
                    # Add match score to metadata
                    doc.metadata['keyword_score'] = match_score / len(key_terms)
                    keyword_matches.append(doc)
            
            # Sort by keyword match score
            keyword_matches.sort(key=lambda x: x.metadata.get('keyword_score', 0), reverse=True)
            
            return keyword_matches[:10]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _extract_key_terms(self, query, doc_type):
        """Extract key terms based on document type and query"""
        query_words = query.lower().split()
        
        # Document-specific key terms
        if doc_type == "invoice":
            key_terms = [word for word in query_words if word in [
                "invoice", "payment", "amount", "due", "discount", "supplier",
                "late", "fee", "penalty", "net30", "net45", "compliance"
            ]]
        elif doc_type == "shipment":
            key_terms = [word for word in query_words if word in [
                "shipment", "route", "carrier", "delivery", "origin", "destination",
                "anomaly", "delay", "risk", "fraud", "tracking"
            ]]
        else:
            key_terms = query_words
        
        # Add numbers and specific identifiers
        import re
        numbers = re.findall(r'\b\d+\b', query)
        key_terms.extend(numbers)
        
        # Add quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        key_terms.extend(quoted)
        
        return list(set(key_terms)) if key_terms else query_words[:5]
    
    def _combine_retrieval_results(self, semantic_docs, keyword_docs):
        """Combine semantic and keyword search results with deduplication"""
        combined = []
        seen_content = set()
        
        # Add semantic results first (usually higher quality)
        for doc in semantic_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                doc.metadata['retrieval_method'] = 'semantic'
                combined.append(doc)
                seen_content.add(content_hash)
        
        # Add keyword results that aren't duplicates
        for doc in keyword_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                doc.metadata['retrieval_method'] = 'keyword'
                combined.append(doc)
                seen_content.add(content_hash)
        
        return combined
    
    def _calculate_doc_relevance(self, doc, query):
        """Calculate document relevance score"""
        try:
            content = doc.page_content.lower()
            query_lower = query.lower()
            
            # Basic relevance factors
            relevance = 0.0
            
            # Exact phrase matches
            if query_lower in content:
                relevance += 0.4
            
            # Word overlap
            query_words = set(query_lower.split())
            content_words = set(content.split())
            overlap = len(query_words.intersection(content_words))
            relevance += (overlap / len(query_words)) * 0.3
            
            # Keyword score if available
            keyword_score = doc.metadata.get('keyword_score', 0)
            relevance += keyword_score * 0.3
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Error calculating document relevance: {e}")
            return 0.5
    
    def _fallback_search(self, query, doc_types):
        """Fallback search with relaxed criteria"""
        fallback_docs = []
        
        for doc_type in doc_types:
            vector_store = self.get_vector_store(doc_type)
            if vector_store:
                try:
                    # Use broader search with relaxed threshold
                    docs_with_scores = vector_store.similarity_search_with_score(query, k=10)
                    relaxed_docs = [doc for doc, score in docs_with_scores if score < 1.2]
                    fallback_docs.extend(relaxed_docs[:3])  # Limit fallback results
                except Exception as e:
                    logger.error(f"Error in fallback search for {doc_type}: {e}")
        
        return fallback_docs
    
    def _generate_no_results_response(self, query, doc_types, start_time):
        """Generate response when no relevant documents found"""
        doc_types_str = ", ".join(doc_types)
        
        return {
            "answer": f"I couldn't find specific information about '{query}' in the current {doc_types_str} data. This could mean:\n\n" +
                     "1. The information may not be in the loaded datasets\n" +
                     "2. Try rephrasing your question with different keywords\n" +
                     "3. Check if you're asking about the right document type (invoices, shipments, or policies)\n\n" +
                     "For example, try asking:\n" +
                     "- 'Show me invoices from [supplier name]'\n" +
                     "- 'What shipments have delays?'\n" +
                     "- 'List recent payment anomalies'",
            "sources": [],
            "confidence": 0.1,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query_time_ms": (time.time() - start_time) * 1000,
                "doc_types_searched": doc_types,
                "result_type": "no_results"
            }
        }
    
    def _rank_documents(self, docs, query):
        """Rank documents by relevance, recency, and document type priority"""
        try:
            # Calculate scores for each document
            scored_docs = []
            
            for doc in docs:
                score = 0.0
                
                # Relevance score (40% weight)
                relevance = self._calculate_doc_relevance(doc, query)
                score += relevance * 0.4
                
                # Document type priority (30% weight)
                doc_type = doc.metadata.get('doc_type', 'unknown')
                type_priority = {'invoice': 0.9, 'shipment': 0.8, 'policy': 0.7}.get(doc_type, 0.5)
                score += type_priority * 0.3
                
                # Recency score (20% weight) - newer is better
                try:
                    source = doc.metadata.get('source', '')
                    if 'comprehensive' in source.lower():
                        recency = 1.0  # Comprehensive files are most recent
                    elif any(x in source.lower() for x in ['001', '002', '003', '004']):
                        recency = 0.8  # Individual files are older
                    else:
                        recency = 0.6
                    score += recency * 0.2
                except:
                    score += 0.6 * 0.2
                
                # Keyword match bonus (10% weight)
                keyword_score = doc.metadata.get('keyword_score', 0)
                score += keyword_score * 0.1
                
                scored_docs.append((doc, score))
            
            # Sort by score (highest first)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs]
            
        except Exception as e:
            logger.error(f"Error ranking documents: {e}")
            return docs
    
    def _create_enhanced_context(self, docs, query, primary_type):
        """Create enhanced context with better structure and formatting"""
        if not docs:
            return ""
        
        context_parts = []
        
        # Add context header
        context_parts.append(f"=== RELEVANT {primary_type.upper()} INFORMATION ===\n")
        
        # Group documents by type for better organization
        doc_groups = {}
        for doc in docs:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            if doc_type not in doc_groups:
                doc_groups[doc_type] = []
            doc_groups[doc_type].append(doc)
        
        # Add each group with headers
        for doc_type, group_docs in doc_groups.items():
            if group_docs:
                context_parts.append(f"\n--- {doc_type.upper()} DATA ---")
                for i, doc in enumerate(group_docs[:5]):  # Limit per type
                    source = doc.metadata.get('source', 'unknown')
                    context_parts.append(f"\n[{doc_type.upper()} #{i+1} from {source}]")
                    context_parts.append(doc.page_content.strip())
                    context_parts.append("")  # Add spacing
        
        context_parts.append("\n=== END OF CONTEXT ===")
        
        return "\n".join(context_parts)
    
    def _create_dynamic_prompt(self, query, doc_type):
        """Create dynamic prompt based on query type and document type"""
        # Analyze query intent
        query_lower = query.lower()
        
        # Determine prompt type based on intent
        if any(word in query_lower for word in ['why', 'reason', 'cause', 'explain']):
            prompt_type = 'explanation'
        elif any(word in query_lower for word in ['how much', 'calculate', 'amount', 'cost']):
            prompt_type = 'calculation'
        elif any(word in query_lower for word in ['list', 'show', 'find', 'search']):
            prompt_type = 'search'
        elif any(word in query_lower for word in ['risk', 'anomaly', 'suspicious', 'fraud']):
            prompt_type = 'risk_assessment'
        else:
            prompt_type = 'general'
        
        # Select appropriate prompt template
        if doc_type == "invoice" and prompt_type == 'calculation':
            template = """You are LogisticsPulse Copilot, specialized in financial calculations and invoice analysis.

CONTEXT:
{context}

QUERY: {query}

Please provide:
1. Exact calculations with step-by-step breakdown
2. Relevant invoice amounts, dates, and terms
3. Compliance status and any policy violations
4. Risk assessment if applicable
5. Actionable recommendations

Show all mathematical work and cite specific invoices by ID.

ANSWER:"""
        
        elif doc_type == "shipment" and prompt_type == 'risk_assessment':
            template = """You are LogisticsPulse Copilot, specialized in logistics risk assessment and anomaly analysis.

CONTEXT:
{context}

QUERY: {query}

Please provide:
1. Risk score and severity level
2. Specific factors that contributed to the risk assessment
3. Comparison with historical patterns
4. Impact assessment and potential consequences
5. Recommended actions and mitigation strategies

Cite specific shipment IDs and provide detailed reasoning.

ANSWER:"""
        
        elif prompt_type == 'explanation':
            template = """You are LogisticsPulse Copilot, specialized in providing clear explanations of logistics and financial processes.

CONTEXT:
{context}

QUERY: {query}

Please provide:
1. Clear, step-by-step explanation
2. Root cause analysis where applicable
3. Relevant context from policies and procedures
4. Historical patterns or precedents
5. Preventive measures for the future

Use specific examples and data from the context.

ANSWER:"""
        
        else:
            # Use document-type specific template
            template = self.prompt_templates[doc_type]["query"]
        
        return PromptTemplate(
            input_variables=["query", "context"],
            template=template
        )
    
    def _post_process_answer(self, answer, query, doc_type):
        """Post-process answer for consistency and quality"""
        try:
            # Remove redundant phrases
            redundant_phrases = [
                "Based on the provided context,",
                "According to the information given,",
                "From the context provided,"
            ]
            
            for phrase in redundant_phrases:
                answer = answer.replace(phrase, "").strip()
            
            # Ensure answer starts appropriately
            if not answer.startswith(("Based on", "According to", "The", "This", "Here", "From")):
                if doc_type == "invoice":
                    answer = f"Based on the invoice data: {answer}"
                elif doc_type == "shipment":
                    answer = f"According to the shipment information: {answer}"
                else:
                    answer = f"Based on the available data: {answer}"
            
            # Clean up formatting
            answer = answer.replace("\n\n\n", "\n\n")
            answer = answer.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error post-processing answer: {e}")
            return answer
    
    def _calculate_enhanced_confidence(self, answer, docs, query, sources_info):
        """Calculate enhanced confidence score"""
        try:
            base_confidence = 0.6
            
            # Answer quality factors
            if len(answer) > 200:
                base_confidence += 0.1
            if any(phrase in answer.lower() for phrase in ["specific", "exactly", "precisely"]):
                base_confidence += 0.1
            if "I don't have enough information" in answer:
                base_confidence = 0.2
                
            # Source quality factors
            if len(docs) >= 5:
                base_confidence += 0.15
            elif len(docs) >= 3:
                base_confidence += 0.1
            elif len(docs) == 0:
                base_confidence = 0.1
                
            # Relevance factors
            avg_relevance = sum(source.get('relevance_score', 0.5) for source in sources_info) / max(len(sources_info), 1)
            base_confidence += avg_relevance * 0.2
            
            # Query complexity factors
            query_words = len(query.split())
            if query_words <= 3:
                base_confidence += 0.05  # Simple queries are more reliable
            elif query_words > 10:
                base_confidence -= 0.05  # Complex queries are harder
                
            return min(0.95, max(0.1, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence: {e}")
            return 0.5
    
    def _format_enhanced_sources(self, sources_info):
        """Format sources with enhanced information"""
        try:
            formatted_sources = []
            
            for i, source in enumerate(sources_info):
                doc_type = source.get('type', 'unknown')
                source_file = source.get('source', 'unknown')
                doc_id = source.get('doc_id', 'unknown')
                relevance = source.get('relevance_score', 0)
                
                # Create formatted source string
                source_str = f"{doc_type.title()}: {source_file}"
                if doc_id != 'unknown':
                    source_str += f" (ID: {doc_id})"
                if relevance > 0.7:
                    source_str += " [High Relevance]"
                elif relevance > 0.5:
                    source_str += " [Medium Relevance]"
                
                formatted_sources.append(source_str)
            
            return formatted_sources
            
        except Exception as e:
            logger.error(f"Error formatting sources: {e}")
            return [f"Source {i+1}" for i in range(len(sources_info))]