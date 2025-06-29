import React, { useState, useRef, useEffect } from 'react';
import { queryLogisticsCopilot } from '../lib/api';
import { FaArrowCircleRight, FaSpinner, FaLightbulb, FaHistory, FaCopy } from 'react-icons/fa';

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [queryHistory, setQueryHistory] = useState([]);
  const messagesEndRef = useRef(null);
  
  // Enhanced query suggestions based on context
  const smartSuggestions = {
    invoice: [
      "What are the late fees for overdue invoices?",
      "Which suppliers offer early payment discounts?",
      "Show me invoices requiring manager approval",
      "Calculate total outstanding amounts"
    ],
    shipment: [
      "Why was shipment #SH001 flagged as anomalous?", 
      "List all high-risk shipments this week",
      "Show delivery delays for Express Worldwide",
      "What are the current route deviations?"
    ],
    compliance: [
      "Check compliance status for recent invoices",
      "What are the current approval thresholds?",
      "Show policy violations this month",
      "List emergency processing protocols"
    ],
    general: [
      "Show me the latest anomalies detected",
      "What's the current system status?",
      "Generate a risk assessment report",
      "Find all items needing attention"
    ]
  };
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Update suggestions based on input
  useEffect(() => {
    if (input.length > 2) {
      const inputLower = input.toLowerCase();
      let relevantSuggestions = [];
      
      if (inputLower.includes('invoice') || inputLower.includes('payment')) {
        relevantSuggestions = smartSuggestions.invoice;
      } else if (inputLower.includes('ship') || inputLower.includes('delivery')) {
        relevantSuggestions = smartSuggestions.shipment;
      } else if (inputLower.includes('compliance') || inputLower.includes('policy')) {
        relevantSuggestions = smartSuggestions.compliance;
      } else {
        relevantSuggestions = smartSuggestions.general;
      }
      
      // Filter suggestions that match current input
      const filtered = relevantSuggestions.filter(s => 
        s.toLowerCase().includes(inputLower) || 
        inputLower.split(' ').some(word => s.toLowerCase().includes(word))
      );
      
      setSuggestions(filtered.slice(0, 3));
      setShowSuggestions(filtered.length > 0);
    } else {
      setShowSuggestions(false);
    }
  }, [input]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    const userMessage = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    
    // Add to query history
    const newQuery = input.trim();
    setQueryHistory(prev => {
      const updated = [newQuery, ...prev.filter(q => q !== newQuery)].slice(0, 10);
      localStorage.setItem('logistics-query-history', JSON.stringify(updated));
      return updated;
    });
    
    setInput('');
    setShowSuggestions(false);
    setIsLoading(true);
    
    try {
      const startTime = Date.now();
      const response = await queryLogisticsCopilot(newQuery);
      const responseTime = Date.now() - startTime;
      
      const copilotMessage = {
        id: (Date.now() + 1).toString(),
        text: response.answer,
        sender: 'copilot',
        timestamp: new Date(),
        sources: response.sources,
        confidence: response.confidence,
        responseTime: responseTime,
        metadata: response.metadata
      };
      
      setMessages((prev) => [...prev, copilotMessage]);
    } catch (error) {
      console.error('Error querying copilot:', error);
      
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error processing your request. Please check if the backend is running and try again.',
        sender: 'copilot',
        timestamp: new Date(),
        isError: true
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion);
    setShowSuggestions(false);
  };
  
  const handleHistoryClick = (query) => {
    setInput(query);
  };
  
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };
  
  // Load query history on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('logistics-query-history');
    if (savedHistory) {
      try {
        setQueryHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Error loading query history:', e);
      }
    }
  }, []);
  
  const renderMessage = (message) => {
    const isUser = message.sender === 'user';
    
    return (
      <div
        key={message.id}
        className={`mb-4 ${isUser ? 'text-right' : 'text-left'}`}
      >
        <div
          className={`inline-block p-3 rounded-lg ${
            isUser
              ? 'bg-primary-600 text-white'
              : message.isError
              ? 'bg-red-100 text-red-800 border border-red-200'
              : 'bg-gray-100 text-gray-800'
          }`}
          style={{ maxWidth: '80%', textAlign: 'left' }}
        >
          <p className="mb-0 whitespace-pre-wrap">{message.text}</p>
          
          {/* Enhanced message metadata for copilot responses */}
          {!isUser && !message.isError && (
            <div className="mt-3 space-y-2">
              {/* Confidence indicator */}
              {message.confidence && (
                <div className="flex items-center text-xs text-gray-500">
                  <span className="mr-2">Confidence:</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        message.confidence > 0.8 ? 'bg-green-500' :
                        message.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${message.confidence * 100}%` }}
                    />
                  </div>
                  <span className="ml-2">{(message.confidence * 100).toFixed(0)}%</span>
                </div>
              )}
              
              {/* Sources */}
              {message.sources && message.sources.length > 0 && (
                <div className="text-xs text-gray-500">
                  <strong>Sources:</strong>
                  <ul className="mt-1 ml-4 space-y-1">
                    {message.sources.map((source, idx) => (
                      <li key={idx} className="list-disc">{source}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Response time and metadata */}
              <div className="flex items-center justify-between text-xs text-gray-400">
                {message.responseTime && (
                  <span>Response time: {message.responseTime}ms</span>
                )}
                {message.metadata?.documents_retrieved && (
                  <span>{message.metadata.documents_retrieved} docs analyzed</span>
                )}
                <button
                  onClick={() => copyToClipboard(message.text)}
                  className="text-gray-400 hover:text-gray-600"
                  title="Copy response"
                >
                  <FaCopy size={12} />
                </button>
              </div>
            </div>
          )}
        </div>
        
        {/* Timestamp */}
        <div className={`text-xs text-gray-400 mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    );
  };
  
  return (
    <div className="card h-[700px] flex flex-col">
      <div className="card-header">
        <h5 className="text-lg font-medium flex items-center">
          <FaLightbulb className="mr-2 text-yellow-500" />
          Ask Logistics Pulse Copilot
        </h5>
        <p className="text-sm text-gray-500">
          AI-powered insights for invoices, shipments, compliance, and anomaly detection
        </p>
      </div>
      
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 p-4 overflow-y-auto scrollbar-thin">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <FaLightbulb size={48} className="mb-4 text-gray-300" />
              <p className="mb-4 text-lg font-medium">Ready to help with your logistics queries</p>
              <p className="text-sm mb-4">Try asking something like:</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full max-w-4xl">
                {smartSuggestions.general.map((suggestion, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg border text-sm transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
              
              {/* Query history */}
              {queryHistory.length > 0 && (
                <div className="mt-6 w-full max-w-2xl">
                  <h6 className="text-sm font-medium text-gray-600 mb-2 flex items-center">
                    <FaHistory className="mr-2" />
                    Recent Queries
                  </h6>
                  <div className="space-y-1">
                    {queryHistory.slice(0, 5).map((query, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleHistoryClick(query)}
                        className="text-left w-full p-2 text-xs text-gray-500 hover:text-gray-700 hover:bg-gray-50 rounded transition-colors"
                      >
                        {query}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            messages.map(renderMessage)
          )}
          
          {/* Loading indicator */}
          {isLoading && (
            <div className="text-left mb-4">
              <div className="inline-block p-3 bg-gray-100 rounded-lg">
                <div className="flex items-center text-gray-600">
                  <FaSpinner className="animate-spin mr-2" />
                  <span>Analyzing your query...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input area with suggestions */}
        <div className="border-t p-4 relative">
          {/* Suggestions dropdown */}
          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute bottom-full left-4 right-4 mb-2 bg-white border border-gray-200 rounded-lg shadow-lg z-10">
              {suggestions.map((suggestion, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="w-full text-left p-3 hover:bg-gray-50 first:rounded-t-lg last:rounded-b-lg border-b last:border-b-0 text-sm"
                >
                  <FaLightbulb className="inline mr-2 text-yellow-500" size={12} />
                  {suggestion}
                </button>
              ))}
            </div>
          )}
          
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about invoices, shipments, compliance, or anomalies..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              {isLoading ? (
                <FaSpinner className="animate-spin" />
              ) : (
                <FaArrowCircleRight />
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
                      <p className="font-semibold mb-1">Sources:</p>
                      <ul className="space-y-1 pl-4">
                        {message.sources.slice(0, 3).map((source, idx) => (
                          <li key={idx} className="list-disc">
                            {source.metadata?.filename || 'Document'} 
                            {source.metadata?.doc_type && ` (${source.metadata.doc_type})`}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                <div className={`text-xs text-gray-500 mt-1 ${
                  message.sender === 'user' ? 'text-right' : 'text-left'
                }`}>
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="p-4 border-t border-gray-200">
          <form onSubmit={handleSubmit} className="flex">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
              placeholder="Type your question here..."
              className="form-control"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="ml-2 bg-primary-600 hover:bg-primary-700 text-white p-2 rounded-md"
            >
              {isLoading ? (
                <FaSpinner className="animate-spin h-5 w-5" />
              ) : (
                <FaArrowCircleRight className="h-5 w-5" />
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;