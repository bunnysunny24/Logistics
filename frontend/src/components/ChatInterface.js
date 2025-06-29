import React, { useState, useRef, useEffect } from 'react';
import { queryLogisticsCopilot } from '../lib/api';
import { FaArrowCircleRight, FaSpinner } from 'react-icons/fa';

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
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
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await queryLogisticsCopilot(input);
      
      const copilotMessage = {
        id: (Date.now() + 1).toString(),
        text: response.answer,
        sender: 'copilot',
        timestamp: new Date(),
        sources: response.sources,
      };
      
      setMessages((prev) => [...prev, copilotMessage]);
    } catch (error) {
      console.error('Error querying copilot:', error);
      
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'copilot',
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="card" style={{ height: '600px' }}>
      <div className="card-header">
        <h5 className="card-title mb-0">Ask Logistics Pulse Copilot</h5>
        <p className="text-muted small mb-0">
          Ask questions about invoices, shipments, compliance, or anomalies
        </p>
      </div>
      
      <div className="card-body d-flex flex-column p-0">
        <div className="flex-grow-1 p-3 overflow-auto scrollbar-thin" style={{ maxHeight: 'calc(600px - 140px)' }}>
          {messages.length === 0 ? (
            <div className="d-flex flex-column align-items-center justify-content-center h-100 text-muted">
              <p className="mb-2">No messages yet</p>
              <p className="small">Try asking something like:</p>
              <ul className="small mt-2">
                <li>"What are the late fees for invoice #123?"</li>
                <li>"Why was Shipment #234 flagged?"</li>
                <li>"Which suppliers offer early payment discounts?"</li>
                <li>"List all anomalies detected today"</li>
              </ul>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`mb-3 ${
                  message.sender === 'user' ? 'text-end' : 'text-start'
                }`}
              >
                <div
                  className={`d-inline-block p-3 rounded ${
                    message.sender === 'user'
                      ? 'bg-primary text-white'
                      : 'bg-light text-dark'
                  }`}
                  style={{ maxWidth: '80%', textAlign: 'left' }}
                >
                  <p className="mb-0">{message.text}</p>
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-2 pt-2 border-top small text-muted">
                      <p className="fw-bold mb-1">Sources:</p>
                      <ul className="mb-0 ps-3">
                        {message.sources.slice(0, 3).map((source, idx) => (
                          <li key={idx}>
                            {source.metadata?.filename || 'Document'} 
                            {source.metadata?.doc_type && ` (${source.metadata.doc_type})`}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                <div className={`small text-muted mt-1 ${
                  message.sender === 'user' ? 'text-end' : 'text-start'
                }`}>
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="p-3 border-top">
          <form onSubmit={handleSubmit} className="d-flex">
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
              className="btn btn-primary ms-2"
            >
              {isLoading ? (
                <FaSpinner className="fa-spin" />
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