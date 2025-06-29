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
    <div className="card h-[600px] flex flex-col">
      <div className="card-header">
        <h5 className="text-lg font-medium">Ask Logistics Pulse Copilot</h5>
        <p className="text-sm text-gray-500">
          Ask questions about invoices, shipments, compliance, or anomalies
        </p>
      </div>
      
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 p-4 overflow-y-auto scrollbar-thin">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <p className="mb-2">No messages yet</p>
              <p className="text-sm">Try asking something like:</p>
              <ul className="text-sm mt-2 space-y-1">
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
                className={`mb-4 ${
                  message.sender === 'user' ? 'text-right' : 'text-left'
                }`}
              >
                <div
                  className={`inline-block p-3 rounded-lg ${
                    message.sender === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                  style={{ maxWidth: '80%', textAlign: 'left' }}
                >
                  <p className="mb-0">{message.text}</p>
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500">
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