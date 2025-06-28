// src/components/ChatInterface.tsx
"use client";

import { useState, useRef, useEffect } from 'react';
import { queryLogisticsCopilot } from '@/lib/api';
import { FiSend, FiLoader } from 'react-icons/fi';

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'copilot';
  timestamp: Date;
  sources?: Source[];
};

type Source = {
  metadata?: {
    filename?: string;
    doc_type?: string;
  };
};

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    const userMessage: Message = {
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
      
      const copilotMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.answer,
        sender: 'copilot',
        timestamp: new Date(),
        sources: response.sources,
      };
      
      setMessages((prev) => [...prev, copilotMessage]);
    } catch (error) {
      console.error('Error querying copilot:', error);
      
      const errorMessage: Message = {
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
    <div className="flex flex-col h-[600px] bg-white shadow-md rounded-lg">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold">Ask Logistics Pulse Copilot</h2>
        <p className="text-sm text-gray-500">
          Ask questions about invoices, shipments, compliance, or anomalies
        </p>
      </div>
      
      <div className="flex-1 p-4 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <p className="mb-2">No messages yet</p>
            <p className="text-sm">Try asking something like:</p>
            <ul className="text-sm mt-2 space-y-1">
              <li>• &ldquo;What are the late fees for invoice #123?&rdquo;</li>
              <li>• &ldquo;Why was Shipment #234 flagged?&rdquo;</li>
              <li>• &ldquo;Which suppliers offer early payment discounts?&rdquo;</li>
              <li>• &ldquo;List all anomalies detected today&rdquo;</li>
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
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-800'
                }`}
                style={{ maxWidth: '80%', textAlign: 'left' }}
              >
                <p>{message.text}</p>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500">
                    <p className="font-semibold">Sources:</p>
                    <ul className="mt-1 space-y-1">
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
              <div
                className={`text-xs text-gray-500 mt-1 ${
                  message.sender === 'user' ? 'text-right' : 'text-left'
                }`}
              >
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
            className="flex-1 p-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="bg-blue-500 text-white p-2 rounded-r-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {isLoading ? (
              <FiLoader className="h-5 w-5 animate-spin" />
            ) : (
              <FiSend className="h-5 w-5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}