// src/app/page.tsx
"use client";

import { useState } from 'react';
import ChatInterface from '@/components/ChatInterface';
import DocumentUploader from '@/components/DocumentUploader';
import AnomalyDashboard from '@/components/AnomalyDashboard';

export default function Home() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  
  const handleDocumentUploaded = () => {
    // Trigger a refresh of the dashboard components
    setRefreshTrigger(prev => prev + 1);
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header Section */}
        <div className="text-center mb-8">
          <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white py-8 px-6 rounded-xl shadow-lg mb-8">
            <h1 className="text-4xl font-bold mb-4 animate-fade-in">
              Logistics Pulse Copilot
            </h1>
            <p className="text-xl text-blue-100 max-w-2xl mx-auto">
              Real-time AI copilot for logistics and finance document processing with intelligent anomaly detection
            </p>
          </div>
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Document Upload Section */}
          <div className="lg:col-span-3">
            <DocumentUploader onUploadComplete={handleDocumentUploaded} />
          </div>
          
          {/* Anomaly Dashboard */}
          <div className="lg:col-span-2">
            <AnomalyDashboard refreshTrigger={refreshTrigger} />
          </div>
          
          {/* Chat Interface */}
          <div className="lg:col-span-1">
            <ChatInterface />
          </div>
        </div>
      </div>
    </div>
  );
}