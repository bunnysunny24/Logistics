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
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="col-span-1 md:col-span-2">
        <h1 className="text-2xl font-bold mb-4">Logistics Pulse Copilot</h1>
        <p className="mb-6 text-gray-600">
          Real-time AI copilot for logistics and finance document processing
        </p>
      </div>
      
      <div className="col-span-1 md:col-span-2">
        <DocumentUploader onUploadComplete={handleDocumentUploaded} />
      </div>
      
      <div className="col-span-1 md:col-span-2">
        <AnomalyDashboard refreshTrigger={refreshTrigger} />
      </div>
      
      <div className="col-span-1 md:col-span-2">
        <ChatInterface />
      </div>
    </div>
  );
}