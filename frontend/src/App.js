import React, { useState } from 'react';
import './App.css';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import DocumentUploader from './components/DocumentUploader';
import AnomalyDashboard from './components/AnomalyDashboard';
import SystemStatusPanel from './components/SystemStatusPanel';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  
  const handleDocumentUploaded = () => {
    // Trigger a refresh of the dashboard components
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        <div className="flex">
          <Sidebar />
          <div className="flex-1 ml-64">
            <Header />
            <div className="container mx-auto px-4 py-6">
              <div className="mb-6">
                <h1 className="text-2xl font-semibold text-gray-800">Logistics Pulse Copilot</h1>
                <p className="text-gray-600">
                  Real-time AI copilot for logistics and finance document processing
                </p>
              </div>
              
              <ErrorBoundary>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                  <div className="lg:col-span-2">
                    <DocumentUploader onUploadComplete={handleDocumentUploaded} />
                  </div>
                  <div>
                    <SystemStatusPanel />
                  </div>
                </div>
              </ErrorBoundary>
              
              <ErrorBoundary>
                <div className="mb-6">
                  <AnomalyDashboard refreshTrigger={refreshTrigger} />
                </div>
              </ErrorBoundary>
              
              <ErrorBoundary>
                <div>
                  <ChatInterface />
                </div>
              </ErrorBoundary>
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;