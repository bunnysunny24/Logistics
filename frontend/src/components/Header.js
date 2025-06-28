// src/components/Header.tsx
"use client";

import { useState, useEffect } from 'react';
import { getSystemStatus } from '@/lib/api';

export default function Header() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await getSystemStatus();
        setStatus(data);
      } catch (error) {
        console.error('Error fetching system status:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30 seconds
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <div>
          <h1 className="text-lg font-semibold text-gray-900">Logistics Pulse Copilot</h1>
        </div>
        
        <div className="flex items-center space-x-4">
          {loading ? (
            <div className="text-sm text-gray-500">Loading status...</div>
          ) : status ? (
            <div className="flex items-center space-x-6">
              <div className="text-sm">
                <span className="text-gray-500">Documents: </span>
                <span className="font-medium">{status.documents_processed?.total || 0}</span>
              </div>
              
              <div className="text-sm">
                <span className="text-gray-500">Status: </span>
                <span className={`font-medium ${status.status === 'operational' ? 'text-green-500' : 'text-red-500'}`}>
                  {status.status === 'operational' ? 'Online' : 'Offline'}
                </span>
              </div>
              
              <div className="text-sm">
                <span className="text-gray-500">Last Update: </span>
                <span className="font-medium">
                  {status.last_update ? new Date(status.last_update).toLocaleTimeString() : 'Never'}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-sm text-red-500">Unable to fetch status</div>
          )}
        </div>
      </div>
    </header>
  );
}