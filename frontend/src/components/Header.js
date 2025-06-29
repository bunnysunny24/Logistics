import React, { useState, useEffect } from 'react';
import { getSystemStatus } from '../lib/api';

function Header() {
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
    <header className="bg-white shadow-sm border-b border-gray-200 px-4 py-3">
      <div className="flex justify-between items-center">
        <h1 className="text-lg font-medium text-gray-800">Logistics Pulse Copilot</h1>
        
        <div className="flex items-center">
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
                <span className={`font-medium ${status.status === 'operational' ? 'text-green-600' : 'text-red-600'}`}>
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
            <div className="text-sm text-red-600">Unable to fetch status</div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;