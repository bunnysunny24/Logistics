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
    <header className="bg-white shadow-sm border-bottom p-3">
      <div className="d-flex justify-content-between align-items-center">
        <h1 className="h5 mb-0">Logistics Pulse Copilot</h1>
        
        <div className="d-flex align-items-center">
          {loading ? (
            <div className="text-muted small">Loading status...</div>
          ) : status ? (
            <div className="d-flex align-items-center gap-4">
              <div className="small">
                <span className="text-muted">Documents: </span>
                <span className="fw-medium">{status.documents_processed?.total || 0}</span>
              </div>
              
              <div className="small">
                <span className="text-muted">Status: </span>
                <span className={`fw-medium ${status.status === 'operational' ? 'text-success' : 'text-danger'}`}>
                  {status.status === 'operational' ? 'Online' : 'Offline'}
                </span>
              </div>
              
              <div className="small">
                <span className="text-muted">Last Update: </span>
                <span className="fw-medium">
                  {status.last_update ? new Date(status.last_update).toLocaleTimeString() : 'Never'}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-danger small">Unable to fetch status</div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;