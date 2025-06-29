import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import { 
  FaCircle, 
  FaCheck, 
  FaTimes, 
  FaSync, 
  FaServer,
  FaDatabase,
  FaRobot,
  FaShieldAlt,
  FaFileAlt
} from 'react-icons/fa';

function SystemStatusPanel() {
  const [status, setStatus] = useState({
    isOnline: false,
    health: null,
    systemStatus: null,
    lastChecked: null,
    error: null
  });
  const [loading, setLoading] = useState(true);

  const checkSystemStatus = async () => {
    try {
      setLoading(true);
      
      // Check basic health
      const healthResponse = await apiService.checkHealth();
      
      // Get detailed system status
      const systemResponse = await apiService.getSystemStatus();
      
      setStatus({
        isOnline: true,
        health: healthResponse,
        systemStatus: systemResponse,
        lastChecked: new Date(),
        error: null
      });
    } catch (error) {
      console.error('System status check failed:', error);
      setStatus({
        isOnline: false,
        health: null,
        systemStatus: null,
        lastChecked: new Date(),
        error: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkSystemStatus();
    
    // Set up periodic health checks every 30 seconds
    const interval = setInterval(checkSystemStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getComponentStatus = (component) => {
    if (!status.systemStatus?.components) return null;
    
    const comp = status.systemStatus.components[component];
    if (typeof comp === 'boolean') {
      return comp;
    }
    if (typeof comp === 'object' && comp !== null) {
      return !comp.error;
    }
    return false;
  };

  const StatusIndicator = ({ isActive, label, details }) => (
    <div className="flex items-center justify-between p-3 border rounded-lg">
      <div className="flex items-center">
        <div className="flex items-center mr-3">
          {isActive ? (
            <FaCircle className="text-green-500 text-xs mr-2" />
          ) : (
            <FaCircle className="text-red-500 text-xs mr-2" />
          )}
          <span className="font-medium">{label}</span>
        </div>
        {details && (
          <span className="text-sm text-gray-500">
            {details}
          </span>
        )}
      </div>
      {isActive ? (
        <FaCheck className="text-green-500" />
      ) : (
        <FaTimes className="text-red-500" />
      )}
    </div>
  );

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <FaServer className="mr-2 text-blue-500" />
          System Status
        </h3>
        <button
          onClick={checkSystemStatus}
          disabled={loading}
          className="flex items-center px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
        >
          <FaSync className={`mr-1 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Overall Status */}
      <div className="mb-4">
        <div className={`flex items-center p-3 rounded-lg ${
          status.isOnline ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
        } border`}>
          <FaCircle className={`text-xs mr-2 ${
            status.isOnline ? 'text-green-500' : 'text-red-500'
          }`} />
          <span className="font-medium">
            Backend API: {status.isOnline ? 'Online' : 'Offline'}
          </span>
          {status.lastChecked && (
            <span className="ml-auto text-sm text-gray-500">
              Last checked: {status.lastChecked.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Component Status */}
      {status.systemStatus && (
        <div className="space-y-2 mb-4">
          <h4 className="font-medium text-gray-700">Components</h4>
          
          <StatusIndicator
            isActive={getComponentStatus('rag_model')}
            label="RAG Model"
            details="Document retrieval and AI responses"
          />
          
          <StatusIndicator
            isActive={getComponentStatus('anomaly_detector')}
            label="Anomaly Detector"
            details="Real-time anomaly detection"
          />
          
          <StatusIndicator
            isActive={getComponentStatus('document_processor')}
            label="Document Processor"
            details="PDF and file processing"
          />
          
          <StatusIndicator
            isActive={status.systemStatus.components?.local_mode}
            label="Local Mode"
            details="Privacy-focused local processing"
          />
        </div>
      )}

      {/* Data Summary */}
      {status.systemStatus?.data_summary && (
        <div className="space-y-2 mb-4">
          <h4 className="font-medium text-gray-700">Data Summary</h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <div className="text-center p-2 bg-blue-50 rounded">
              <div className="text-lg font-bold text-blue-600">
                {status.systemStatus.data_summary.invoices || 0}
              </div>
              <div className="text-xs text-gray-600">Invoices</div>
            </div>
            <div className="text-center p-2 bg-green-50 rounded">
              <div className="text-lg font-bold text-green-600">
                {status.systemStatus.data_summary.shipments || 0}
              </div>
              <div className="text-xs text-gray-600">Shipments</div>
            </div>
            <div className="text-center p-2 bg-purple-50 rounded">
              <div className="text-lg font-bold text-purple-600">
                {status.systemStatus.data_summary.policies || 0}
              </div>
              <div className="text-xs text-gray-600">Policies</div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {status.error && (
        <div className="bg-red-50 border border-red-200 rounded p-3">
          <div className="flex items-center text-red-800">
            <FaTimes className="mr-2" />
            <span className="font-medium">Connection Error</span>
          </div>
          <p className="text-red-600 text-sm mt-1">{status.error}</p>
          <p className="text-red-500 text-xs mt-2">
            Make sure the backend server is running on http://localhost:8000
          </p>
        </div>
      )}

      {/* System Info */}
      {status.systemStatus && (
        <div className="mt-4 pt-4 border-t">
          <div className="text-xs text-gray-500">
            Status: {status.systemStatus.status} | 
            Backend: {status.health?.status} | 
            Timestamp: {status.health?.timestamp}
          </div>
        </div>
      )}
    </div>
  );
}

export default SystemStatusPanel;
