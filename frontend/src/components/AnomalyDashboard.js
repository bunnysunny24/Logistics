import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import { 
  FaExclamationTriangle, 
  FaSync, 
  FaFilter, 
  FaChartBar,
  FaFileAlt,
  FaTruck,
  FaCalendar,
  FaSpinner
} from 'react-icons/fa';

function AnomalyDashboard() {
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    min_risk_score: 0,
    severity: '',
    doc_type: '',
    anomaly_type: '',
    start_date: '',
    end_date: ''
  });
  const [stats, setStats] = useState({
    total: 0,
    high_risk: 0,
    medium_risk: 0,
    low_risk: 0
  });
  const [isDetecting, setIsDetecting] = useState(false);

  useEffect(() => {
    loadAnomalies();
  }, [filters]); // loadAnomalies is called inside, dependency on filters is correct

  const loadAnomalies = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await apiService.getAnomalies(filters);
      setAnomalies(data);
      
      // Calculate stats
      const high_risk = data.filter(a => a.risk_score >= 0.8).length;
      const medium_risk = data.filter(a => a.risk_score >= 0.5 && a.risk_score < 0.8).length;
      const low_risk = data.filter(a => a.risk_score < 0.5).length;
      
      setStats({
        total: data.length,
        high_risk,
        medium_risk,
        low_risk
      });
    } catch (err) {
      console.error('Error loading anomalies:', err);
      setError('Failed to load anomalies. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const triggerDetection = async () => {
    try {
      setIsDetecting(true);
      const result = await apiService.triggerAnomalyDetection();
      
      if (result.success) {
        // Reload anomalies after detection
        await loadAnomalies();
        
        // Show success message
        alert(`Detection complete! Found ${result.anomalies} anomalies.`);
      } else {
        alert(`Detection failed: ${result.message}`);
      }
    } catch (err) {
      console.error('Error triggering detection:', err);
      alert('Failed to trigger anomaly detection.');
    } finally {
      setIsDetecting(false);
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const clearFilters = () => {
    setFilters({
      min_risk_score: 0,
      severity: '',
      doc_type: '',
      anomaly_type: '',
      start_date: '',
      end_date: ''
    });
  };

  const getRiskColor = (score) => {
    if (score >= 0.8) return 'text-red-600 bg-red-100';
    if (score >= 0.5) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high': return 'bg-red-500 text-white';
      case 'medium': return 'bg-yellow-500 text-white';
      case 'low': return 'bg-green-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  if (loading && anomalies.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <FaSpinner className="animate-spin text-4xl text-blue-500" />
        <span className="ml-3 text-xl">Loading anomalies...</span>
      </div>
    );
  }

  return (
    <div className="anomaly-dashboard p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800 flex items-center">
            <FaExclamationTriangle className="mr-3 text-red-500" />
            Anomaly Detection Dashboard
          </h2>
          <p className="text-gray-600 mt-1">Real-time monitoring of logistics and financial anomalies</p>
        </div>
        
        <div className="flex gap-3">
          <button
            onClick={triggerDetection}
            disabled={isDetecting}
            className="flex items-center px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            {isDetecting ? (
              <FaSpinner className="animate-spin mr-2" />
            ) : (
              <FaSync className="mr-2" />
            )}
            {isDetecting ? 'Detecting...' : 'Trigger Detection'}
          </button>
          
          <button
            onClick={loadAnomalies}
            className="flex items-center px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
          >
            <FaSync className="mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Anomalies</p>
              <p className="text-2xl font-bold text-gray-800">{stats.total}</p>
            </div>
            <FaChartBar className="text-3xl text-blue-500" />
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">High Risk</p>
              <p className="text-2xl font-bold text-red-600">{stats.high_risk}</p>
            </div>
            <FaExclamationTriangle className="text-3xl text-red-500" />
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Medium Risk</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.medium_risk}</p>
            </div>
            <FaExclamationTriangle className="text-3xl text-yellow-500" />
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Low Risk</p>
              <p className="text-2xl font-bold text-green-600">{stats.low_risk}</p>
            </div>
            <FaExclamationTriangle className="text-3xl text-green-500" />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-lg shadow-sm border mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center">
            <FaFilter className="mr-2" />
            Filters
          </h3>
          <button
            onClick={clearFilters}
            className="text-sm text-blue-500 hover:text-blue-700"
          >
            Clear All
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Min Risk Score
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={filters.min_risk_score}
              onChange={(e) => handleFilterChange('min_risk_score', parseFloat(e.target.value) || 0)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Severity
            </label>
            <select
              value={filters.severity}
              onChange={(e) => handleFilterChange('severity', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Severities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Document Type
            </label>
            <select
              value={filters.doc_type}
              onChange={(e) => handleFilterChange('doc_type', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Types</option>
              <option value="invoice">Invoice</option>
              <option value="shipment">Shipment</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Anomaly Type
            </label>
            <input
              type="text"
              value={filters.anomaly_type}
              onChange={(e) => handleFilterChange('anomaly_type', e.target.value)}
              placeholder="e.g., amount_deviation"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Start Date
            </label>
            <input
              type="datetime-local"
              value={filters.start_date}
              onChange={(e) => handleFilterChange('start_date', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              End Date
            </label>
            <input
              type="datetime-local"
              value={filters.end_date}
              onChange={(e) => handleFilterChange('end_date', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Anomalies List */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-4 border-b">
          <h3 className="text-lg font-semibold">Detected Anomalies ({anomalies.length})</h3>
        </div>
        
        {loading ? (
          <div className="flex items-center justify-center p-8">
            <FaSpinner className="animate-spin text-2xl text-blue-500 mr-3" />
            <span>Loading anomalies...</span>
          </div>
        ) : anomalies.length === 0 ? (
          <div className="text-center p-8 text-gray-500">
            <FaExclamationTriangle className="text-4xl mx-auto mb-4 opacity-50" />
            <p>No anomalies found matching the current filters.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {anomalies.map((anomaly) => (
              <div key={anomaly.id} className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="flex items-center">
                        {anomaly.anomaly_type?.includes('invoice') ? (
                          <FaFileAlt className="text-blue-500 mr-1" />
                        ) : (
                          <FaTruck className="text-green-500 mr-1" />
                        )}
                        <span className="font-semibold text-gray-800">
                          {anomaly.document_id}
                        </span>
                      </span>
                      
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(anomaly.severity)}`}>
                        {anomaly.severity?.toUpperCase()}
                      </span>
                      
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(anomaly.risk_score)}`}>
                        Risk: {(anomaly.risk_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <h4 className="font-medium text-gray-900 mb-2">
                      {anomaly.anomaly_type?.replace(/_/g, ' ').toUpperCase()}
                    </h4>
                    
                    <p className="text-gray-600 mb-3">{anomaly.description}</p>
                    
                    {anomaly.evidence && anomaly.evidence.length > 0 && (
                      <div className="mb-3">
                        <h5 className="text-sm font-medium text-gray-700 mb-1">Evidence:</h5>
                        <ul className="text-sm text-gray-600 list-disc list-inside">
                          {anomaly.evidence.map((item, idx) => (
                            <li key={idx}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {anomaly.recommendations && anomaly.recommendations.length > 0 && (
                      <div className="mb-3">
                        <h5 className="text-sm font-medium text-gray-700 mb-1">Recommendations:</h5>
                        <ul className="text-sm text-gray-600 list-disc list-inside">
                          {anomaly.recommendations.map((item, idx) => (
                            <li key={idx}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                  
                  <div className="text-right text-sm text-gray-500">
                    <div className="flex items-center mb-1">
                      <FaCalendar className="mr-1" />
                      {formatTimestamp(anomaly.timestamp)}
                    </div>
                    {anomaly.metadata?.mock_data && (
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                        Demo Data
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default AnomalyDashboard;
