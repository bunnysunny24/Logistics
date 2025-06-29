import React, { useState, useEffect } from 'react';
import { getAnomalies } from '../lib/api';
import { FaExclamationTriangle, FaChartBar, FaCalendarAlt, FaFilter, FaChevronDown, FaChevronUp } from 'react-icons/fa';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function AnomalyDashboard({ refreshTrigger = 0 }) {
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [minRiskScore, setMinRiskScore] = useState(0.5);
  const [expandedAnomaly, setExpandedAnomaly] = useState(null);
  const [chartData, setChartData] = useState(null);
  
  useEffect(() => {
    const fetchAnomalies = async () => {
      setLoading(true);
      try {
        const response = await getAnomalies({ minRiskScore });
        // Handle both array response and object with anomalies property
        const anomaliesData = Array.isArray(response) ? response : (response.anomalies || []);
        setAnomalies(anomaliesData);
        
        // Prepare chart data
        prepareChartData(anomaliesData);
      } catch (error) {
        console.error('Error fetching anomalies:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchAnomalies();
  }, [minRiskScore, refreshTrigger]);
  
  const prepareChartData = (anomalyData) => {
    try {
      // Group anomalies by type
      const anomalyTypes = {};
      
      if (Array.isArray(anomalyData)) {
        anomalyData.forEach(anomaly => {
          const type = anomaly.anomaly_type || anomaly.type || 'Unknown';
          if (!anomalyTypes[type]) {
            anomalyTypes[type] = 0;
          }
          anomalyTypes[type]++;
        });
      }
      
      const labels = Object.keys(anomalyTypes);
      const data = Object.values(anomalyTypes);
      
      setChartData({
        labels,
        datasets: [
          {
            label: 'Anomaly Count',
            data,
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
          },
        ],
      });
    } catch (error) {
      console.error('Error preparing chart data:', error);
      setChartData(null);
    }
  };
  
  const toggleAnomaly = (id) => {
    if (expandedAnomaly === id) {
      setExpandedAnomaly(null);
    } else {
      setExpandedAnomaly(id);
    }
  };
  
  const formatTimestamp = (timestamp) => {
    try {
      // Handle both Unix timestamp and ISO string formats
      if (typeof timestamp === 'number') {
        return new Date(timestamp * 1000).toLocaleString();
      } else if (typeof timestamp === 'string') {
        return new Date(timestamp).toLocaleString();
      }
      return 'Unknown';
    } catch (error) {
      console.error('Error formatting timestamp:', error);
      return 'Invalid Date';
    }
  };
  
  const getRiskLevelClass = (score) => {
    if (score >= 0.8) return 'bg-red-100 text-red-800';
    if (score >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-blue-100 text-blue-800';
  };
  
  return (
    <div className="card">
      <div className="card-header flex justify-between items-center">
        <div>
          <h5 className="text-lg font-medium">Anomaly Dashboard</h5>
          <p className="text-sm text-gray-500">
            {anomalies.length} anomalies detected
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div>
            <label className="form-label mb-0">
              Min Risk Score
            </label>
            <select
              value={minRiskScore}
              onChange={(e) => setMinRiskScore(parseFloat(e.target.value))}
              className="form-select text-sm py-1"
            >
              <option value="0">All</option>
              <option value="0.5">Medium (0.5+)</option>
              <option value="0.7">High (0.7+)</option>
              <option value="0.9">Critical (0.9+)</option>
            </select>
          </div>
        </div>
      </div>
      
      <div className="card-body">
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600"></div>
          </div>
        ) : (
          <>
            {chartData && (
              <div className="mb-6">
                <h6 className="text-sm font-medium mb-2">Anomaly Distribution</h6>
                <div className="h-64">
                  <Bar 
                    data={chartData} 
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false,
                        },
                        title: {
                          display: false,
                        },
                      },
                    }} 
                  />
                </div>
              </div>
            )}
            
            <div>
              <h6 className="text-sm font-medium mb-2">Recent Anomalies</h6>
              
              {anomalies.length === 0 ? (
                <div className="text-center p-6 border border-gray-200 rounded-lg">
                  <FaExclamationTriangle className="mx-auto h-8 w-8 text-gray-400 mb-2" />
                  <p className="text-gray-500">No anomalies found with the current filter</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {anomalies.slice(0, 5).map((anomaly) => (
                    <div 
                      key={anomaly.id}
                      className="border border-gray-200 rounded-lg overflow-hidden"
                    >
                      <div 
                        className="flex justify-between items-center p-3 cursor-pointer hover:bg-gray-50"
                        onClick={() => toggleAnomaly(anomaly.id)}
                      >
                        <div className="flex items-center">
                          <span className={`badge mr-3 ${getRiskLevelClass(anomaly.risk_score || 0)}`}>
                            {(anomaly.risk_score || 0).toFixed(2)}
                          </span>
                          <div>
                            <h6 className="text-sm font-medium">
                              {(anomaly.anomaly_type || anomaly.type || 'Unknown').replace(/_/g, ' ')}
                            </h6>
                            <p className="text-xs text-gray-500">
                              Document: {anomaly.document_id || anomaly.id || 'N/A'}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center">
                          <span className="text-xs text-gray-500 mr-2">
                            {formatTimestamp(anomaly.timestamp)}
                          </span>
                          {expandedAnomaly === anomaly.id ? (
                            <FaChevronUp className="text-gray-400 h-4 w-4" />
                          ) : (
                            <FaChevronDown className="text-gray-400 h-4 w-4" />
                          )}
                        </div>
                      </div>
                      
                      {expandedAnomaly === anomaly.id && (
                        <div className="p-3 border-t border-gray-200 bg-gray-50">
                          <p className="text-sm mb-2">{anomaly.description}</p>
                          
                          {anomaly.metadata && (
                            <div className="text-xs text-gray-600">
                              <h6 className="font-medium mb-1">Metadata:</h6>
                              <ul className="space-y-1">
                                {Object.entries(anomaly.metadata).map(([key, value]) => (
                                  <li key={key}>
                                    <span className="font-medium">{key}:</span>{' '}
                                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                  
                  {anomalies.length > 5 && (
                    <div className="text-center">
                      <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                        View all {anomalies.length} anomalies
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default AnomalyDashboard;