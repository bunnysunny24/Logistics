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
        setAnomalies(response.anomalies || []);
        
        // Prepare chart data
        prepareChartData(response.anomalies || []);
      } catch (error) {
        console.error('Error fetching anomalies:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchAnomalies();
  }, [minRiskScore, refreshTrigger]);
  
  const prepareChartData = (anomalyData) => {
    // Group anomalies by type
    const anomalyTypes = {};
    
    anomalyData.forEach(anomaly => {
      const type = anomaly.anomaly_type;
      if (!anomalyTypes[type]) {
        anomalyTypes[type] = 0;
      }
      anomalyTypes[type]++;
    });
    
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
  };
  
  const toggleAnomaly = (id) => {
    if (expandedAnomaly === id) {
      setExpandedAnomaly(null);
    } else {
      setExpandedAnomaly(id);
    }
  };
  
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };
  
  const getRiskLevelClass = (score) => {
    if (score >= 0.8) return 'bg-danger text-white';
    if (score >= 0.6) return 'bg-warning text-dark';
    return 'bg-info text-dark';
  };
  
  return (
    <div className="card mb-4">
      <div className="card-header d-flex justify-content-between align-items-center">
        <div>
          <h5 className="card-title mb-0">Anomaly Dashboard</h5>
          <p className="text-muted small mb-0">
            {anomalies.length} anomalies detected
          </p>
        </div>
        
        <div className="d-flex align-items-center gap-3">
          <div>
            <label className="form-label small mb-0">
              Min Risk Score
            </label>
            <select
              value={minRiskScore}
              onChange={(e) => setMinRiskScore(parseFloat(e.target.value))}
              className="form-select form-select-sm"
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
          <div className="d-flex justify-content-center align-items-center" style={{ height: '300px' }}>
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : (
          <>
            {chartData && (
              <div className="mb-4">
                <h6 className="card-subtitle mb-2">Anomaly Distribution</h6>
                <div style={{ height: '250px' }}>
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
              <h6 className="card-subtitle mb-2">Recent Anomalies</h6>
              
              {anomalies.length === 0 ? (
                <div className="text-center p-4 border rounded">
                  <FaExclamationTriangle className="mb-2" style={{ fontSize: '2rem', opacity: 0.5 }} />
                  <p className="text-muted">No anomalies found with the current filter</p>
                </div>
              ) : (
                <div className="d-flex flex-column gap-3">
                  {anomalies.slice(0, 5).map((anomaly) => (
                    <div 
                      key={anomaly.id}
                      className="border rounded overflow-hidden"
                    >
                      <div 
                        className="d-flex justify-content-between align-items-center p-3 cursor-pointer hover-bg-light"
                        onClick={() => toggleAnomaly(anomaly.id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <div className="d-flex align-items-center">
                          <span className={`badge me-2 ${getRiskLevelClass(anomaly.risk_score)}`}>
                            {anomaly.risk_score.toFixed(2)}
                          </span>
                          <div>
                            <h6 className="mb-0">{anomaly.anomaly_type.replace(/_/g, ' ')}</h6>
                            <p className="mb-0 small text-muted">
                              Document: {anomaly.document_id}
                            </p>
                          </div>
                        </div>
                        
                        <div className="d-flex align-items-center">
                          <span className="small text-muted me-2">
                            {formatTimestamp(anomaly.timestamp)}
                          </span>
                          {expandedAnomaly === anomaly.id ? (
                            <FaChevronUp className="text-muted" />
                          ) : (
                            <FaChevronDown className="text-muted" />
                          )}
                        </div>
                      </div>
                      
                      {expandedAnomaly === anomaly.id && (
                        <div className="p-3 border-top bg-light">
                          <p className="mb-2">{anomaly.description}</p>
                          
                          {anomaly.metadata && (
                            <div className="small text-muted">
                              <h6 className="mb-1">Metadata:</h6>
                              <ul className="mb-0">
                                {Object.entries(anomaly.metadata).map(([key, value]) => (
                                  <li key={key}>
                                    <span className="fw-medium">{key}:</span>{' '}
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
                      <button className="btn btn-link text-decoration-none">
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