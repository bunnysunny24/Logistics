// src/components/AnomalyDashboard.tsx
"use client";

import { useState, useEffect } from 'react';
import { getAnomalies } from '@/lib/api';
import { FiAlertTriangle, FiChevronDown, FiChevronUp } from 'react-icons/fi';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ChartData } from 'chart.js';
import { Bar } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

type Props = {
  refreshTrigger?: number;
};

type Anomaly = {
  id: string;
  anomaly_type: string;
  risk_score: number;
  document_id: string;
  timestamp: number;
  description: string;
  metadata?: Record<string, unknown>;
};

export default function AnomalyDashboard({ refreshTrigger = 0 }: Props) {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [loading, setLoading] = useState(true);
  const [minRiskScore, setMinRiskScore] = useState(0.5);
  const [expandedAnomaly, setExpandedAnomaly] = useState<string | null>(null);
  const [chartData, setChartData] = useState<ChartData<'bar'> | null>(null);
  
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
  
  const prepareChartData = (anomalyData: Anomaly[]) => {
    // Group anomalies by type
    const anomalyTypes: Record<string, number> = {};
    
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
  
  const toggleAnomaly = (id: string) => {
    if (expandedAnomaly === id) {
      setExpandedAnomaly(null);
    } else {
      setExpandedAnomaly(id);
    }
  };
  
  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };
  
  const getRiskLevelClass = (score: number) => {
    if (score >= 0.8) return 'bg-red-100 text-red-800';
    if (score >= 0.6) return 'bg-orange-100 text-orange-800';
    return 'bg-yellow-100 text-yellow-800';
  };
  
  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-lg font-semibold">Anomaly Dashboard</h2>
          <p className="text-sm text-gray-500">
            {anomalies.length} anomalies detected
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Min Risk Score
            </label>
            <select
              value={minRiskScore}
              onChange={(e) => setMinRiskScore(parseFloat(e.target.value))}
              className="border border-gray-300 rounded-md p-1 text-sm"
            >
              <option value="0">All</option>
              <option value="0.5">Medium (0.5+)</option>
              <option value="0.7">High (0.7+)</option>
              <option value="0.9">Critical (0.9+)</option>
            </select>
          </div>
        </div>
      </div>
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : (
        <>
          {chartData && (
            <div className="mb-6">
              <h3 className="text-md font-medium mb-2">Anomaly Distribution</h3>
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
            <h3 className="text-md font-medium mb-2">Recent Anomalies</h3>
            
            {anomalies.length === 0 ? (
              <div className="text-center p-6 border border-gray-200 rounded-lg">
                <FiAlertTriangle className="mx-auto h-8 w-8 text-gray-400" />
                <p className="mt-2 text-gray-500">No anomalies found with the current filter</p>
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
                        <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium mr-3 ${getRiskLevelClass(anomaly.risk_score)}`}>
                          {anomaly.risk_score.toFixed(2)}
                        </div>
                        <div>
                          <h4 className="text-sm font-medium">{anomaly.anomaly_type.replace(/_/g, ' ')}</h4>
                          <p className="text-xs text-gray-500">
                            Document: {anomaly.document_id}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center">
                        <span className="text-xs text-gray-500 mr-2">
                          {formatTimestamp(anomaly.timestamp)}
                        </span>
                        {expandedAnomaly === anomaly.id ? (
                          <FiChevronUp className="h-4 w-4 text-gray-400" />
                        ) : (
                          <FiChevronDown className="h-4 w-4 text-gray-400" />
                        )}
                      </div>
                    </div>
                    
                    {expandedAnomaly === anomaly.id && (
                      <div className="p-3 border-t border-gray-200 bg-gray-50">
                        <p className="text-sm mb-2">{anomaly.description}</p>
                        
                        {anomaly.metadata && (
                          <div className="text-xs text-gray-600">
                            <h5 className="font-medium mb-1">Metadata:</h5>
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
                    <button className="text-blue-500 hover:text-blue-700 text-sm font-medium">
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
  );
}