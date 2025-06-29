import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import { 
  FaShieldAlt, 
  FaExclamationTriangle, 
  FaCheckCircle, 
  FaClock, 
  FaArrowRight,
  FaChartPie,
  FaListAlt,
  FaFileAlt
} from 'react-icons/fa';

function RiskBasedHoldsPanel({ refreshTrigger = 0 }) {
  const [holds, setHolds] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all'); // all, active, resolved

  useEffect(() => {
    const fetchHolds = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiService.getRiskBasedHolds();
        setHolds(response.holds || []);
      } catch (err) {
        console.error('Error fetching risk-based holds:', err);
        setError('Failed to load risk-based holds');
      } finally {
        setLoading(false);
      }
    };

    fetchHolds();
  }, [refreshTrigger]);

  const filteredHolds = holds.filter(hold => {
    if (filter === 'active') return hold.status === 'active';
    if (filter === 'resolved') return hold.status === 'resolved';
    return true;
  });

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return <FaExclamationTriangle className="text-red-500" />;
      case 'resolved':
        return <FaCheckCircle className="text-green-500" />;
      default:
        return <FaClock className="text-yellow-500" />;
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'high':
        return 'text-red-600 bg-red-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      case 'low':
        return 'text-green-600 bg-green-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getSummaryStats = () => {
    const activeHolds = holds.filter(h => h.status === 'active').length;
    const resolvedHolds = holds.filter(h => h.status === 'resolved').length;
    const highRiskHolds = holds.filter(h => h.risk_level === 'high').length;
    
    return { activeHolds, resolvedHolds, highRiskHolds, totalHolds: holds.length };
  };

  const stats = getSummaryStats();

  if (loading) {
    return (
      <div className="p-6 bg-white rounded-lg shadow">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading risk-based holds...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-white rounded-lg shadow">
        <div className="text-center text-red-600">
          <FaExclamationTriangle className="mx-auto h-8 w-8 mb-2" />
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow border-l-4 border-blue-500">
          <div className="flex items-center">
            <FaListAlt className="text-blue-500 text-xl mr-3" />
            <div>
              <p className="text-sm text-gray-600">Total Holds</p>
              <p className="text-2xl font-bold text-gray-800">{stats.totalHolds}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow border-l-4 border-red-500">
          <div className="flex items-center">
            <FaExclamationTriangle className="text-red-500 text-xl mr-3" />
            <div>
              <p className="text-sm text-gray-600">Active</p>
              <p className="text-2xl font-bold text-gray-800">{stats.activeHolds}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow border-l-4 border-green-500">
          <div className="flex items-center">
            <FaCheckCircle className="text-green-500 text-xl mr-3" />
            <div>
              <p className="text-sm text-gray-600">Resolved</p>
              <p className="text-2xl font-bold text-gray-800">{stats.resolvedHolds}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow border-l-4 border-orange-500">
          <div className="flex items-center">
            <FaShieldAlt className="text-orange-500 text-xl mr-3" />
            <div>
              <p className="text-sm text-gray-600">High Risk</p>
              <p className="text-2xl font-bold text-gray-800">{stats.highRiskHolds}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Filter Controls */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-700">Filter:</span>
          <div className="flex space-x-2">
            {['all', 'active', 'resolved'].map(filterType => (
              <button
                key={filterType}
                onClick={() => setFilter(filterType)}
                className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                  filter === filterType
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {filterType.charAt(0).toUpperCase() + filterType.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Holds List */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 bg-gray-50 border-b">
          <div className="flex items-center">
            <FaShieldAlt className="text-gray-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-800">Risk-Based Holds</h3>
            <span className="ml-2 bg-gray-200 text-gray-700 px-2 py-1 rounded-full text-sm">
              {filteredHolds.length}
            </span>
          </div>
        </div>

        {filteredHolds.length === 0 ? (
          <div className="p-8 text-center">
            <FaShieldAlt className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <p className="text-gray-600">No risk-based holds found</p>
            {filter !== 'all' && (
              <button
                onClick={() => setFilter('all')}
                className="mt-2 text-blue-600 hover:text-blue-800"
              >
                View all holds
              </button>
            )}
          </div>
        ) : (
          <div className="divide-y">
            {filteredHolds.map((hold, index) => (
              <div key={index} className="p-6 hover:bg-gray-50 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center mb-2">
                      {getStatusIcon(hold.status)}
                      <h4 className="ml-2 text-lg font-medium text-gray-800">
                        {hold.hold_type || 'Risk Hold'}
                      </h4>
                      <span className={`ml-3 px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(hold.risk_level)}`}>
                        {hold.risk_level || 'Unknown'} Risk
                      </span>
                    </div>
                    
                    <p className="text-gray-600 mb-3">{hold.reason || 'No reason provided'}</p>
                    
                    {hold.affected_entities && hold.affected_entities.length > 0 && (
                      <div className="mb-3">
                        <p className="text-sm font-medium text-gray-700 mb-1">Affected:</p>
                        <div className="flex flex-wrap gap-2">
                          {hold.affected_entities.map((entity, idx) => (
                            <span key={idx} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                              <FaFileAlt className="inline mr-1" />
                              {entity.type}: {entity.id}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {hold.causal_chain && hold.causal_chain.length > 0 && (
                      <div className="mb-3">
                        <p className="text-sm font-medium text-gray-700 mb-2">Causal Chain:</p>
                        <div className="flex items-center text-sm text-gray-600">
                          {hold.causal_chain.map((step, idx) => (
                            <React.Fragment key={idx}>
                              <span className="bg-gray-100 px-2 py-1 rounded">{step}</span>
                              {idx < hold.causal_chain.length - 1 && (
                                <FaArrowRight className="mx-2 text-gray-400" />
                              )}
                            </React.Fragment>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex items-center text-sm text-gray-500">
                      <FaClock className="mr-1" />
                      Created: {new Date(hold.created_at || Date.now()).toLocaleString()}
                      {hold.resolved_at && (
                        <>
                          <span className="mx-2">•</span>
                          <FaCheckCircle className="mr-1" />
                          Resolved: {new Date(hold.resolved_at).toLocaleString()}
                        </>
                      )}
                    </div>
                  </div>
                  
                  <div className="ml-4 flex-shrink-0">
                    {hold.risk_score && (
                      <div className="text-center">
                        <div className="text-2xl font-bold text-gray-800">
                          {(hold.risk_score * 100).toFixed(0)}%
                        </div>
                        <div className="text-sm text-gray-600">Risk Score</div>
                      </div>
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

export default RiskBasedHoldsPanel;