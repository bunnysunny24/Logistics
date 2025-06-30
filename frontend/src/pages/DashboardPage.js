import React, { useState, useEffect } from 'react';
import DocumentUploader from '../components/DocumentUploader';
import AnomalyDashboard from '../components/AnomalyDashboard';
import SystemStatusPanel from '../components/SystemStatusPanel';
import ChatInterface from '../components/ChatInterface';
import RiskBasedHoldsPanel from '../components/RiskBasedHoldsPanel';
import ErrorBoundary from '../components/ErrorBoundary';
import apiService from '../services/api';
import { 
  FaFileInvoiceDollar, 
  FaTruck, 
  FaExclamationTriangle, 
  FaChartLine,
  FaUpload,
  FaRobot,
  FaShieldAlt
} from 'react-icons/fa';

function DashboardPage() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [systemStatus, setSystemStatus] = useState({ status: 'operational' });
  const [stats, setStats] = useState({
    totalDocuments: 0,
    totalAnomalies: 0,
    highRiskAnomalies: 0,
    processingRate: 0
  });
  const [loading, setLoading] = useState(true);

  const handleDocumentUploaded = () => {
    // Trigger a refresh of the dashboard components
    setRefreshTrigger(prev => prev + 1);
  };

  const handleRefreshData = () => {
    // Trigger a refresh of all data
    setRefreshTrigger(prev => prev + 1);
  };

  // Load dashboard statistics
  useEffect(() => {
    const loadStats = async () => {
      try {
        setLoading(true);
        const [anomalies, systemStats, dashboardStats] = await Promise.all([
          apiService.getAnomalies(),
          apiService.getStats(),
          apiService.getDashboardStats().catch(() => ({ // Fallback if endpoint doesn't exist
            totalDocuments: 0,
            processingRate: 95.2
          }))
        ]);

        // Calculate real stats from API data
        const totalAnomalies = anomalies.length;
        const highRiskAnomalies = anomalies.filter(a => a.risk_score >= 0.8).length;
        
        setStats({
          totalDocuments: dashboardStats.totalDocuments || systemStats.total_documents || 0,
          totalAnomalies: totalAnomalies,
          highRiskAnomalies: highRiskAnomalies,
          processingRate: dashboardStats.processingRate || systemStats.processing_rate || 95.2
        });
      } catch (error) {
        console.error('Error loading dashboard stats:', error);
      } finally {
        setLoading(false);
      }
    };

    loadStats();
  }, [refreshTrigger]);

  const overviewCards = [
    {
      title: 'Total Documents',
      value: loading ? '...' : stats.totalDocuments,
      icon: FaFileInvoiceDollar,
      color: 'bg-blue-500',
      change: '+12%',
      changeType: 'positive'
    },
    {
      title: 'Active Anomalies',
      value: loading ? '...' : stats.totalAnomalies,
      icon: FaExclamationTriangle,
      color: 'bg-red-500',
      change: stats.totalAnomalies > 20 ? '+8%' : '-8%',
      changeType: stats.totalAnomalies > 20 ? 'positive' : 'negative'
    },
    {
      title: 'High Risk Items',
      value: loading ? '...' : stats.highRiskAnomalies,
      icon: FaShieldAlt,
      color: 'bg-orange-500',
      change: stats.highRiskAnomalies > 5 ? '+3%' : '-3%',
      changeType: stats.highRiskAnomalies > 5 ? 'positive' : 'negative'
    },
    {
      title: 'Processing Rate',
      value: loading ? '...' : `${stats.processingRate}%`,
      icon: FaChartLine,
      color: 'bg-green-500',
      change: '+2.1%',
      changeType: 'positive'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-700 rounded-3xl p-8 text-white mb-8 shadow-2xl relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-16 translate-x-16 animate-pulse" />
        <div className="absolute bottom-0 left-0 w-24 h-24 bg-white/5 rounded-full translate-y-12 -translate-x-12" />
        
        <div className="relative z-10 flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
              Welcome to Logistics Pulse Copilot
            </h2>
            <p className="text-blue-100 text-lg font-medium">
              AI-powered logistics and finance document processing system
            </p>
            <div className="mt-4 flex items-center space-x-4">
              <div className="bg-white/20 px-4 py-2 rounded-full backdrop-blur-sm">
                <span className="text-sm font-semibold">🚀 Enhanced AI Analytics</span>
              </div>
              <div className="bg-white/20 px-4 py-2 rounded-full backdrop-blur-sm">
                <span className="text-sm font-semibold">⚡ Real-time Processing</span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-6">
            <div className="text-right">
              <div className="text-sm text-blue-100 mb-1">System Status</div>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  systemStatus?.status === 'operational' ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                }`} />
                <span className="font-bold text-lg">
                  {systemStatus?.status === 'operational' ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>
            <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center backdrop-blur-sm">
              <FaRobot className="text-2xl text-white" />
            </div>
          </div>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {overviewCards.map((card, index) => (
          <div 
            key={index} 
            className="group relative bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/20 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-300 overflow-hidden"
          >
            {/* Animated background gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-transparent via-transparent to-blue-50/30 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            
            <div className="relative z-10 flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm text-gray-600 font-semibold mb-2 tracking-wide uppercase">
                  {card.title}
                </p>
                <div className="text-3xl font-bold text-gray-900 mb-3">
                  {loading ? (
                    <div className="h-9 w-20 bg-gradient-to-r from-gray-200 to-gray-300 animate-pulse rounded-lg"></div>
                  ) : (
                    <span className="bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                      {card.value}
                    </span>
                  )}
                </div>
                <div className="flex items-center">
                  <span className={`text-sm font-bold px-2 py-1 rounded-full ${
                    card.changeType === 'positive' 
                      ? 'text-green-700 bg-green-100' 
                      : 'text-red-700 bg-red-100'
                  }`}>
                    {card.change}
                  </span>
                  <span className="text-sm text-gray-500 ml-2">vs last week</span>
                </div>
              </div>
              <div className={`relative p-4 rounded-2xl ${card.color} shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                {React.createElement(card.icon, { className: "h-7 w-7 text-white" })}
                <div className="absolute inset-0 rounded-2xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </div>
            </div>
            
            {/* Decorative elements */}
            <div className="absolute top-2 right-2 w-20 h-20 bg-gradient-to-br from-blue-100/20 to-indigo-100/20 rounded-full transform translate-x-10 -translate-y-10 group-hover:scale-150 transition-transform duration-500" />
          </div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        {/* Document Upload Section */}
        <div className="lg:col-span-2">
          <ErrorBoundary>
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition-all duration-300 overflow-hidden">
              <div className="p-6 border-b border-gray-100 bg-gradient-to-r from-blue-50 to-indigo-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg">
                      <FaUpload className="h-6 w-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-xl font-bold text-gray-900">Document Processing</h3>
                      <p className="text-sm text-gray-600 mt-1">
                        Upload invoices, shipment documents, and other logistics files
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    <span className="text-sm text-green-600 font-semibold">AI Ready</span>
                  </div>
                </div>
              </div>
              <div className="p-6">
                <DocumentUploader onUploadComplete={handleDocumentUploaded} />
              </div>
            </div>
          </ErrorBoundary>
        </div>

        {/* System Status */}
        <div>
          <ErrorBoundary>
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition-all duration-300">
              <SystemStatusPanel />
            </div>
          </ErrorBoundary>
        </div>
      </div>

      {/* Risk-Based Holds and Anomalies */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        <div className="lg:col-span-2">
          <ErrorBoundary>
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition-all duration-300 overflow-hidden">
              <div className="p-6 border-b border-gray-100 bg-gradient-to-r from-red-50 to-orange-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="p-3 bg-gradient-to-r from-red-500 to-orange-600 rounded-xl shadow-lg">
                      <FaExclamationTriangle className="h-6 w-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-xl font-bold text-gray-900">Anomaly Detection</h3>
                      <p className="text-sm text-gray-600 mt-1">
                        AI-powered risk detection and alerts
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" />
                      <span className="text-sm text-orange-600 font-semibold">Monitoring</span>
                    </div>
                    <button
                      onClick={handleRefreshData}
                      className="px-4 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 transition-all duration-300 font-semibold shadow-lg hover:shadow-xl transform hover:scale-105"
                    >
                      Refresh
                    </button>
                  </div>
                </div>
              </div>
              <div className="p-6">
                <AnomalyDashboard refreshTrigger={refreshTrigger} />
              </div>
            </div>
          </ErrorBoundary>
        </div>

        <div>
          <ErrorBoundary>
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition-all duration-300">
              <RiskBasedHoldsPanel />
            </div>
          </ErrorBoundary>
        </div>
      </div>

      {/* AI Chat Interface */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition-all duration-300 overflow-hidden">
        <div className="p-6 border-b border-gray-100 bg-gradient-to-r from-green-50 to-emerald-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl shadow-lg">
                <FaRobot className="h-6 w-6 text-white" />
              </div>
              <div className="ml-4">
                <h3 className="text-xl font-bold text-gray-900">AI Assistant</h3>
                <p className="text-sm text-gray-600 mt-1">
                  Ask questions about your logistics data and get AI-powered insights
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm text-green-600 font-semibold">AI Online</span>
              </div>
              <div className="px-3 py-1 bg-gradient-to-r from-green-100 to-emerald-100 rounded-full">
                <span className="text-xs font-bold text-green-700">GPT-4 Enhanced</span>
              </div>
            </div>
          </div>
        </div>
        <div className="p-6">
          <ErrorBoundary>
            <ChatInterface />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;