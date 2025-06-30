import React, { useState, useEffect } from 'react';
import { FaBell, FaSearch, FaUser, FaCog, FaChevronDown } from 'react-icons/fa';
import { getSystemStatus } from '../lib/api';

function Header({ onToggleSidebar, sidebarOpen, systemStatus }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showNotifications, setShowNotifications] = useState(false);
  
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
    const interval = setInterval(fetchStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <header className="bg-white/80 backdrop-blur-sm shadow-lg border-b border-white/20 px-6 py-4 sticky top-0 z-40">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-xl hover:bg-gray-100 transition-colors duration-200 lg:hidden"
          >
            <div className="w-6 h-6 flex flex-col justify-center space-y-1">
              <div className="w-full h-0.5 bg-gray-600 rounded transition-all duration-300"></div>
              <div className="w-full h-0.5 bg-gray-600 rounded transition-all duration-300"></div>
              <div className="w-full h-0.5 bg-gray-600 rounded transition-all duration-300"></div>
            </div>
          </button>
          
          <div className="hidden md:flex items-center space-x-4">
            <div className="relative">
              <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 text-sm" />
              <input
                type="text"
                placeholder="Search documents, anomalies..."
                className="pl-10 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-300 w-80"
              />
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Status indicators */}
          <div className="hidden lg:flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className="text-sm text-gray-600">Documents:</div>
              <div className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-semibold">
                {status?.documents_processed?.total || 0}
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className="text-sm text-gray-600">Status:</div>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemStatus?.status === 'operational' ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                }`} />
                <span className={`text-sm font-semibold ${
                  systemStatus?.status === 'operational' ? 'text-green-600' : 'text-red-600'
                }`}>
                  {systemStatus?.status === 'operational' ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>
          </div>
          
          {/* Notifications */}
          <div className="relative">
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative p-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl hover:from-blue-100 hover:to-indigo-100 transition-all duration-300 group"
            >
              <FaBell className="text-gray-600 group-hover:text-blue-600 transition-colors duration-300" />
              <div className="absolute -top-1 -right-1 w-5 h-5 bg-gradient-to-r from-red-500 to-red-600 rounded-full flex items-center justify-center">
                <span className="text-xs text-white font-bold">3</span>
              </div>
            </button>
            
            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 bg-white rounded-2xl shadow-2xl border border-gray-100 z-50">
                <div className="p-4 border-b border-gray-100">
                  <h3 className="font-semibold text-gray-900">Notifications</h3>
                </div>
                <div className="p-2 max-h-64 overflow-y-auto">
                  <div className="p-3 hover:bg-gray-50 rounded-xl cursor-pointer">
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">High-risk anomaly detected</p>
                        <p className="text-xs text-gray-500">Invoice #INV-2025-001 flagged for review</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* User menu */}
          <div className="flex items-center space-x-3 p-2 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl hover:from-gray-100 hover:to-slate-100 transition-all duration-300 cursor-pointer group">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full flex items-center justify-center">
              <FaUser className="text-white text-sm" />
            </div>
            <div className="hidden md:block">
              <p className="text-sm font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                {process.env.REACT_APP_USER_NAME || 'Admin'}
              </p>
              <p className="text-xs text-gray-500">Logistics Analyst</p>
            </div>
            <FaChevronDown className="text-gray-400 text-xs group-hover:text-blue-600 transition-colors" />
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;