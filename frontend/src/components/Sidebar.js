import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  FaHome, 
  FaFileInvoiceDollar, 
  FaTruck, 
  FaCog, 
  FaUser
} from 'react-icons/fa';

function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  
  const navigation = [
    { name: 'Dashboard', icon: FaHome, path: '/', color: 'text-blue-400' },
    { name: 'Invoices', icon: FaFileInvoiceDollar, path: '/invoices', color: 'text-green-400' },
    { name: 'Shipments', icon: FaTruck, path: '/shipments', color: 'text-purple-400' },
    { name: 'Settings', icon: FaCog, path: '/settings', color: 'text-gray-400' },
  ];

  const handleNavigation = (path) => {
    navigate(path);
  };
  
  return (
    <div className="fixed w-64 h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white shadow-2xl">
      <div className="flex flex-col h-full">
        <div className="p-6 border-b border-slate-700/50">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
              <FaTruck className="text-white text-lg" />
            </div>
            <div>
              <h2 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
                Logistics Pulse
              </h2>
              <p className="text-xs text-slate-400">AI Copilot</p>
            </div>
          </div>
        </div>
        
        <div className="flex-1 p-4">
          <nav className="space-y-2">
            {navigation.map((item, index) => {
              const isActive = location.pathname === item.path;
              return (
                <div key={item.name} className="relative group">
                  <button 
                    onClick={() => handleNavigation(item.path)}
                    className={`w-full flex items-center px-4 py-3 rounded-xl transition-all duration-300 transform hover:scale-105 ${
                      isActive 
                        ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/25' 
                        : 'text-slate-300 hover:bg-slate-700/50 hover:text-white hover:shadow-lg'
                    }`}
                  >
                    <item.icon className={`mr-3 h-5 w-5 transition-colors duration-300 ${
                      isActive ? 'text-white' : item.color
                    }`} />
                    <span className="font-medium">{item.name}</span>
                    {isActive && (
                      <div className="absolute right-2 w-2 h-2 bg-white rounded-full animate-pulse" />
                    )}
                  </button>
                  {/* Hover glow effect */}
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-600/20 to-indigo-600/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10" />
                </div>
              );
            })}
          </nav>
        </div>
        
        <div className="p-4 border-t border-slate-700/50">
          <div className="flex items-center p-3 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 transition-all duration-300 cursor-pointer group">
            <div className="relative">
              <div className="flex items-center justify-center bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full h-12 w-12 shadow-lg">
                <FaUser className="text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-slate-800 animate-pulse" />
            </div>
            <div className="ml-3 flex-1">
              <p className="text-sm font-semibold text-white group-hover:text-blue-400 transition-colors">
                {process.env.REACT_APP_USER_NAME || 'Admin User'}
              </p>
              <p className="text-xs text-slate-400 group-hover:text-slate-300 transition-colors">
                Logistics Analyst
              </p>
            </div>
            <div className="w-2 h-2 bg-blue-400 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;