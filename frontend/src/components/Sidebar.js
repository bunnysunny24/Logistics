import React from 'react';
import { 
  FaHome, 
  FaFileInvoiceDollar, 
  FaTruck, 
  FaExclamationTriangle, 
  FaCog, 
  FaUser 
} from 'react-icons/fa';

function Sidebar() {
  const navigation = [
    { name: 'Dashboard', icon: FaHome, active: true },
    { name: 'Invoices', icon: FaFileInvoiceDollar, active: false },
    { name: 'Shipments', icon: FaTruck, active: false },
    { name: 'Anomalies', icon: FaExclamationTriangle, active: false },
    { name: 'Settings', icon: FaCog, active: false },
  ];
  
  return (
    <div className="fixed w-64 h-screen bg-gray-800 text-white">
      <div className="flex flex-col h-full">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-xl font-semibold">Logistics Pulse</h2>
        </div>
        
        <div className="flex-1 p-4">
          <ul className="space-y-2">
            {navigation.map((item) => (
              <li key={item.name}>
                <a 
                  href="#" 
                  className={`flex items-center px-3 py-2 rounded-md ${
                    item.active ? 'bg-gray-700 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`}
                  onClick={(e) => e.preventDefault()}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                </a>
              </li>
            ))}
          </ul>
        </div>
        
        <div className="p-4 border-t border-gray-700">
          <div className="flex items-center">
            <div className="flex items-center justify-center bg-gray-600 rounded-full h-10 w-10">
              <FaUser />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium">{process.env.REACT_APP_USER_NAME || 'User'}</p>
              <p className="text-xs text-gray-400">Logistics Analyst</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;