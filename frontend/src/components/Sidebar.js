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
    <div className="sidebar bg-dark text-white" style={{ width: '250px', position: 'fixed', height: '100vh', left: 0, top: 0 }}>
      <div className="d-flex flex-column h-100">
        <div className="p-3 border-bottom border-secondary">
          <h2 className="h5 mb-0">Logistics Pulse</h2>
        </div>
        
        <div className="p-3">
          <ul className="nav flex-column">
            {navigation.map((item) => (
              <li key={item.name} className="nav-item mb-2">
                <a 
                  href="#" 
                  className={`nav-link d-flex align-items-center ${
                    item.active ? 'text-white bg-secondary rounded' : 'text-light'
                  }`}
                  onClick={(e) => e.preventDefault()}
                >
                  <item.icon className="me-2" />
                  {item.name}
                </a>
              </li>
            ))}
          </ul>
        </div>
        
        <div className="mt-auto p-3 border-top border-secondary">
          <div className="d-flex align-items-center">
            <div className="d-flex align-items-center justify-content-center bg-secondary rounded-circle" style={{ width: '40px', height: '40px' }}>
              <FaUser />
            </div>
            <div className="ms-3">
              <p className="mb-0 small fw-medium">{process.env.REACT_APP_USER_NAME || 'User'}</p>
              <p className="mb-0 small text-light opacity-75">Logistics Analyst</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;