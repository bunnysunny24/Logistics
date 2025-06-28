// src/components/Sidebar.tsx
"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FiHome, FiFileText, FiTruck, FiAlertTriangle, FiSettings } from 'react-icons/fi';

export default function Sidebar() {
  const pathname = usePathname();
  
  const navigation = [
    { name: 'Dashboard', href: '/', icon: FiHome },
    { name: 'Invoices', href: '/invoices', icon: FiFileText },
    { name: 'Shipments', href: '/shipments', icon: FiTruck },
    { name: 'Anomalies', href: '/anomalies', icon: FiAlertTriangle },
    { name: 'Settings', href: '/settings', icon: FiSettings },
  ];
  
  return (
    <div className="hidden md:flex md:flex-shrink-0">
      <div className="flex flex-col w-64">
        <div className="flex flex-col h-0 flex-1 bg-gray-800">
          <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
            <div className="flex items-center flex-shrink-0 px-4">
              <span className="text-white text-xl font-bold">Logistics Pulse</span>
            </div>
            <nav className="mt-5 flex-1 px-2 space-y-1">
              {navigation.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                      isActive
                        ? 'bg-gray-900 text-white'
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                    }`}
                  >
                    <item.icon
                      className={`mr-3 flex-shrink-0 h-6 w-6 ${
                        isActive ? 'text-white' : 'text-gray-400 group-hover:text-gray-300'
                      }`}
                      aria-hidden="true"
                    />
                    {item.name}
                  </Link>
                );
              })}
            </nav>
          </div>
          <div className="flex-shrink-0 flex bg-gray-700 p-4">
            <div className="flex-shrink-0 w-full group block">
              <div className="flex items-center">
                <div>
                  <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-700">
                    {/* User initial or avatar would go here */}
                    U
                  </div>
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-white">{process.env.NEXT_PUBLIC_USER_NAME || 'User'}</p>
                  <p className="text-xs font-medium text-gray-300 group-hover:text-gray-200">Logistics Analyst</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}