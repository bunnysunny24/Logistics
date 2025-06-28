"use client";

import { useState, useEffect } from 'react';
import { FiDollarSign, FiFileText, FiTrendingUp, FiCalendar } from 'react-icons/fi';

export default function InvoiceStats() {
  const [stats, setStats] = useState({
    totalInvoices: 0,
    totalAmount: 0,
    pendingInvoices: 0,
    averageProcessingTime: 0,
    loading: true
  });

  useEffect(() => {
    // Simulate API call for stats
    const fetchStats = async () => {
      try {
        // This would normally be an API call
        // const response = await api.get('/api/invoice-stats');
        
        // Mock data for now
        setTimeout(() => {
          setStats({
            totalInvoices: 1247,
            totalAmount: 892345.67,
            pendingInvoices: 23,
            averageProcessingTime: 2.3,
            loading: false
          });
        }, 1000);
      } catch (error) {
        console.error('Error fetching invoice stats:', error);
        setStats(prev => ({ ...prev, loading: false }));
      }
    };

    fetchStats();
  }, []);

  if (stats.loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            <div className="h-8 bg-gray-200 rounded"></div>
            <div className="h-8 bg-gray-200 rounded"></div>
            <div className="h-8 bg-gray-200 rounded"></div>
            <div className="h-8 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <FiFileText className="mr-2" />
        Invoice Statistics
      </h2>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-blue-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Invoices</p>
              <p className="text-2xl font-bold text-blue-600">{stats.totalInvoices.toLocaleString()}</p>
            </div>
            <FiFileText className="text-blue-500 text-2xl" />
          </div>
        </div>
        
        <div className="p-4 bg-green-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Amount</p>
              <p className="text-2xl font-bold text-green-600">${stats.totalAmount.toLocaleString()}</p>
            </div>
            <FiDollarSign className="text-green-500 text-2xl" />
          </div>
        </div>
        
        <div className="p-4 bg-yellow-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.pendingInvoices}</p>
            </div>
            <FiCalendar className="text-yellow-500 text-2xl" />
          </div>
        </div>
        
        <div className="p-4 bg-purple-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Avg Processing (days)</p>
              <p className="text-2xl font-bold text-purple-600">{stats.averageProcessingTime}</p>
            </div>
            <FiTrendingUp className="text-purple-500 text-2xl" />
          </div>
        </div>
      </div>
    </div>
  );
}
