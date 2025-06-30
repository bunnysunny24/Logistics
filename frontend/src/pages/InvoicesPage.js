import React, { useState, useEffect } from 'react';
import { FaFileInvoiceDollar, FaCalendarAlt, FaDollarSign, FaBuilding, FaEye } from 'react-icons/fa';

function InvoicesPage() {
  const [invoices, setInvoices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchInvoices();
  }, []);

  const fetchInvoices = async () => {
    try {
      const response = await fetch('/api/invoices');
      if (response.ok) {
        const data = await response.json();
        setInvoices(data.invoices || []);
      } else {
        setError('Failed to fetch invoices');
      }
    } catch (err) {
      setError('Error loading invoices: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const formatAmount = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-2 text-gray-600">Loading invoices...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="text-red-500 mr-2">⚠️</div>
          <span className="text-red-700">{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <FaFileInvoiceDollar className="text-green-500 text-2xl mr-3" />
              <div>
                <h1 className="text-2xl font-semibold text-gray-900">Invoices</h1>
                <p className="text-sm text-gray-600">
                  {invoices.length} invoice{invoices.length !== 1 ? 's' : ''} found
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6">
          {invoices.length === 0 ? (
            <div className="text-center py-12">
              <FaFileInvoiceDollar className="mx-auto text-gray-300 text-6xl mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No invoices found</h3>
              <p className="text-gray-500">
                Upload invoice documents to see them listed here.
              </p>
            </div>
          ) : (
            <div className="grid gap-4">
              {invoices.map((invoice, index) => (
                <div key={index} className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center mb-2">
                        <h3 className="text-lg font-medium text-gray-900 mr-4">
                          {invoice.invoice_number || `Invoice ${index + 1}`}
                        </h3>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          invoice.status === 'paid' ? 'bg-green-100 text-green-800' :
                          invoice.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {invoice.status || 'Unknown'}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm text-gray-600">
                        <div className="flex items-center">
                          <FaBuilding className="mr-2 text-gray-400" />
                          <span>{invoice.vendor || 'Unknown Vendor'}</span>
                        </div>
                        <div className="flex items-center">
                          <FaDollarSign className="mr-2 text-gray-400" />
                          <span>{formatAmount(invoice.amount || 0)}</span>
                        </div>
                        <div className="flex items-center">
                          <FaCalendarAlt className="mr-2 text-gray-400" />
                          <span>{invoice.date ? formatDate(invoice.date) : 'No date'}</span>
                        </div>
                      </div>
                      
                      {invoice.description && (
                        <p className="mt-2 text-sm text-gray-600 line-clamp-2">
                          {invoice.description}
                        </p>
                      )}
                    </div>
                    
                    <div className="ml-4 flex items-center space-x-2">
                      <button className="p-2 text-gray-400 hover:text-blue-500 transition-colors">
                        <FaEye />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default InvoicesPage;
