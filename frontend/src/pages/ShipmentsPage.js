import React, { useState, useEffect } from 'react';
import { FaTruck, FaMapMarkerAlt, FaCalendarAlt, FaBox, FaEye, FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa';

function ShipmentsPage() {
  const [shipments, setShipments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchShipments();
  }, []);

  const fetchShipments = async () => {
    try {
      const response = await fetch('/api/shipments');
      if (response.ok) {
        const data = await response.json();
        setShipments(data.shipments || []);
      } else {
        setError('Failed to fetch shipments');
      }
    } catch (err) {
      setError('Error loading shipments: ' + err.message);
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

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'delivered':
        return <FaCheckCircle className="text-green-500" />;
      case 'in_transit':
        return <FaTruck className="text-blue-500 animate-pulse" />;
      case 'delayed':
        return <FaExclamationTriangle className="text-yellow-500" />;
      case 'pending':
        return <FaBox className="text-gray-500" />;
      default:
        return <FaBox className="text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'delivered':
        return 'bg-green-100 text-green-800';
      case 'in_transit':
        return 'bg-blue-100 text-blue-800';
      case 'delayed':
        return 'bg-yellow-100 text-yellow-800';
      case 'pending':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-2 text-gray-600">Loading shipments...</span>
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
              <FaTruck className="text-purple-500 text-2xl mr-3" />
              <div>
                <h1 className="text-2xl font-semibold text-gray-900">Shipments</h1>
                <p className="text-sm text-gray-600">
                  {shipments.length} shipment{shipments.length !== 1 ? 's' : ''} found
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6">
          {shipments.length === 0 ? (
            <div className="text-center py-12">
              <FaTruck className="mx-auto text-gray-300 text-6xl mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No shipments found</h3>
              <p className="text-gray-500">
                Upload shipment documents to see them listed here.
              </p>
            </div>
          ) : (
            <div className="grid gap-4">
              {shipments.map((shipment, index) => (
                <div key={index} className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center mb-2">
                        <h3 className="text-lg font-medium text-gray-900 mr-4">
                          {shipment.tracking_number || shipment.shipment_id || `Shipment ${index + 1}`}
                        </h3>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(shipment.status)}`}>
                          {getStatusIcon(shipment.status)}
                          <span className="ml-1">{shipment.status || 'Unknown'}</span>
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm text-gray-600">
                        <div className="flex items-center">
                          <FaMapMarkerAlt className="mr-2 text-gray-400" />
                          <span>
                            {shipment.origin && shipment.destination 
                              ? `${shipment.origin} → ${shipment.destination}`
                              : shipment.route || 'Route not specified'
                            }
                          </span>
                        </div>
                        <div className="flex items-center">
                          <FaCalendarAlt className="mr-2 text-gray-400" />
                          <span>
                            {shipment.ship_date || shipment.expected_delivery 
                              ? formatDate(shipment.ship_date || shipment.expected_delivery)
                              : 'No date'
                            }
                          </span>
                        </div>
                        <div className="flex items-center">
                          <FaBox className="mr-2 text-gray-400" />
                          <span>{shipment.carrier || shipment.shipping_method || 'Unknown carrier'}</span>
                        </div>
                      </div>
                      
                      {shipment.description && (
                        <p className="mt-2 text-sm text-gray-600 line-clamp-2">
                          {shipment.description}
                        </p>
                      )}
                      
                      {shipment.items && shipment.items.length > 0 && (
                        <div className="mt-2">
                          <span className="text-xs text-gray-500">
                            Items: {shipment.items.map(item => item.name || item.description).join(', ')}
                          </span>
                        </div>
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

export default ShipmentsPage;
