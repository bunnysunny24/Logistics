"use client";

import { useState, useEffect } from 'react';
import { FiMapPin, FiTruck, FiNavigation } from 'react-icons/fi';

export default function ShipmentMap() {
  const [shipments, setShipments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeShipment, setActiveShipment] = useState(null);

  useEffect(() => {
    // Simulate API call for shipment data
    const fetchShipments = async () => {
      try {
        // This would normally be an API call
        // const response = await api.get('/api/shipments');
        
        // Mock data for now
        setTimeout(() => {
          setShipments([
            {
              id: 'SH001',
              origin: 'New York, NY',
              destination: 'Los Angeles, CA',
              status: 'In Transit',
              progress: 65,
              estimatedDelivery: '2025-07-02',
              trackingNumber: 'TR-2025-001'
            },
            {
              id: 'SH002',
              origin: 'Chicago, IL',
              destination: 'Houston, TX',
              status: 'Delivered',
              progress: 100,
              estimatedDelivery: '2025-06-28',
              trackingNumber: 'TR-2025-002'
            },
            {
              id: 'SH003',
              origin: 'Seattle, WA',
              destination: 'Miami, FL',
              status: 'Pending',
              progress: 5,
              estimatedDelivery: '2025-07-05',
              trackingNumber: 'TR-2025-003'
            }
          ]);
          setLoading(false);
        }, 1000);
      } catch (error) {
        console.error('Error fetching shipments:', error);
        setLoading(false);
      }
    };

    fetchShipments();
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'Delivered':
        return 'text-green-600 bg-green-100';
      case 'In Transit':
        return 'text-blue-600 bg-blue-100';
      case 'Pending':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <FiMapPin className="mr-2" />
        Shipment Tracking
      </h2>
      
      {/* Map placeholder - in a real app, this would be an interactive map */}
      <div className="relative h-64 bg-gray-100 rounded-lg mb-6 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <FiNavigation className="text-4xl mx-auto mb-2" />
          <p>Interactive map would be displayed here</p>
          <p className="text-sm">Showing {shipments.length} active shipments</p>
        </div>
      </div>
      
      {/* Shipment list */}
      <div className="space-y-4">
        <h3 className="font-medium text-gray-900">Active Shipments</h3>
        {shipments.map((shipment) => (
          <div 
            key={shipment.id}
            className={`p-4 border rounded-lg cursor-pointer transition-colors ${
              activeShipment === shipment.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => setActiveShipment(activeShipment === shipment.id ? null : shipment.id)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <FiTruck className="text-gray-500" />
                <div>
                  <p className="font-medium">{shipment.trackingNumber}</p>
                  <p className="text-sm text-gray-600">{shipment.origin} → {shipment.destination}</p>
                </div>
              </div>
              <div className="text-right">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(shipment.status)}`}>
                  {shipment.status}
                </span>
                <p className="text-sm text-gray-600 mt-1">ETA: {shipment.estimatedDelivery}</p>
              </div>
            </div>
            
            {/* Progress bar */}
            <div className="mt-3">
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>Progress</span>
                <span>{shipment.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${shipment.progress}%` }}
                ></div>
              </div>
            </div>
            
            {activeShipment === shipment.id && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600">Shipment ID</p>
                    <p className="font-medium">{shipment.id}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Estimated Delivery</p>
                    <p className="font-medium">{shipment.estimatedDelivery}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
