// src/lib/api.ts
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const queryLogisticsCopilot = async (query: string, context?: unknown) => {
  try {
    const response = await api.post('/query', {
      message: query,
      context,
    });
    return response.data;
  } catch (error) {
    console.error('Error querying Logistics Copilot:', error);
    throw error;
  }
};

export const uploadDocument = async (file: File, docType: string, metadata?: unknown) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('type', docType);
  
  if (metadata) {
    formData.append('metadata', JSON.stringify(metadata));
  }
  
  try {
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading document:', error);
    throw error;
  }
};

export const getAnomalies = async (params?: { 
  startDate?: string; 
  endDate?: string;
  minRiskScore?: number;
}) => {
  try {
    const response = await api.get('/anomalies', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching anomalies:', error);
    throw error;
  }
};

export const getSystemStatus = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Error fetching system status:', error);
    throw error;
  }
};

export default api;