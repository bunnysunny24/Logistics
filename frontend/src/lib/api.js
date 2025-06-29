import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const queryLogisticsCopilot = async (query, context = {}) => {
  try {
    const response = await api.post('/api/query', {
      message: query,
      context: context,
    });
    return response.data;
  } catch (error) {
    console.error('Error querying Logistics Copilot:', error);
    throw error;
  }
};

export const uploadDocument = async (file, docType, metadata = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await api.post('/api/upload', formData, {
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

export const getAnomalies = async (params = {}) => {
  try {
    const response = await api.get('/api/anomalies', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching anomalies:', error);
    throw error;
  }
};

export const getRiskBasedHolds = async (params = {}) => {
  try {
    const response = await api.get('/api/risk-holds', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching risk-based holds:', error);
    throw error;
  }
};

export const getSystemStatus = async () => {
  try {
    const response = await api.get('/api/status');
    return response.data;
  } catch (error) {
    console.error('Error fetching system status:', error);
    throw error;
  }
};

export default api;