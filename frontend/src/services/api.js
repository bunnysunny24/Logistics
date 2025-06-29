/**
 * API Service for Logistics Pulse Copilot
 * Centralizes all backend API calls
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Helper method for making requests
  async makeRequest(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API Error for ${endpoint}:`, error);
      throw error;
    }
  }

  // Health check
  async checkHealth() {
    return this.makeRequest('/health');
  }

  // System status
  async getSystemStatus() {
    return this.makeRequest('/api/status');
  }

  // Query endpoints
  async submitQuery(message, context = null) {
    return this.makeRequest('/api/query', {
      method: 'POST',
      body: JSON.stringify({ message, context }),
    });
  }

  async submitLegacyQuery(message, context = null) {
    return this.makeRequest('/query', {
      method: 'POST',
      body: JSON.stringify({ message, context }),
    });
  }

  // Anomaly detection
  async getAnomalies(filters = {}) {
    const params = new URLSearchParams();
    
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== '') {
        params.append(key, value);
      }
    });
    
    const queryString = params.toString();
    const endpoint = queryString ? `/api/anomalies?${queryString}` : '/api/anomalies';
    
    return this.makeRequest(endpoint);
  }

  async getLegacyAnomalies() {
    return this.makeRequest('/anomalies');
  }

  async triggerAnomalyDetection() {
    return this.makeRequest('/api/detect-anomalies', {
      method: 'POST',
    });
  }

  // Risk-based holds
  async getRiskBasedHolds(filters = {}) {
    const params = new URLSearchParams();
    
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== '') {
        params.append(key, value);
      }
    });
    
    const queryString = params.toString();
    const endpoint = queryString ? `/api/risk-holds?${queryString}` : '/api/risk-holds';
    
    return this.makeRequest(endpoint);
  }

  // Document management
  async uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    return this.makeRequest('/api/upload', {
      method: 'POST',
      headers: {}, // Remove Content-Type to let browser set it with boundary
      body: formData,
    });
  }

  async uploadLegacyDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    return this.makeRequest('/upload', {
      method: 'POST',
      headers: {}, // Remove Content-Type to let browser set it with boundary
      body: formData,
    });
  }

  async getIndexedDocuments() {
    return this.makeRequest('/api/indexed-documents');
  }

  // Statistics
  async getStats() {
    return this.makeRequest('/stats');
  }

  // Feedback
  async submitFeedback(query, answer, rating, feedbackText = null) {
    return this.makeRequest('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({
        query,
        answer,
        rating,
        feedback_text: feedbackText,
      }),
    });
  }

  // Memory management
  async clearMemory() {
    return this.makeRequest('/api/memory', {
      method: 'DELETE',
    });
  }

  // Root endpoint
  async getRootInfo() {
    return this.makeRequest('/');
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService;

// Export individual methods for convenience
export const {
  checkHealth,
  getSystemStatus,
  submitQuery,
  submitLegacyQuery,
  getAnomalies,
  getLegacyAnomalies,
  triggerAnomalyDetection,
  getRiskBasedHolds,
  uploadDocument,
  uploadLegacyDocument,
  getIndexedDocuments,
  getStats,
  submitFeedback,
  clearMemory,
  getRootInfo,
} = apiService;
