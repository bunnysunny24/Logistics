import React, { useState } from 'react';
import { uploadDocument } from '../lib/api';
import { FaUpload, FaFile, FaFileInvoice, FaTruck, FaBook, FaCheck, FaTimes } from 'react-icons/fa';

function DocumentUploader({ onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [docType, setDocType] = useState('invoice');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('idle'); // 'idle', 'success', 'error'
  const [errorMessage, setErrorMessage] = useState('');
  
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadStatus('idle');
      setErrorMessage('');
    }
  };
  
  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!file) return;
    
    setIsUploading(true);
    setUploadStatus('idle');
    setErrorMessage('');
    
    try {
      await uploadDocument(file, docType);
      setUploadStatus('success');
      setFile(null);
      if (onUploadComplete) {
        onUploadComplete();
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('error');
      setErrorMessage(error.response?.data?.detail || 'Error uploading document');
    } finally {
      setIsUploading(false);
    }
  };
  
  return (
    <div className="card">
      <div className="card-header">
        <h5 className="text-lg font-medium">Upload Document</h5>
      </div>
      <div className="card-body">
        <form onSubmit={handleUpload}>
          <div className="mb-4">
            <label className="form-label">Document Type</label>
            <div className="flex space-x-4">
              <div 
                className={`p-3 border rounded-md cursor-pointer flex items-center ${
                  docType === 'invoice' ? 'border-primary-500 bg-primary-50' : 'border-gray-300'
                }`}
                onClick={() => setDocType('invoice')}
              >
                <input
                  type="radio"
                  name="docType"
                  value="invoice"
                  checked={docType === 'invoice'}
                  onChange={() => setDocType('invoice')}
                  className="sr-only"
                />
                <FaFileInvoice className="mr-2 text-gray-600" />
                <span>Invoice</span>
              </div>
              
              <div 
                className={`p-3 border rounded-md cursor-pointer flex items-center ${
                  docType === 'shipment' ? 'border-primary-500 bg-primary-50' : 'border-gray-300'
                }`}
                onClick={() => setDocType('shipment')}
              >
                <input
                  type="radio"
                  name="docType"
                  value="shipment"
                  checked={docType === 'shipment'}
                  onChange={() => setDocType('shipment')}
                  className="sr-only"
                />
                <FaTruck className="mr-2 text-gray-600" />
                <span>Shipment</span>
              </div>
              
              <div 
                className={`p-3 border rounded-md cursor-pointer flex items-center ${
                  docType === 'policy' ? 'border-primary-500 bg-primary-50' : 'border-gray-300'
                }`}
                onClick={() => setDocType('policy')}
              >
                <input
                  type="radio"
                  name="docType"
                  value="policy"
                  checked={docType === 'policy'}
                  onChange={() => setDocType('policy')}
                  className="sr-only"
                />
                <FaBook className="mr-2 text-gray-600" />
                <span>Policy</span>
              </div>
            </div>
          </div>
          
          <div className="mb-4">
            <label className="form-label">File</label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:bg-gray-50">
              <input
                type="file"
                id="fileUpload"
                className="hidden"
                accept=".pdf,.csv"
                onChange={handleFileChange}
              />
              <label htmlFor="fileUpload" className="cursor-pointer">
                {file ? (
                  <div className="text-center">
                    <FaFile className="mx-auto text-4xl mb-2 text-gray-500" />
                    <p className="mb-1 text-gray-700">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(file.size / 1024).toFixed(2)} KB
                    </p>
                  </div>
                ) : (
                  <div className="text-center">
                    <FaUpload className="mx-auto text-4xl mb-2 text-gray-500" />
                    <p className="mb-1 text-gray-700">
                      <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-sm text-gray-500">
                      PDF or CSV (max. 10MB)
                    </p>
                  </div>
                )}
              </label>
            </div>
          </div>
          
          {uploadStatus === 'success' && (
            <div className="alert-success mb-4">
              <FaCheck className="mr-2" />
              <div>Document uploaded successfully!</div>
            </div>
          )}
          
          {uploadStatus === 'error' && (
            <div className="alert-danger mb-4">
              <FaTimes className="mr-2" />
              <div>{errorMessage || 'Error uploading document'}</div>
            </div>
          )}
          
          <button
            type="submit"
            disabled={!file || isUploading}
            className={`w-full py-2 px-4 rounded-md ${
              !file || isUploading
                ? 'bg-gray-300 cursor-not-allowed'
                : 'btn-primary'
            }`}
          >
            {isUploading ? 'Uploading...' : 'Upload Document'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default DocumentUploader;