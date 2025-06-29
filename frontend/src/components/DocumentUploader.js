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
    <div className="card mb-4">
      <div className="card-header">
        <h5 className="card-title mb-0">Upload Document</h5>
      </div>
      <div className="card-body">
        <form onSubmit={handleUpload}>
          <div className="mb-4">
            <label className="form-label">Document Type</label>
            <div className="d-flex gap-3">
              <div className={`card p-3 cursor-pointer ${docType === 'invoice' ? 'border-primary bg-light' : 'border'}`} 
                   onClick={() => setDocType('invoice')}
                   style={{ cursor: 'pointer' }}>
                <div className="d-flex align-items-center">
                  <input
                    type="radio"
                    name="docType"
                    value="invoice"
                    checked={docType === 'invoice'}
                    onChange={() => setDocType('invoice')}
                    className="d-none"
                  />
                  <FaFileInvoice className="me-2 text-muted" />
                  <span>Invoice</span>
                </div>
              </div>
              
              <div className={`card p-3 cursor-pointer ${docType === 'shipment' ? 'border-primary bg-light' : 'border'}`}
                   onClick={() => setDocType('shipment')}
                   style={{ cursor: 'pointer' }}>
                <div className="d-flex align-items-center">
                  <input
                    type="radio"
                    name="docType"
                    value="shipment"
                    checked={docType === 'shipment'}
                    onChange={() => setDocType('shipment')}
                    className="d-none"
                  />
                  <FaTruck className="me-2 text-muted" />
                  <span>Shipment</span>
                </div>
              </div>
              
              <div className={`card p-3 cursor-pointer ${docType === 'policy' ? 'border-primary bg-light' : 'border'}`}
                   onClick={() => setDocType('policy')}
                   style={{ cursor: 'pointer' }}>
                <div className="d-flex align-items-center">
                  <input
                    type="radio"
                    name="docType"
                    value="policy"
                    checked={docType === 'policy'}
                    onChange={() => setDocType('policy')}
                    className="d-none"
                  />
                  <FaBook className="me-2 text-muted" />
                  <span>Policy</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mb-4">
            <label className="form-label">File</label>
            <div className="border border-2 border-dashed rounded p-4 text-center" style={{ cursor: 'pointer' }}>
              <input
                type="file"
                id="fileUpload"
                className="d-none"
                accept=".pdf,.csv"
                onChange={handleFileChange}
              />
              <label htmlFor="fileUpload" className="mb-0 d-block">
                {file ? (
                  <div className="text-center">
                    <FaFile className="mb-2" style={{ fontSize: '2rem' }} />
                    <p className="mb-1">{file.name}</p>
                    <p className="text-muted small">
                      {(file.size / 1024).toFixed(2)} KB
                    </p>
                  </div>
                ) : (
                  <div className="text-center">
                    <FaUpload className="mb-2" style={{ fontSize: '2rem' }} />
                    <p className="mb-1">
                      <span className="fw-bold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-muted small">
                      PDF or CSV (max. 10MB)
                    </p>
                  </div>
                )}
              </label>
            </div>
          </div>
          
          {uploadStatus === 'success' && (
            <div className="alert alert-success d-flex align-items-center" role="alert">
              <FaCheck className="me-2" />
              <div>Document uploaded successfully!</div>
            </div>
          )}
          
          {uploadStatus === 'error' && (
            <div className="alert alert-danger d-flex align-items-center" role="alert">
              <FaTimes className="me-2" />
              <div>{errorMessage || 'Error uploading document'}</div>
            </div>
          )}
          
          <button
            type="submit"
            disabled={!file || isUploading}
            className="btn btn-primary w-100"
          >
            {isUploading ? 'Uploading...' : 'Upload Document'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default DocumentUploader;