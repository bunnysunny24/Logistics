// src/components/DocumentUploader.tsx
"use client";

import { useState } from 'react';
import { uploadDocument } from '@/lib/api';
import { FiUpload, FiFile, FiFileText, FiTruck, FiBook, FiCheck, FiX } from 'react-icons/fi';

type Props = {
  onUploadComplete?: () => void;
};

export default function DocumentUploader({ onUploadComplete }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [docType, setDocType] = useState<string>('invoice');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadStatus('idle');
      setErrorMessage('');
    }
  };
  
  const handleUpload = async (e: React.FormEvent) => {
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
    } catch (error: unknown) {
      console.error('Upload error:', error);
      setUploadStatus('error');
      const errorMessage = error instanceof Error 
        ? error.message 
        : 'Error uploading document';
      setErrorMessage(errorMessage);
    } finally {
      setIsUploading(false);
    }
  };
  
  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 className="text-lg font-semibold mb-4">Upload Document</h2>
      
      <form onSubmit={handleUpload}>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Document Type
          </label>
          <div className="flex space-x-4">
            <label className={`flex items-center p-3 border rounded-md cursor-pointer ${
              docType === 'invoice' ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}>
              <input
                type="radio"
                name="docType"
                value="invoice"
                checked={docType === 'invoice'}
                onChange={() => setDocType('invoice')}
                className="sr-only"
              />
              <FiFileText className="h-5 w-5 mr-2 text-gray-600" />
              <span>Invoice</span>
            </label>
            
            <label className={`flex items-center p-3 border rounded-md cursor-pointer ${
              docType === 'shipment' ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}>
              <input
                type="radio"
                name="docType"
                value="shipment"
                checked={docType === 'shipment'}
                onChange={() => setDocType('shipment')}
                className="sr-only"
              />
              <FiTruck className="h-5 w-5 mr-2 text-gray-600" />
              <span>Shipment</span>
            </label>
            
            <label className={`flex items-center p-3 border rounded-md cursor-pointer ${
              docType === 'policy' ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}>
              <input
                type="radio"
                name="docType"
                value="policy"
                checked={docType === 'policy'}
                onChange={() => setDocType('policy')}
                className="sr-only"
              />
              <FiBook className="h-5 w-5 mr-2 text-gray-600" />
              <span>Policy</span>
            </label>
          </div>
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            File
          </label>
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                {file ? (
                  <>
                    <FiFile className="w-8 h-8 mb-2 text-gray-500" />
                    <p className="text-sm text-gray-500">{file.name}</p>
                    <p className="text-xs text-gray-500">
                      {(file.size / 1024).toFixed(2)} KB
                    </p>
                  </>
                ) : (
                  <>
                    <FiUpload className="w-8 h-8 mb-2 text-gray-500" />
                    <p className="text-sm text-gray-500">
                      <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-gray-500">
                      PDF or CSV (max. 10MB)
                    </p>
                  </>
                )}
              </div>
              <input
                type="file"
                className="hidden"
                accept=".pdf,.csv"
                onChange={handleFileChange}
              />
            </label>
          </div>
        </div>
        
        {uploadStatus === 'success' && (
          <div className="mb-4 p-3 bg-green-50 text-green-700 rounded-md flex items-center">
            <FiCheck className="h-5 w-5 mr-2" />
            <span>Document uploaded successfully!</span>
          </div>
        )}
        
        {uploadStatus === 'error' && (
          <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md flex items-center">
            <FiX className="h-5 w-5 mr-2" />
            <span>{errorMessage || 'Error uploading document'}</span>
          </div>
        )}
        
        <button
          type="submit"
          disabled={!file || isUploading}
          className={`w-full py-2 px-4 rounded-md ${
            !file || isUploading
              ? 'bg-gray-300 cursor-not-allowed'
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
        >
          {isUploading ? 'Uploading...' : 'Upload Document'}
        </button>
      </form>
    </div>
  );
}