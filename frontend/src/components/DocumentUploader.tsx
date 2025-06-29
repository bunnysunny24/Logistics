// src/components/DocumentUploader.tsx
"use client";

import { useState } from 'react';
import { uploadDocument } from '@/lib/api';
import { FiUpload, FiFileText, FiTruck, FiBook, FiCheck, FiX } from 'react-icons/fi';

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
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6 animate-fade-in">
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 flex items-center">
          <FiUpload className="mr-3 text-blue-600" />
          Document Upload Center
        </h2>
      </div>
      
      <form onSubmit={handleUpload} className="space-y-6">
        {/* Document Type Selection */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">
            Select Document Type
          </label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <label className={`relative flex items-center p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
              docType === 'invoice' 
                ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200' 
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }`}>
              <input
                type="radio"
                name="docType"
                value="invoice"
                checked={docType === 'invoice'}
                onChange={() => setDocType('invoice')}
                className="sr-only"
              />
              <div className="flex items-center">
                <FiFileText className={`h-6 w-6 mr-3 ${
                  docType === 'invoice' ? 'text-blue-600' : 'text-gray-500'
                }`} />
                <div>
                  <span className="font-medium text-gray-800">Invoice</span>
                  <p className="text-xs text-gray-500">Bills, receipts, payment docs</p>
                </div>
              </div>
              {docType === 'invoice' && (
                <FiCheck className="h-5 w-5 text-blue-600" />
              )}
            </label>
            
            <label className={`relative flex items-center justify-between p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
              docType === 'shipment' 
                ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200' 
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }`}>
              <input
                type="radio"
                name="docType"
                value="shipment"
                checked={docType === 'shipment'}
                onChange={() => setDocType('shipment')}
                className="sr-only"
              />
              <div className="flex items-center">
                <FiTruck className={`h-6 w-6 mr-3 ${
                  docType === 'shipment' ? 'text-blue-600' : 'text-gray-500'
                }`} />
                <div>
                  <span className="font-medium text-gray-800">Shipment</span>
                  <p className="text-xs text-gray-500">Delivery docs, BOL, manifests</p>
                </div>
              </div>
              {docType === 'shipment' && (
                <FiCheck className="h-5 w-5 text-blue-600" />
              )}
            </label>
            
            <label className={`relative flex items-center justify-between p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
              docType === 'policy' 
                ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200' 
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }`}>
              <input
                type="radio"
                name="docType"
                value="policy"
                checked={docType === 'policy'}
                onChange={() => setDocType('policy')}
                className="sr-only"
              />
              <div className="flex items-center">
                <FiBook className={`h-6 w-6 mr-3 ${
                  docType === 'policy' ? 'text-blue-600' : 'text-gray-500'
                }`} />
                <div>
                  <span className="font-medium text-gray-800">Policy</span>
                  <p className="text-xs text-gray-500">Guidelines, procedures, terms</p>
                </div>
              </div>
              {docType === 'policy' && (
                <FiCheck className="h-5 w-5 text-blue-600" />
              )}
            </label>
          </div>
        </div>
        
        {/* File Upload Section */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">
            Upload Document
          </label>
          <div className="flex items-center justify-center w-full">
            <label className={`flex flex-col items-center justify-center w-full h-40 border-3 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${
              file 
                ? 'border-green-400 bg-green-50 hover:bg-green-100' 
                : 'border-gray-300 bg-gray-50 hover:bg-gray-100 hover:border-gray-400'
            }`}>
              <div className="flex flex-col items-center justify-center py-6 px-4">
                {file ? (
                  <>
                    <FiCheck className="w-12 h-12 mb-3 text-green-500" />
                    <p className="text-lg font-medium text-green-700">{file.name}</p>
                    <p className="text-sm text-green-600">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB • Ready to upload
                    </p>
                  </>
                ) : (
                  <>
                    <FiUpload className="w-12 h-12 mb-3 text-gray-400" />
                    <p className="text-lg font-medium text-gray-700">
                      Drop your file here or <span className="text-blue-600">click to browse</span>
                    </p>
                    <p className="text-sm text-gray-500 mt-2">
                      Supports PDF, CSV, XLSX up to 10MB
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
        
        {/* Status Messages */}
        {uploadStatus === 'success' && (
          <div className="p-4 bg-green-50 border border-green-200 text-green-800 rounded-lg flex items-center animate-fade-in">
            <FiCheck className="h-6 w-6 mr-3 text-green-600" />
            <div>
              <p className="font-medium">Upload Successful!</p>
              <p className="text-sm text-green-700">Your document has been processed and analyzed.</p>
            </div>
          </div>
        )}

        {uploadStatus === 'error' && errorMessage && (
          <div className="p-4 bg-red-50 border border-red-200 text-red-800 rounded-lg flex items-center animate-fade-in">
            <FiX className="h-6 w-6 mr-3 text-red-600" />
            <div>
              <p className="font-medium">Upload Failed</p>
              <p className="text-sm text-red-700">{errorMessage}</p>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!file || isUploading}
          className={`w-full py-4 px-6 rounded-lg font-semibold text-lg transition-all duration-300 ${
            !file || isUploading
              ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 text-white hover:shadow-lg transform hover:-translate-y-0.5'
          }`}
        >
          {isUploading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
              Processing Document...
            </div>
          ) : (
            <div className="flex items-center justify-center">
              <FiUpload className="mr-3 h-6 w-6" />
              Upload & Analyze Document
            </div>
          )}
        </button>
      </form>
    </div>
  );
}