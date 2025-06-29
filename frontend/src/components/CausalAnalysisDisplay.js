import React from 'react';
import { FaArrowRight, FaClock, FaExclamationTriangle, FaShieldAlt, FaLightbulb, FaCheckCircle } from 'react-icons/fa';

function CausalAnalysisDisplay({ causalAnalysis }) {
  if (!causalAnalysis || (!causalAnalysis.causal_chains?.length && !causalAnalysis.risk_holds?.length)) {
    return null;
  }

  const { causal_chains = [], risk_holds = [], reasoning_summary, confidence_score } = causalAnalysis;

  return (
    <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          <FaLightbulb className="text-blue-600 mr-2" />
          <h3 className="text-lg font-semibold text-blue-800">🧠 Causal Analysis</h3>
        </div>
        {confidence_score > 0 && (
          <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
            Confidence: {(confidence_score * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {reasoning_summary && (
        <div className="mb-4 p-3 bg-white rounded-lg border border-gray-200">
          <p className="text-sm text-gray-700">{reasoning_summary}</p>
        </div>
      )}

      {/* Risk-Based Holds */}
      {risk_holds && risk_holds.length > 0 && (
        <div className="mb-4 p-3 bg-red-50 rounded-lg border border-red-200">
          <div className="flex items-center mb-3">
            <FaShieldAlt className="text-red-600 mr-2" />
            <h4 className="font-semibold text-red-800">🚨 Active Risk-Based Holds</h4>
          </div>
          {risk_holds.map((hold, index) => (
            <div key={hold.id || index} className="mb-3 p-3 bg-white rounded border-l-4 border-red-400">
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <div className="flex items-center">
                    <span className="font-medium text-red-700">{hold.document_id}</span>
                    <span className="ml-2 bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs">
                      {hold.hold_type.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 mt-1">{hold.reason}</p>
                  {hold.requires_approval && (
                    <div className="flex items-center mt-2">
                      <FaExclamationTriangle className="text-yellow-500 mr-1" />
                      <span className="text-xs text-yellow-700">
                        Requires {hold.approver_type.replace('_', ' ')} approval
                      </span>
                    </div>
                  )}
                </div>
                <div className="text-right">
                  <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs font-medium">
                    Risk: {(hold.risk_score * 100).toFixed(0)}%
                  </span>
                  <div className="mt-1">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      hold.status === 'active' ? 'bg-red-100 text-red-800' :
                      hold.status === 'pending_review' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {hold.status.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Causal Chains */}
      {causal_chains && causal_chains.length > 0 && (
        <div>
          <div className="flex items-center mb-3">
            <FaArrowRight className="text-blue-600 mr-2" />
            <h4 className="font-semibold text-blue-700">Cause-and-Effect Analysis</h4>
          </div>
          {causal_chains.map((chain, chainIndex) => (
            <div key={chain.id || chainIndex} className="mb-4 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center">
                  <span className="font-medium text-gray-800">{chain.id}</span>
                  <span className="ml-2 bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                    {chain.impact}
                  </span>
                </div>
                <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">
                  Confidence: {(chain.confidence * 100).toFixed(0)}%
                </span>
              </div>

              {/* Causal Flow */}
              <div className="space-y-3">
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-4 h-4 bg-orange-400 rounded-full mt-1 mr-3"></div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-700 mb-1">ROOT CAUSE</div>
                    <p className="text-sm text-gray-600">{chain.cause}</p>
                  </div>
                </div>
                
                <div className="ml-8 w-px h-6 bg-gray-300"></div>
                
                <div className="flex items-center ml-6">
                  <FaArrowRight className="text-blue-500 mr-2" />
                  <span className="text-sm font-medium text-blue-600">LEADS TO</span>
                </div>
                
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-4 h-4 bg-red-400 rounded-full mt-1 mr-3"></div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-700 mb-1">EFFECT</div>
                    <p className="text-sm text-gray-600">{chain.effect}</p>
                  </div>
                </div>

                {/* Evidence */}
                {chain.evidence && chain.evidence.length > 0 && (
                  <div className="mt-3 p-2 bg-gray-50 rounded">
                    <div className="flex items-center mb-2">
                      <FaCheckCircle className="text-green-500 mr-1" />
                      <span className="text-xs font-medium text-gray-700">SUPPORTING EVIDENCE</span>
                    </div>
                    <ul className="text-xs text-gray-600 space-y-1">
                      {chain.evidence.map((evidence, evidenceIndex) => (
                        <li key={evidenceIndex} className="flex items-start">
                          <span className="text-gray-400 mr-2">•</span>
                          <span>{evidence}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default CausalAnalysisDisplay;
