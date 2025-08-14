import React from 'react';

const ResultsDisplay = ({ result, error }) => {
  if (error) {
    return (
      <div className="card">
        <div className="text-center">
          <div className="text-red-500 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-red-600 mb-2">Error</h3>
          <p className="text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  return (
    <div className="card">
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-900 mb-2">Detection Results</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Status:</span>
            <span className={`ml-2 font-medium ${result.success ? 'text-green-600' : 'text-red-600'}`}>
              {result.success ? 'Success' : 'Failed'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Processing Time:</span>
            <span className="ml-2 font-medium text-gray-900">
              {result.processing_time ? `${result.processing_time.toFixed(2)}s` : 'N/A'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Filename:</span>
            <span className="ml-2 font-medium text-gray-900">{result.filename || 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-500">Timestamp:</span>
            <span className="ml-2 font-medium text-gray-900">
              {result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {result.cards_detected && result.cards_detected.length > 0 && (
        <div>
          <h4 className="text-md font-medium text-gray-900 mb-3">Detected Cards</h4>
          <div className="grid gap-2">
            {result.cards_detected.map((card, index) => (
              <div
                key={index}
                className="bg-gray-50 rounded-lg p-3 flex items-center justify-between"
              >
                <span className="font-medium text-gray-900">{card}</span>
                <div className="flex items-center text-green-600">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(!result.cards_detected || result.cards_detected.length === 0) && (
        <div className="text-center py-4">
          <div className="text-gray-400 mb-2">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.29-1.044-5.709-2.573M15 9.34c0-1.357-.107-2.67-.31-3.934A6 6 0 1112.84 3.34c1.264.203 2.577.31 3.934.31" />
            </svg>
          </div>
          <p className="text-gray-500">No cards detected in this image</p>
        </div>
      )}

      <div className="mt-4 text-center">
        <p className="text-xs text-gray-500">{result.message}</p>
      </div>
    </div>
  );
};

export default ResultsDisplay;