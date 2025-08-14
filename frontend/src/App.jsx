import React, { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import CameraCapture from './components/CameraCapture';
import ResultsDisplay from './components/ResultsDisplay';
import { apiService } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [healthStatus, setHealthStatus] = useState('checking');

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      await apiService.healthCheck();
      setHealthStatus('healthy');
    } catch (err) {
      setHealthStatus('error');
      console.error('Health check failed:', err);
    }
  };

  const handleSuccess = (result) => {
    setResult(result);
    setError(null);
  };

  const handleError = (error) => {
    setError(error);
    setResult(null);
  };

  const clearResults = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🃏 PokerVision
          </h1>
          <p className="text-lg text-gray-600">
            AI-powered poker card detection
          </p>
          <div className="mt-4">
            <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              healthStatus === 'healthy' 
                ? 'bg-green-50 text-green-700' 
                : healthStatus === 'error'
                ? 'bg-red-50 text-red-700'
                : 'bg-gray-50 text-gray-700'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                healthStatus === 'healthy' 
                  ? 'bg-green-400' 
                  : healthStatus === 'error'
                  ? 'bg-red-400'
                  : 'bg-gray-400 animate-pulse'
              }`}></div>
              {healthStatus === 'healthy' ? 'API Connected' : 
               healthStatus === 'error' ? 'API Disconnected' : 'Checking...'}
            </span>
          </div>
        </header>

        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
            <div className="flex flex-wrap gap-2 mb-6">
              <button
                onClick={() => setActiveTab('upload')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeTab === 'upload'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                📁 Upload Image
              </button>
              <button
                onClick={() => setActiveTab('camera')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeTab === 'camera'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                📷 Camera Capture
              </button>
              {(result || error) && (
                <button
                  onClick={clearResults}
                  className="px-4 py-2 rounded-lg font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 ml-auto"
                >
                  🗑️ Clear Results
                </button>
              )}
            </div>

            {activeTab === 'upload' && (
              <ImageUpload
                onUploadSuccess={handleSuccess}
                onUploadError={handleError}
              />
            )}

            {activeTab === 'camera' && (
              <CameraCapture
                onCaptureSuccess={handleSuccess}
                onCaptureError={handleError}
              />
            )}
          </div>

          {(result || error) && (
            <ResultsDisplay result={result} error={error} />
          )}
        </div>

        <footer className="text-center mt-12 text-gray-500">
          <p>Built with React, FastAPI, and ❤️</p>
          <p className="text-sm mt-1">Ready for YOLOv8 integration</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
