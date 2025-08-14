import React, { useState } from 'react';
import { useCamera } from '../hooks/useCamera';
import { apiService } from '../services/api';

const CameraCapture = ({ onCaptureSuccess, onCaptureError }) => {
  const { videoRef, isStreaming, error, startCamera, stopCamera, capturePhoto } = useCamera();
  const [isCapturing, setIsCapturing] = useState(false);

  const handleStartCamera = async () => {
    await startCamera();
  };

  const handleCapture = async () => {
    try {
      setIsCapturing(true);
      const photoBlob = await capturePhoto();
      
      if (photoBlob) {
        const file = new File([photoBlob], 'camera-capture.jpg', { type: 'image/jpeg' });
        const result = await apiService.uploadImage(file);
        onCaptureSuccess?.(result);
      }
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.message || 'Capture failed';
      onCaptureError?.(errorMessage);
    } finally {
      setIsCapturing(false);
    }
  };

  if (error) {
    return (
      <div className="card">
        <div className="text-center">
          <div className="text-error-500 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Camera Error</h3>
          <p className="text-gray-600 mb-4">{error}</p>
          <button onClick={handleStartCamera} className="btn-primary">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      {!isStreaming ? (
        <div className="text-center">
          <div className="text-gray-400 mb-4">
            <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Camera Capture</h3>
          <p className="text-gray-600 mb-4">Take a photo of your poker cards</p>
          <button onClick={handleStartCamera} className="btn-primary">
            Start Camera
          </button>
        </div>
      ) : (
        <div>
          <div className="relative mb-4">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full rounded-lg bg-black"
              style={{ maxHeight: '400px' }}
            />
            {isCapturing && (
              <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
                <div className="text-white text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-2"></div>
                  <p>Processing...</p>
                </div>
              </div>
            )}
          </div>
          
          <div className="flex gap-3 justify-center">
            <button
              onClick={handleCapture}
              disabled={isCapturing}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isCapturing ? 'Capturing...' : 'Capture Photo'}
            </button>
            <button onClick={stopCamera} className="btn-secondary">
              Stop Camera
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraCapture;