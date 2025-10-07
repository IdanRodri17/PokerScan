import React, { useState, Suspense, lazy } from 'react';

// API URL from environment variable (set in Netlify)
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Lazy load components for better performance
const PokerLandingPage = lazy(() => import('./components/PokerLandingPage'));
const PokerResultsPage = lazy(() => import('./components/PokerResultsPage'));

// Loading component
const LoadingSpinner = () => (
  <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black flex items-center justify-center">
    <div className="flex flex-col items-center gap-4">
      <div className="w-12 h-12 border-4 border-yellow-400 border-t-transparent rounded-full animate-spin"></div>
      <p className="text-white font-semibold">Loading...</p>
    </div>
  </div>
);

function App() {
  const [currentView, setCurrentView] = useState('landing'); // 'landing' or 'results'
  const [gameResults, setGameResults] = useState(null);
  const [originalImageURL, setOriginalImageURL] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    setIsProcessing(true);
    setError(null);

    // Create a URL for the uploaded image to display in modal
    const imageURL = URL.createObjectURL(file);
    setOriginalImageURL(imageURL);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Call your backend API
      const response = await fetch(`${API_URL}/upload?create_visualization=true`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setGameResults(data);
        setCurrentView('results');
      } else {
        throw new Error(data.message || 'Failed to analyze image');
      }

    } catch (err) {
      console.error('Upload error:', err);
      setError(err.message || 'Failed to analyze poker image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleBackToUpload = () => {
    setCurrentView('landing');
    setGameResults(null);
    setOriginalImageURL(null);
    setError(null);
  };

  const handleAnalyzeAnother = () => {
    setCurrentView('landing');
    setGameResults(null);
    setOriginalImageURL(null);
    setError(null);
  };

  return (
    <div className="App">
      {/* Error Toast */}
      {error && (
        <div className="fixed top-4 right-4 z-50 bg-red-500/90 backdrop-blur-sm 
                        text-white p-4 rounded-lg shadow-lg border border-red-400/30
                        max-w-md animate-in slide-in-from-right">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-5 h-5 rounded-full bg-red-400 
                            flex items-center justify-center mt-0.5">
              <span className="text-xs font-bold">!</span>
            </div>
            <div>
              <p className="font-semibold mb-1">Upload Error</p>
              <p className="text-sm opacity-90">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="flex-shrink-0 text-white/70 hover:text-white ml-auto"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <Suspense fallback={<LoadingSpinner />}>
        {currentView === 'landing' && (
          <PokerLandingPage
            onImageUpload={handleImageUpload}
            isProcessing={isProcessing}
          />
        )}

        {currentView === 'results' && gameResults && (
          <PokerResultsPage
            gameData={gameResults}
            originalImage={originalImageURL}
            onBack={handleBackToUpload}
            onAnalyzeAnother={handleAnalyzeAnother}
          />
        )}
      </Suspense>
    </div>
  );
}

export default App;