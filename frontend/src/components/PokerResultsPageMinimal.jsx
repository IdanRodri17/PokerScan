import React from 'react';

const PokerResultsPageMinimal = ({ gameData, onBack }) => {
  // Debug: Log what we're receiving
  console.log('PokerResultsPageMinimal received gameData:', gameData);
  
  if (!gameData) {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#1f2937', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: 'white', fontSize: '1.25rem' }}>Loading results...</div>
      </div>
    );
  }

  // If no game analysis, show simple results
  if (!gameData.game_analysis) {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#1f2937', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', color: 'white', maxWidth: '400px' }}>
          <h1 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Detection Complete</h1>
          <p style={{ marginBottom: '1rem' }}>
            Cards detected: {gameData.cards_detected ? gameData.cards_detected.join(', ') : 'None'}
          </p>
          <button 
            onClick={onBack}
            style={{ 
              backgroundColor: '#2563eb', 
              color: 'white', 
              padding: '0.5rem 1rem', 
              borderRadius: '0.25rem',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            Back to Upload
          </button>
        </div>
      </div>
    );
  }

  // If we have game analysis, show it
  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#1f2937', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ textAlign: 'center', color: 'white', maxWidth: '600px' }}>
        <h1 style={{ fontSize: '2rem', marginBottom: '1rem' }}>Poker Analysis Complete</h1>
        
        {gameData.game_analysis.players && gameData.game_analysis.players.length > 0 ? (
          <div style={{ marginBottom: '1rem' }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Players:</h2>
            {gameData.game_analysis.players.map((player, index) => (
              <div key={index} style={{ marginBottom: '0.5rem' }}>
                {player.name}: {player.hole_cards ? player.hole_cards.join(', ') : 'No cards'}
              </div>
            ))}
          </div>
        ) : null}
        
        {gameData.game_analysis.community_cards && gameData.game_analysis.community_cards.length > 0 ? (
          <div style={{ marginBottom: '1rem' }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Community Cards:</h2>
            <p>{gameData.game_analysis.community_cards.join(', ')}</p>
          </div>
        ) : null}
        
        {gameData.game_analysis.winner ? (
          <div style={{ marginBottom: '1rem' }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Winner:</h2>
            <p>{gameData.game_analysis.winner.name} with {gameData.game_analysis.winner.winning_hand}</p>
          </div>
        ) : null}
        
        <button 
          onClick={onBack}
          style={{ 
            backgroundColor: '#2563eb', 
            color: 'white', 
            padding: '0.5rem 1rem', 
            borderRadius: '0.25rem',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Back to Upload
        </button>
      </div>
    </div>
  );
};

export default PokerResultsPageMinimal;