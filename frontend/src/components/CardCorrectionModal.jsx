import React, { useState, useRef, useCallback, useEffect } from 'react';

// Card values and suits for selection
const CARD_VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'];
const CARD_SUITS = [
  { symbol: '♠', name: 'spades', code: 's' },
  { symbol: '♥', name: 'hearts', code: 'h' },
  { symbol: '♦', name: 'diamonds', code: 'd' },
  { symbol: '♣', name: 'clubs', code: 'c' }
];

const CardCorrectionModal = ({ 
  originalImage, 
  detectedCards, 
  onSave, 
  onCancel,
  isOpen = false 
}) => {
  const [correctedCards, setCorrectedCards] = useState([]);
  const [selectedCard, setSelectedCard] = useState(null);
  const [showCardSelector, setShowCardSelector] = useState(false);
  const [addingCardPosition, setAddingCardPosition] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // Debug: Log when modal opens
  useEffect(() => {
    if (isOpen) {
      console.log('🚪 Modal opened!');
      console.log('📦 Received detectedCards prop:', detectedCards);
      console.log('🖼️ Received originalImage prop:', originalImage);
    }
  }, [isOpen]);

  // Initialize corrected cards and history
  useEffect(() => {
    if (detectedCards && detectedCards.length > 0 && isOpen) {
      console.log('🔧 Initializing modal with detected cards:', detectedCards);
      const initialCards = detectedCards.map((card, index) => ({
        ...card,
        id: card.id || `card-${index}`,
        // Use existing group if available, otherwise determine from position
        group: card.group || determineGroup(card)
      }));
      console.log('✅ Initial cards with groups:', initialCards);
      setCorrectedCards(initialCards);
      setHistory([initialCards]);
      setHistoryIndex(0);
    } else if (isOpen) {
      console.log('⚠️ Modal opened but no detected cards!');
      console.log('   detectedCards:', detectedCards);
      console.log('   detectedCards.length:', detectedCards?.length);
    }
  }, [detectedCards, isOpen]);

  // Determine which group a card belongs to based on its position
  const determineGroup = (card) => {
    if (!card.center) return 'community';
    
    const imageHeight = imageRef.current?.naturalHeight || 800;
    const yRatio = card.center[1] / imageHeight;
    
    if (yRatio < 0.33) return 'player1';
    if (yRatio > 0.66) return 'player2';
    return 'community';
  };

  // Add to history for undo/redo
  const addToHistory = useCallback((newCards) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newCards);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  }, [history, historyIndex]);

  // Undo/Redo functionality
  const undo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setCorrectedCards(history[historyIndex - 1]);
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setCorrectedCards(history[historyIndex + 1]);
    }
  };

  // Handle image click to add new card
  const handleImageClick = (event) => {
    if (!imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const scaleX = imageRef.current.naturalWidth / rect.width;
    const scaleY = imageRef.current.naturalHeight / rect.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    setAddingCardPosition({ x, y });
    setShowCardSelector(true);
  };

  // Add new card with duplicate check and card limit validation
  const addCard = (value, suit) => {
    if (!addingCardPosition) return;

    const cardName = `${value}${suit.code}`;

    // Check for duplicate cards
    const isDuplicate = correctedCards.some(card => card.card_name === cardName);
    if (isDuplicate) {
      alert(`⚠️ Card ${value}${suit.symbol} already exists!\n\nEach card can only appear once in a deck.`);
      setShowCardSelector(false);
      setAddingCardPosition(null);
      return;
    }

    const group = determineGroup({
      center: [addingCardPosition.x, addingCardPosition.y]
    });

    // Check card limits for each group
    const groupCounts = {
      player1: correctedCards.filter(c => c.group === 'player1').length,
      player2: correctedCards.filter(c => c.group === 'player2').length,
      community: correctedCards.filter(c => c.group === 'community').length
    };

    const limits = {
      player1: 2,
      player2: 2,
      community: 5
    };

    if (groupCounts[group] >= limits[group]) {
      const groupNames = {
        player1: 'Player 1',
        player2: 'Player 2',
        community: 'Community'
      };
      alert(`⚠️ ${groupNames[group]} already has ${limits[group]} cards (maximum limit).\n\nPlease remove a card before adding a new one.`);
      setShowCardSelector(false);
      setAddingCardPosition(null);
      return;
    }

    const newCard = {
      id: `added-card-${Date.now()}`,
      card_name: cardName,
      confidence: 1.0, // Manual cards get full confidence
      bbox: [
        addingCardPosition.x - 30,
        addingCardPosition.y - 40,
        addingCardPosition.x + 30,
        addingCardPosition.y + 40
      ],
      center: [addingCardPosition.x, addingCardPosition.y],
      group,
      isManual: true
    };

    const updatedCards = [...correctedCards, newCard];
    setCorrectedCards(updatedCards);
    addToHistory(updatedCards);

    setShowCardSelector(false);
    setAddingCardPosition(null);
  };

  // Remove card
  const removeCard = (cardId) => {
    const updatedCards = correctedCards.filter(card => card.id !== cardId);
    setCorrectedCards(updatedCards);
    addToHistory(updatedCards);
  };

  // Edit card with duplicate check
  const editCard = (cardId, newValue, newSuit) => {
    const newCardName = `${newValue}${newSuit.code}`;

    // Check if the new card name already exists (excluding the card being edited)
    const isDuplicate = correctedCards.some(
      card => card.id !== cardId && card.card_name === newCardName
    );

    if (isDuplicate) {
      alert(`⚠️ Card ${newValue}${newSuit.symbol} already exists!\n\nEach card can only appear once in a deck.`);
      return;
    }

    const updatedCards = correctedCards.map(card =>
      card.id === cardId
        ? { ...card, card_name: newCardName }
        : card
    );
    setCorrectedCards(updatedCards);
    addToHistory(updatedCards);
    setSelectedCard(null);
  };

  // Move card between groups with limit check
  const moveCard = (cardId, newGroup) => {
    // Check card limits for the target group
    const groupCounts = {
      player1: correctedCards.filter(c => c.group === 'player1' && c.id !== cardId).length,
      player2: correctedCards.filter(c => c.group === 'player2' && c.id !== cardId).length,
      community: correctedCards.filter(c => c.group === 'community' && c.id !== cardId).length
    };

    const limits = {
      player1: 2,
      player2: 2,
      community: 5
    };

    if (groupCounts[newGroup] >= limits[newGroup]) {
      const groupNames = {
        player1: 'Player 1',
        player2: 'Player 2',
        community: 'Community'
      };
      alert(`⚠️ ${groupNames[newGroup]} already has ${limits[newGroup]} cards (maximum limit).\n\nPlease remove a card from ${groupNames[newGroup]} before moving this card.`);
      return;
    }

    const updatedCards = correctedCards.map(card =>
      card.id === cardId
        ? { ...card, group: newGroup }
        : card
    );

    setCorrectedCards(updatedCards);
    addToHistory(updatedCards);
  };

  // Group cards by position
  const groupedCards = {
    player1: correctedCards.filter(card => card.group === 'player1'),
    community: correctedCards.filter(card => card.group === 'community'),
    player2: correctedCards.filter(card => card.group === 'player2')
  };

  // Debug logging
  console.log('🃏 Current correctedCards:', correctedCards);
  console.log('📊 Grouped cards:', groupedCards);

  // Card component
  const CardComponent = ({ card }) => {
    const isLowConfidence = card.confidence < 0.7;
    const cardDisplay = formatCardName(card.card_name);

    return (
      <div
        className={`card-item ${isLowConfidence ? 'low-confidence' : ''}`}
        onClick={() => setSelectedCard(card)}
      >
        <div className="card-content">
          <span className="card-text">{cardDisplay}</span>
          <span className="confidence">{(card.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="card-actions">
          <div className="move-buttons">
            {card.group !== 'player1' && (
              <button 
                className="move-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  moveCard(card.id, 'player1');
                }}
                title="Move to Player 1"
              >
                ↑
              </button>
            )}
            {card.group !== 'community' && (
              <button 
                className="move-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  moveCard(card.id, 'community');
                }}
                title="Move to Community"
              >
                ↔
              </button>
            )}
            {card.group !== 'player2' && (
              <button 
                className="move-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  moveCard(card.id, 'player2');
                }}
                title="Move to Player 2"
              >
                ↓
              </button>
            )}
          </div>
          <button 
            className="remove-btn"
            onClick={(e) => {
              e.stopPropagation();
              removeCard(card.id);
            }}
          >
            ×
          </button>
        </div>
      </div>
    );
  };

  // Format card name for display
  const formatCardName = (cardName) => {
    if (!cardName || cardName.length < 2) return cardName;
    
    const value = cardName.slice(0, -1);
    const suit = cardName.slice(-1);
    
    const suitSymbol = CARD_SUITS.find(s => s.code === suit)?.symbol || suit;
    return `${value}${suitSymbol}`;
  };

  // Card group component with card count and limits
  const CardGroup = ({ groupName, title, cards }) => {
    const limits = {
      player1: 2,
      player2: 2,
      community: 5
    };

    const currentCount = cards.length;
    const maxCount = limits[groupName];
    const isAtLimit = currentCount >= maxCount;

    console.log(`📋 ${groupName} group received ${cards.length} cards:`, cards);

    return (
      <div className="card-group">
        <h3 className="group-title">
          {title}
          <span style={{ fontSize: '0.8em', marginLeft: '10px', opacity: 0.7 }}>
            ({currentCount}/{maxCount} cards)
          </span>
        </h3>
        <div className="cards-container">
          {cards.length === 0 && (
            <div style={{ color: '#888', fontSize: '0.9em', padding: '10px' }}>
              No cards detected in this position
            </div>
          )}
          {cards.map((card) => (
            <CardComponent key={card.id} card={card} />
          ))}
          <button
            className="add-card-btn"
            disabled={isAtLimit}
            onClick={() => {
              // Set position based on group
              const groupPositions = {
                player1: { x: 400, y: 150 },
                community: { x: 400, y: 400 },
                player2: { x: 400, y: 650 }
              };
              setAddingCardPosition(groupPositions[groupName]);
              setShowCardSelector(true);
            }}
            title={isAtLimit ? `Maximum ${maxCount} cards allowed` : 'Add a card'}
          >
            {isAtLimit ? '✓ Full' : '+ Add Card'}
          </button>
        </div>
      </div>
    );
  };

  // Card selector modal
  const CardSelector = ({ onSelect, onClose }) => {
    const [selectedValue, setSelectedValue] = useState('A');
    const [selectedSuit, setSelectedSuit] = useState(CARD_SUITS[0]);

    return (
      <div className="card-selector-modal">
        <div className="card-selector-content">
          <h3>Select Card</h3>
          <div className="card-selection">
            <div className="value-selection">
              <label>Value:</label>
              <div className="value-grid">
                {CARD_VALUES.map(value => (
                  <button
                    key={value}
                    className={`value-btn ${selectedValue === value ? 'selected' : ''}`}
                    onClick={() => setSelectedValue(value)}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>
            <div className="suit-selection">
              <label>Suit:</label>
              <div className="suit-grid">
                {CARD_SUITS.map(suit => (
                  <button
                    key={suit.code}
                    className={`suit-btn ${selectedSuit.code === suit.code ? 'selected' : ''}`}
                    onClick={() => setSelectedSuit(suit)}
                  >
                    {suit.symbol}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="card-preview">
            Preview: {selectedValue}{selectedSuit.symbol}
          </div>
          <div className="card-selector-actions">
            <button onClick={() => onSelect(selectedValue, selectedSuit)}>
              Add Card
            </button>
            <button onClick={onClose}>Cancel</button>
          </div>
        </div>
      </div>
    );
  };

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.ctrlKey || event.metaKey) {
        if (event.key === 'z' && !event.shiftKey) {
          event.preventDefault();
          undo();
        } else if (event.key === 'z' && event.shiftKey || event.key === 'y') {
          event.preventDefault();
          redo();
        }
      }
      
      if (event.key === 'Delete' && selectedCard) {
        removeCard(selectedCard.id);
        setSelectedCard(null);
      }
    };

    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen, selectedCard, historyIndex]);

  if (!isOpen) return null;

  return (
    <div className="card-correction-modal">
      <div className="modal-overlay" onClick={onCancel} />
      <div className="modal-content">
        <div className="modal-header">
          <h2>Manual Card Correction</h2>
          <div className="header-controls">
            <button 
              onClick={undo} 
              disabled={historyIndex <= 0}
              className="undo-btn"
            >
              ↶ Undo
            </button>
            <button 
              onClick={redo} 
              disabled={historyIndex >= history.length - 1}
              className="redo-btn"
            >
              ↷ Redo
            </button>
            <button onClick={onCancel} className="close-btn">×</button>
          </div>
        </div>

        <div className="modal-body">
          <div className="image-section">
            <div className="image-container">
              <img
                ref={imageRef}
                src={originalImage}
                alt="Poker table"
                onClick={handleImageClick}
                className="poker-image"
              />
              <canvas
                ref={canvasRef}
                className="detection-overlay"
                onClick={handleImageClick}
              />
            </div>
            <div className="image-instructions">
              Click on the image to add missing cards
            </div>
          </div>

          <div className="cards-section">
            <CardGroup 
              groupName="player1" 
              title="Player 1 (Top)" 
              cards={groupedCards.player1} 
            />
            <CardGroup 
              groupName="community" 
              title="Community Cards" 
              cards={groupedCards.community} 
            />
            <CardGroup 
              groupName="player2" 
              title="Player 2 (Bottom)" 
              cards={groupedCards.player2} 
            />
          </div>
        </div>

        <div className="modal-footer">
          <div className="footer-info">
            <span>Total Cards: {correctedCards.length}</span>
            <span>Manual Additions: {correctedCards.filter(c => c.isManual).length}</span>
          </div>
          <div className="footer-actions">
            <button 
              onClick={() => {
                setCorrectedCards(detectedCards.map((card, index) => ({
                  ...card,
                  id: `card-${index}`,
                  group: determineGroup(card)
                })));
              }}
              className="reset-btn"
            >
              Reset to Original
            </button>
            <button onClick={onCancel} className="cancel-btn">
              Cancel
            </button>
            <button 
              onClick={() => onSave(correctedCards)}
              className="save-btn"
            >
              Confirm Corrections
            </button>
          </div>
        </div>

        {/* Card Selector Modal */}
        {showCardSelector && (
          <CardSelector
            onSelect={addCard}
            onClose={() => {
              setShowCardSelector(false);
              setAddingCardPosition(null);
            }}
          />
        )}

        {/* Edit Card Modal */}
        {selectedCard && (
          <CardSelector
            onSelect={(value, suit) => editCard(selectedCard.id, value, suit)}
            onClose={() => setSelectedCard(null)}
          />
        )}
      </div>

      <style jsx>{`
        .card-correction-modal {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .modal-overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.8);
        }

        .modal-content {
          position: relative;
          background: #1a1a1a;
          color: #ffffff;
          border-radius: 12px;
          width: 95vw;
          height: 90vh;
          max-width: 1400px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }

        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          border-bottom: 1px solid #333;
          background: #2a2a2a;
        }

        .modal-header h2 {
          margin: 0;
          color: #ffd700;
        }

        .header-controls {
          display: flex;
          gap: 10px;
        }

        .header-controls button {
          padding: 8px 12px;
          background: #444;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .header-controls button:hover:not(:disabled) {
          background: #555;
        }

        .header-controls button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .close-btn {
          font-size: 20px !important;
          width: 32px !important;
          height: 32px !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
          padding: 0 !important;
        }

        .modal-body {
          flex: 1;
          display: flex;
          overflow: hidden;
        }

        .image-section {
          flex: 1;
          padding: 20px;
          display: flex;
          flex-direction: column;
        }

        .image-container {
          position: relative;
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #333;
          border-radius: 8px;
          overflow: hidden;
        }

        .poker-image {
          max-width: 100%;
          max-height: 100%;
          cursor: crosshair;
          border-radius: 8px;
        }

        .detection-overlay {
          position: absolute;
          top: 0;
          left: 0;
          pointer-events: none;
        }

        .image-instructions {
          text-align: center;
          margin-top: 10px;
          color: #ccc;
          font-size: 14px;
        }

        .cards-section {
          width: 400px;
          padding: 20px;
          background: #2a2a2a;
          border-left: 1px solid #333;
          overflow-y: auto;
        }

        /* Mobile Responsive Layout */
        @media (max-width: 768px) {
          .modal-content {
            width: 100vw;
            height: 100vh;
            max-width: 100vw;
            border-radius: 0;
          }

          .modal-body {
            flex-direction: column;
          }

          .image-section {
            flex: none;
            height: 40vh;
            min-height: 250px;
            padding: 10px;
          }

          .image-container {
            min-height: 200px;
          }

          .cards-section {
            width: 100%;
            flex: 1;
            border-left: none;
            border-top: 1px solid #333;
            padding: 15px;
          }

          .modal-header h2 {
            font-size: 18px;
          }

          .header-controls {
            gap: 5px;
          }

          .header-controls button {
            padding: 6px 10px;
            font-size: 12px;
          }
        }

        /* Extra small phones */
        @media (max-width: 480px) {
          .image-section {
            height: 35vh;
            min-height: 200px;
            padding: 8px;
          }

          .cards-section {
            padding: 10px;
          }

          .card-group {
            margin-bottom: 15px;
            padding: 10px;
          }

          .group-title {
            font-size: 14px;
          }

          .cards-container {
            gap: 8px;
          }

          .card-item {
            min-width: 70px;
            padding: 6px 6px 28px 6px;
          }

          .card-text {
            font-size: 14px;
          }

          .move-btn, .remove-btn {
            width: 18px;
            height: 18px;
            font-size: 11px;
          }

          .add-card-btn {
            padding: 8px;
            font-size: 12px;
          }

          .modal-header {
            padding: 12px;
          }

          .modal-header h2 {
            font-size: 16px;
          }

          .footer-info {
            flex-direction: column;
            gap: 5px;
            font-size: 12px;
          }

          .footer-actions {
            flex-wrap: wrap;
            gap: 8px;
          }

          .footer-actions button {
            padding: 8px 12px;
            font-size: 12px;
            flex: 1;
            min-width: 100px;
          }

          .image-instructions {
            font-size: 12px;
          }
        }

        .card-group {
          margin-bottom: 30px;
          padding: 15px;
          background: #333;
          border-radius: 8px;
          transition: all 0.2s ease;
        }

        .card-group.drag-over {
          background: #444;
          border: 2px dashed #ffd700;
        }

        .group-title {
          margin: 0 0 15px 0;
          color: #ffd700;
          font-size: 16px;
          font-weight: 600;
        }

        .cards-container {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
        }

        .card-item {
          position: relative;
          background: #444;
          border: 2px solid #555;
          border-radius: 8px;
          padding: 8px 8px 30px 8px;
          min-width: 80px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .card-item:hover {
          background: #555;
          border-color: #ffd700;
        }

        .card-item.low-confidence {
          border-color: #ff8c00;
          background: rgba(255, 140, 0, 0.1);
        }

        .card-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin-bottom: 5px;
        }

        .card-text {
          font-size: 16px;
          font-weight: bold;
          color: white;
        }

        .confidence {
          font-size: 10px;
          color: #ccc;
        }

        .card-actions {
          position: absolute;
          bottom: 2px;
          left: 2px;
          right: 2px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .move-buttons {
          display: flex;
          gap: 2px;
        }

        .move-btn {
          width: 16px;
          height: 16px;
          background: #6c757d;
          color: white;
          border: none;
          border-radius: 3px;
          font-size: 10px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
        }

        .move-btn:hover {
          background: #5a6268;
          transform: scale(1.1);
        }

        .remove-btn {
          width: 16px;
          height: 16px;
          background: #dc3545;
          color: white;
          border: none;
          border-radius: 3px;
          font-size: 10px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .remove-btn:hover {
          background: #c82333;
        }

        .add-card-btn {
          background: #28a745;
          color: white;
          border: none;
          border-radius: 8px;
          padding: 10px;
          cursor: pointer;
          font-size: 12px;
          transition: all 0.2s ease;
        }

        .add-card-btn:hover {
          background: #218838;
        }

        .modal-footer {
          padding: 20px;
          border-top: 1px solid #333;
          background: #2a2a2a;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .footer-info {
          display: flex;
          gap: 20px;
          color: #ccc;
          font-size: 14px;
        }

        .footer-actions {
          display: flex;
          gap: 10px;
        }

        .footer-actions button {
          padding: 10px 20px;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 500;
          transition: all 0.2s ease;
        }

        .reset-btn {
          background: #6c757d;
          color: white;
        }

        .reset-btn:hover {
          background: #5a6268;
        }

        .cancel-btn {
          background: #dc3545;
          color: white;
        }

        .cancel-btn:hover {
          background: #c82333;
        }

        .save-btn {
          background: #ffd700;
          color: #000;
        }

        .save-btn:hover {
          background: #ffed4e;
        }

        .card-selector-modal {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 100;
        }

        .card-selector-content {
          background: #2a2a2a;
          border-radius: 12px;
          padding: 30px;
          width: 400px;
          max-width: 90vw;
        }

        .card-selector-content h3 {
          margin: 0 0 20px 0;
          color: #ffd700;
          text-align: center;
        }

        .card-selection {
          margin-bottom: 20px;
        }

        .value-selection,
        .suit-selection {
          margin-bottom: 20px;
        }

        .value-selection label,
        .suit-selection label {
          display: block;
          margin-bottom: 10px;
          color: #ccc;
          font-weight: 500;
        }

        .value-grid {
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 8px;
        }

        .suit-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 8px;
        }

        .value-btn,
        .suit-btn {
          padding: 10px;
          background: #444;
          color: white;
          border: 2px solid #555;
          border-radius: 6px;
          cursor: pointer;
          font-weight: bold;
          transition: all 0.2s ease;
        }

        .value-btn:hover,
        .suit-btn:hover {
          background: #555;
          border-color: #ffd700;
        }

        .value-btn.selected,
        .suit-btn.selected {
          background: #ffd700;
          color: #000;
          border-color: #ffd700;
        }

        .suit-btn {
          font-size: 18px;
        }

        .card-preview {
          text-align: center;
          font-size: 24px;
          font-weight: bold;
          color: #ffd700;
          margin-bottom: 20px;
          padding: 15px;
          background: #333;
          border-radius: 8px;
        }

        .card-selector-actions {
          display: flex;
          gap: 10px;
          justify-content: center;
        }

        .card-selector-actions button {
          padding: 10px 20px;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 500;
          transition: all 0.2s ease;
        }

        .card-selector-actions button:first-child {
          background: #28a745;
          color: white;
        }

        .card-selector-actions button:first-child:hover {
          background: #218838;
        }

        .card-selector-actions button:last-child {
          background: #6c757d;
          color: white;
        }

        .card-selector-actions button:last-child:hover {
          background: #5a6268;
        }

        @media (max-width: 768px) {
          .modal-content {
            width: 100vw;
            height: 100vh;
            border-radius: 0;
          }

          .modal-body {
            flex-direction: column;
          }

          .image-section {
            flex: 0 0 40%;
          }

          .cards-section {
            width: 100%;
            flex: 1;
          }
        }
      `}</style>
    </div>
  );
};

export default CardCorrectionModal;