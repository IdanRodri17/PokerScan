import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Share2, RotateCcw, Edit3 } from 'lucide-react';
import WinnerAnnouncement from './WinnerAnnouncement';
import PokerTableView from './PokerTableView';
import HandComparisonPanel from './HandComparisonPanel';
import CardCorrectionModal from './CardCorrectionModal';

// API URL from environment variable
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PokerResultsPage = ({ 
  gameData, 
  onBack, 
  onAnalyzeAnother,
  originalImage 
}) => {
  const [currentSection, setCurrentSection] = useState(0);
  const [showWinner, setShowWinner] = useState(false);
  const [showCorrectionModal, setShowCorrectionModal] = useState(false);
  const [correctedGameData, setCorrectedGameData] = useState(gameData);

  // Sections timing
  useEffect(() => {
    const timings = [
      { delay: 500, section: 0, action: () => setShowWinner(true) }, // Winner announcement
      { delay: 4000, section: 1 }, // Poker table
      { delay: 6000, section: 2 }  // Hand comparison
    ];

    const timeouts = timings.map(({ delay, section, action }) =>
      setTimeout(() => {
        setCurrentSection(section);
        if (action) action();
      }, delay)
    );

    return () => timeouts.forEach(clearTimeout);
  }, []);

  // Handle manual corrections and re-evaluate winner
  const handleSaveCorrections = async (correctedCards) => {
    console.log('🔧 User corrections received:', correctedCards);

    // Group cards by position
    const player1Cards = correctedCards.filter(card => card.group === 'player1').map(c => c.card_name);
    const communityCards = correctedCards.filter(card => card.group === 'community').map(c => c.card_name);
    const player2Cards = correctedCards.filter(card => card.group === 'player2').map(c => c.card_name);

    console.log('📊 Grouped cards:', {
      player1: player1Cards,
      community: communityCards,
      player2: player2Cards
    });

    // Validate poker rules
    if (player1Cards.length !== 2 || player2Cards.length !== 2) {
      alert(`⚠️ Each player must have exactly 2 cards.\nPlayer 1: ${player1Cards.length} cards\nPlayer 2: ${player2Cards.length} cards`);
      return;
    }

    if (communityCards.length < 3 || communityCards.length > 5) {
      alert(`⚠️ Community cards must be 3, 4, or 5 cards (not ${communityCards.length})`);
      return;
    }

    try {
      // Call backend API to re-evaluate winner with corrected cards
      console.log('🔄 Re-evaluating winner with backend...');

      const response = await fetch(`${API_URL}/evaluate-winner`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          player1_cards: player1Cards,
          community_cards: communityCards,
          player2_cards: player2Cards
        })
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const winnerData = await response.json();
      console.log('✅ Winner re-evaluated:', winnerData);

      // Update game data with corrected cards and new winner
      const updatedGameData = {
        ...gameData,
        cards: correctedCards,
        game_analysis: winnerData.game_analysis
      };

      console.log('🎉 Updated game data:', updatedGameData);
      setCorrectedGameData(updatedGameData);
      setShowCorrectionModal(false);

      // Show success message
      alert('✅ Winner re-evaluated successfully!');

    } catch (error) {
      console.error('❌ Failed to re-evaluate winner:', error);
      alert('⚠️ Failed to re-evaluate winner. Please try again.');
    }
  };

  // Get detected cards in the format expected by CardCorrectionModal
  const getDetectedCardsForModal = () => {
    // Cards can be in either gameData.cards or gameData.detection_results
    const allCards = gameData?.detection_results || gameData?.cards || [];
    console.log('🔍 Getting detected cards for modal. Raw cards:', allCards);
    console.log('🔍 gameData structure:', gameData);
    const formattedCards = allCards.map((card, index) => ({
      id: `card-${index}`,
      card_name: card.card || card.card_name,
      confidence: card.confidence,
      bbox: card.bbox,
      center: card.center,
      group: determineCardGroup(card)
    }));
    console.log('✅ Formatted cards for modal:', formattedCards);
    return formattedCards;
  };

  const determineCardGroup = (card) => {
    // Logic to determine which group the card belongs to based on game analysis
    const gameAnalysis = correctedGameData?.game_analysis || gameData?.game_analysis;
    if (!gameAnalysis) return 'community';

    const cardName = card.card || card.card_name;

    // Check if card belongs to Player 1 (first player in the array)
    if (gameAnalysis.players?.[0]?.hole_cards?.includes(cardName)) {
      return 'player1';
    }

    // Check if card belongs to Player 2 (second player in the array)
    if (gameAnalysis.players?.[1]?.hole_cards?.includes(cardName)) {
      return 'player2';
    }

    // Check if card belongs to community cards
    if (gameAnalysis.community_cards?.includes(cardName)) {
      return 'community';
    }

    // Default based on position if not found in analysis
    if (card.center) {
      const yRatio = card.center[1] / 800; // Assuming image height of 800
      if (yRatio < 0.33) return 'player1';
      if (yRatio > 0.66) return 'player2';
    }

    return 'community';
  };

  const shareResults = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'PokerVision Results',
          text: `${correctedGameData?.game_analysis?.winner?.name} wins with ${correctedGameData?.game_analysis?.winner?.winning_hand}!`,
          url: window.location.href
        });
      } catch (err) {
        console.log('Error sharing:', err);
      }
    } else {
      // Fallback: copy to clipboard
      const text = `PokerVision Results: ${gameData?.game_analysis?.winner?.name} wins with ${gameData?.game_analysis?.winner?.winning_hand}!`;
      navigator.clipboard.writeText(text);
      // You could show a toast here
    }
  };

  // Debug log to see what we're getting
  console.log('PokerResultsPage gameData:', gameData);

  if (!gameData) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading results...</div>
      </div>
    );
  }

  // Check if we have game analysis - if not, show simple results
  if (!gameData.game_analysis) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center text-white max-w-md">
          <h1 className="text-2xl mb-4">Detection Complete</h1>
          <p className="mb-4">
            Cards: {gameData.cards_detected ? gameData.cards_detected.join(', ') : 'None detected'}
          </p>
          <button 
            onClick={onBack}
            className="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded text-white"
          >
            Back to Upload
          </button>
        </div>
      </div>
    );
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.3 }
    }
  };

  const sectionVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.8, ease: "easeOut" }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black overflow-hidden">
      {/* Background Pattern */}
      <div className="fixed inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 25% 25%, rgba(255,215,0,0.1) 0%, transparent 50%),
                           radial-gradient(circle at 75% 75%, rgba(34,197,94,0.1) 0%, transparent 50%)`
        }} />
      </div>

      {/* Header Controls */}
      <motion.div 
        className="relative z-50 p-4 md:p-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="max-w-6xl mx-auto">
          {/* Mobile Layout */}
          <div className="flex flex-col gap-4 md:hidden">
            <button
              onClick={onBack}
              className="flex items-center gap-2 px-4 py-3 bg-gray-800/50 hover:bg-gray-700/50 
                         text-white rounded-lg transition-colors backdrop-blur-sm border border-gray-600/30
                         justify-center font-medium"
            >
              <ArrowLeft size={20} />
              Back to Upload
            </button>

            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setShowCorrectionModal(true)}
                className="flex items-center justify-center gap-2 p-3 bg-orange-600/80 hover:bg-orange-500/80 
                           text-white rounded-lg transition-colors backdrop-blur-sm text-sm"
              >
                <Edit3 size={18} />
                <span className="text-xs">Fix</span>
              </button>

              <button
                onClick={shareResults}
                className="flex items-center justify-center gap-2 p-3 bg-blue-600/80 hover:bg-blue-500/80 
                           text-white rounded-lg transition-colors backdrop-blur-sm text-sm"
              >
                <Share2 size={18} />
                <span className="text-xs">Share</span>
              </button>

              <button
                onClick={onAnalyzeAnother}
                className="flex items-center justify-center gap-2 p-3 bg-green-600/80 hover:bg-green-500/80 
                           text-white rounded-lg transition-colors backdrop-blur-sm text-sm"
              >
                <RotateCcw size={18} />
                <span className="text-xs">Again</span>
              </button>
            </div>
          </div>

          {/* Desktop Layout */}
          <div className="hidden md:flex justify-between items-center">
            <button
              onClick={onBack}
              className="flex items-center gap-2 px-4 py-2 bg-gray-800/50 hover:bg-gray-700/50 
                         text-white rounded-lg transition-colors backdrop-blur-sm border border-gray-600/30"
            >
              <ArrowLeft size={20} />
              Back to Upload
            </button>

            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowCorrectionModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-orange-600/80 hover:bg-orange-500/80 
                           text-white rounded-lg transition-colors backdrop-blur-sm"
              >
                <Edit3 size={20} />
                Fix Detection
              </button>

              <button
                onClick={shareResults}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600/80 hover:bg-blue-500/80 
                           text-white rounded-lg transition-colors backdrop-blur-sm"
              >
                <Share2 size={20} />
                Share
              </button>

              <button
                onClick={onAnalyzeAnother}
                className="flex items-center gap-2 px-4 py-2 bg-green-600/80 hover:bg-green-500/80 
                           text-white rounded-lg transition-colors backdrop-blur-sm"
              >
                <RotateCcw size={20} />
                Analyze Another
              </button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main Content */}
      <motion.div
        className="relative z-10 px-4 md:px-6 pb-8 md:pb-12"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <div className="max-w-6xl mx-auto space-y-8 md:space-y-12">
          
          {/* Section 1: Winner Spotlight or Fix Detection Prompt */}
          <motion.section
            variants={sectionVariants}
            className="min-h-[50vh] flex items-center justify-center"
          >
            {correctedGameData.game_analysis.winner || correctedGameData.game_analysis.tie ? (
              <WinnerAnnouncement
                winner={correctedGameData.game_analysis.winner}
                isTie={correctedGameData.game_analysis.tie}
                tiedPlayers={correctedGameData.game_analysis.tied_players || []}
                isVisible={showWinner}
              />
            ) : (
              /* Fallback: No winner detected - prompt user to fix detection */
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.8, type: "spring", bounce: 0.3 }}
                className="max-w-2xl mx-auto bg-gradient-to-br from-orange-500/20 via-yellow-600/20 to-orange-500/20
                           backdrop-blur-lg rounded-3xl p-8 md:p-12 border-2 border-orange-400/40
                           shadow-2xl shadow-orange-400/20"
              >
                {/* Alert Icon */}
                <div className="flex justify-center mb-6">
                  <motion.div
                    animate={{
                      rotate: [0, -10, 10, -10, 0],
                      scale: [1, 1.1, 1]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                    className="w-20 h-20 md:w-24 md:h-24 bg-orange-500/30 rounded-full flex items-center justify-center border-2 border-orange-400/50"
                  >
                    <Edit3 size={40} className="text-orange-300" />
                  </motion.div>
                </div>

                {/* Title */}
                <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 bg-gradient-to-r from-orange-300 via-yellow-300 to-orange-300 bg-clip-text text-transparent">
                  Incomplete Detection
                </h2>

                {/* Description */}
                <p className="text-white/90 text-center text-base md:text-lg mb-6 leading-relaxed">
                  I couldn't detect all 9 cards needed to determine a winner.
                  Don't worry - you can easily add the missing cards manually!
                </p>

                {/* Cards Count Info */}
                <div className="bg-black/30 backdrop-blur-sm rounded-xl p-4 mb-6 border border-orange-400/20">
                  <div className="flex items-center justify-center gap-3 text-white/80">
                    <span className="text-2xl font-bold text-orange-300">
                      {correctedGameData?.detection_results?.length || correctedGameData?.cards?.length || 0}
                    </span>
                    <span className="text-lg">/</span>
                    <span className="text-lg text-white/60">9 cards detected</span>
                  </div>
                </div>

                {/* Fix Detection Button */}
                <motion.button
                  onClick={() => setShowCorrectionModal(true)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-orange-500 to-yellow-500
                             hover:from-orange-600 hover:to-yellow-600 text-white font-bold text-lg rounded-xl
                             transition-all shadow-lg shadow-orange-500/30 border border-orange-400/50"
                >
                  <Edit3 size={24} />
                  Fix Detection & See Winner
                </motion.button>

                {/* Help Text */}
                <p className="text-white/60 text-center text-sm mt-4">
                  Click to add missing cards and determine the winner
                </p>
              </motion.div>
            )}
          </motion.section>

          {/* Section 2: Poker Table View */}
          {currentSection >= 1 && (
            <motion.section
              variants={sectionVariants}
              className="py-8"
            >
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="text-center mb-8"
              >
                <h2 className="text-4xl font-bold text-white mb-4">
                  Game Analysis
                </h2>
                <p className="text-gray-300 text-lg max-w-2xl mx-auto">
                  See how the cards were distributed{correctedGameData.game_analysis.tie ? ' and why it was a tie' : ' and why '}
                  {!correctedGameData.game_analysis.tie && (
                    <span className="text-yellow-400 font-semibold ml-1">
                      {correctedGameData.game_analysis.winner?.name}
                    </span>
                  )}
                  {!correctedGameData.game_analysis.tie && ' won'}
                </p>
              </motion.div>

              <PokerTableView 
                gameAnalysis={correctedGameData.game_analysis}
                showAnimation={currentSection >= 1}
              />
            </motion.section>
          )}

          {/* Section 3: Hand Comparison */}
          {currentSection >= 2 && (
            <motion.section
              variants={sectionVariants}
              className="py-8"
            >
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="text-center mb-8"
              >
                <h2 className="text-4xl font-bold text-white mb-4">
                  Detailed Hand Comparison
                </h2>
                <p className="text-gray-300 text-lg max-w-2xl mx-auto">
                  Compare the best possible hands each player could make
                </p>
              </motion.div>

              <HandComparisonPanel 
                gameAnalysis={correctedGameData.game_analysis}
                showAnimation={currentSection >= 2}
              />
            </motion.section>
          )}

          {/* App Purpose & Photo Tips */}
          <motion.section
            variants={sectionVariants}
            className="py-4 md:py-8"
          >
            <div className="bg-gradient-to-br from-gray-800/50 via-gray-700/40 to-gray-800/50 
                            backdrop-blur-lg rounded-3xl p-6 md:p-8 border border-gray-600/30 
                            shadow-2xl relative overflow-hidden">
              
              {/* Background Decoration */}
              <div className="absolute top-0 right-0 opacity-5 text-8xl text-yellow-400">
                🃏
              </div>
              <div className="absolute bottom-0 left-0 opacity-5 text-6xl text-green-400">
                🏆
              </div>
              
              <div className="relative z-10">
                {/* Main Story */}
                <div className="text-center mb-8">
                  <motion.h2 
                    className="text-3xl md:text-4xl font-bold text-white mb-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    <span className="text-yellow-400">The End</span> of Your Poker Arguments! 
                  </motion.h2>
                  
                  <motion.div
                    className="max-w-4xl mx-auto space-y-4 text-gray-300 text-lg md:text-xl leading-relaxed"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    <p className="mb-4">
                      <span className="text-red-400 font-semibold">"I won!"</span> 
                      <span className="mx-4">vs</span>
                      <span className="text-blue-400 font-semibold">"No, I won!"</span>
                    </p>
                    
                    <p className="text-white font-medium">
                      Sound familiar? We've all been there - that heated moment when two players 
                      are absolutely convinced they have the winning hand. The argument gets louder, 
                      cards get mixed up, and nobody can agree on who actually won.
                    </p>
                    
                    <p>
                      <span className="text-yellow-400 font-bold">PokerVision</span> ends these disputes 
                      instantly! Just take a photo of your poker table, and our AI will analyze every 
                      card, evaluate both hands using official poker rules, and declare the winner
                      with <span className="text-green-400 font-semibold">AI-powered analysis</span>.
                    </p>
                  </motion.div>
                </div>

                {/* Photo Tips Section */}
                <motion.div
                  className="bg-black/30 backdrop-blur-sm rounded-2xl p-6 border border-yellow-400/30"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                >
                  <h3 className="text-2xl font-bold text-yellow-300 mb-4 text-center flex items-center justify-center gap-2">
                    📸 Perfect Photo Tips
                  </h3>
                  
                  <div className="grid md:grid-cols-2 gap-6 items-center">
                    {/* Tips Text */}
                    <div className="space-y-3">
                      <p className="text-white font-semibold text-lg">
                        For best results, arrange your cards like this:
                      </p>
                      <ul className="space-y-2 text-gray-300">
                        <li className="flex items-center gap-3">
                          <span className="text-green-400 text-xl">🔝</span>
                          <span><strong className="text-white">Top:</strong> Player 1's 2 hole cards</span>
                        </li>
                        <li className="flex items-center gap-3">
                          <span className="text-yellow-400 text-xl">🎯</span>
                          <span><strong className="text-white">Center:</strong> 5 community cards in a row</span>
                        </li>
                        <li className="flex items-center gap-3">
                          <span className="text-blue-400 text-xl">🔽</span>
                          <span><strong className="text-white">Bottom:</strong> Player 2's 2 hole cards</span>
                        </li>
                      </ul>
                      <p className="text-sm text-gray-400 italic mt-4">
                        💡 Keep cards clearly visible and well-lit for the best results!
                      </p>
                    </div>

                    {/* Visual Layout Guide */}
                    <div className="bg-gradient-to-br from-green-800/30 to-green-900/30 rounded-2xl p-4 border border-green-600/30">
                      <div className="text-center space-y-3">
                        {/* Player 1 */}
                        <div className="flex justify-center gap-2">
                          <div className="w-8 h-10 bg-green-400/80 rounded text-xs flex items-center justify-center text-black font-bold">
                            P1
                          </div>
                          <div className="w-8 h-10 bg-green-400/80 rounded text-xs flex items-center justify-center text-black font-bold">
                            P1
                          </div>
                        </div>
                        
                        {/* Community Cards */}
                        <div className="flex justify-center gap-1">
                          {[1,2,3,4,5].map(i => (
                            <div key={i} className="w-6 h-8 bg-yellow-400/80 rounded text-xs flex items-center justify-center text-black font-bold">
                              {i}
                            </div>
                          ))}
                        </div>
                        
                        {/* Player 2 */}
                        <div className="flex justify-center gap-2">
                          <div className="w-8 h-10 bg-blue-400/80 rounded text-xs flex items-center justify-center text-black font-bold">
                            P2
                          </div>
                          <div className="w-8 h-10 bg-blue-400/80 rounded text-xs flex items-center justify-center text-black font-bold">
                            P2
                          </div>
                        </div>
                        
                        <p className="text-xs text-gray-400 mt-2">
                          Ideal card layout for AI analysis
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  {/* Quick Stats */}
                  <div className="grid grid-cols-3 gap-4 mt-6 pt-4 border-t border-gray-600/30">
                    <div className="text-center">
                      <div className="text-xl font-bold text-blue-400">
                        {gameData.processing_time?.toFixed(1)}s
                      </div>
                      <div className="text-xs text-gray-400">Analysis Time</div>
                    </div>
                    <div className="text-center">
                      <div className="text-xl font-bold text-green-400">
                        {gameData.cards_detected?.length || 0}/9
                      </div>
                      <div className="text-xs text-gray-400">Cards Found</div>
                    </div>
                    <div className="text-center">
                      {/* TODO: Replace with a real, measured accuracy number once we have benchmark data. */}
                      <div className="text-xl font-bold text-purple-400">
                        AI
                      </div>
                      <div className="text-xs text-gray-400">Powered</div>
                    </div>
                  </div>
                </motion.div>
              </div>
            </div>
          </motion.section>

          {/* Call to Action */}
          <motion.section
            variants={sectionVariants}
            className="text-center py-12"
          >
            <motion.div
              className="bg-gradient-to-br from-gray-800/50 via-gray-700/50 to-gray-800/50 
                         backdrop-blur-lg rounded-3xl p-12 border border-gray-600/30"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", damping: 10 }}
            >
              <h3 className="text-3xl font-bold text-white mb-4">
                Ready for Another Game?
              </h3>
              <p className="text-gray-300 mb-8 max-w-md mx-auto">
                Upload another poker image to instantly see who wins
              </p>
              
              <button
                onClick={onAnalyzeAnother}
                className="px-8 py-4 bg-gradient-to-r from-green-500 to-blue-500 
                           hover:from-green-400 hover:to-blue-400 text-white font-bold 
                           text-lg rounded-2xl transition-all transform hover:scale-105 
                           shadow-lg shadow-green-500/25"
              >
                Analyze Another Image
              </button>
            </motion.div>
          </motion.section>

        </div>
      </motion.div>

      {/* Card Correction Modal */}
      <CardCorrectionModal
        isOpen={showCorrectionModal}
        originalImage={originalImage}
        detectedCards={getDetectedCardsForModal()}
        onSave={handleSaveCorrections}
        onCancel={() => setShowCorrectionModal(false)}
      />
    </div>
  );
};

export default PokerResultsPage;