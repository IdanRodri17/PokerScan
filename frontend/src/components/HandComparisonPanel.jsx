import React from 'react';
import { motion } from 'framer-motion';
import PlayingCard from './PlayingCard';
import { Trophy, TrendingUp, Star } from 'lucide-react';

const HandComparisonPanel = ({ gameAnalysis, showAnimation = true }) => {
  if (!gameAnalysis || !gameAnalysis.players) return null;

  const { players, winner, community_cards = [] } = gameAnalysis;

  // Hand strength mapping for visual comparison
  const handStrength = {
    'High Card': 1,
    'Pair': 2,
    'Two Pair': 3,
    'Three of a Kind': 4,
    'Straight': 5,
    'Flush': 6,
    'Full House': 7,
    'Four of a Kind': 8,
    'Straight Flush': 9,
    'Royal Flush': 10
  };

  const getHandStrength = (handType) => {
    return handStrength[handType] || 1;
  };

  const getStrengthPercentage = (handType) => {
    return (getHandStrength(handType) / 10) * 100;
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        staggerChildren: 0.2,
        duration: 0.6
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { 
      opacity: 1, 
      x: 0,
      transition: { type: "spring", damping: 12 }
    }
  };

  return (
    <motion.div
      className="w-full max-w-6xl mx-auto"
      variants={showAnimation ? containerVariants : undefined}
      initial={showAnimation ? "hidden" : undefined}
      animate={showAnimation ? "visible" : undefined}
    >
      <div className="bg-gradient-to-br from-gray-900/50 via-gray-800/50 to-gray-900/50 
                      backdrop-blur-lg rounded-2xl md:rounded-3xl p-4 md:p-8 border border-gray-600/30 
                      shadow-2xl">
        
        {/* Header */}
        <motion.div 
          variants={itemVariants}
          className="text-center mb-6 md:mb-8"
        >
          <h2 className="text-2xl md:text-3xl font-bold text-white mb-2 flex items-center justify-center gap-2 md:gap-3">
            <TrendingUp className="text-blue-400" size={24} />
            <span className="hidden sm:inline">Hand Comparison</span>
            <span className="sm:hidden">Comparison</span>
            <TrendingUp className="text-blue-400" size={24} />
          </h2>
          <p className="text-gray-300 text-sm md:text-base">Best possible 5-card hands</p>
        </motion.div>

        {/* Players Comparison */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-8">
          {players.map((player, index) => {
            const isWinner = winner?.name === player.name;
            const strength = getStrengthPercentage(player.best_hand);
            
            return (
              <motion.div
                key={player.name}
                variants={itemVariants}
                className={`
                  relative p-4 md:p-6 rounded-2xl border-2 transition-all duration-500
                  ${isWinner 
                    ? 'bg-gradient-to-br from-yellow-400/10 via-yellow-500/20 to-orange-500/10 border-yellow-400/50 shadow-lg shadow-yellow-400/20' 
                    : 'bg-gradient-to-br from-gray-700/20 to-gray-800/30 border-gray-600/30'
                  }
                `}
                style={{
                  transform: isWinner ? 'scale(1.02)' : 'scale(1)',
                }}
              >
                {/* Winner Badge */}
                {isWinner && (
                  <motion.div
                    className="absolute -top-3 left-1/2 transform -translate-x-1/2"
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ delay: 1, type: "spring", damping: 10 }}
                  >
                    <div className="bg-gradient-to-r from-yellow-400 to-orange-500 
                                    px-4 py-1 rounded-full flex items-center gap-2">
                      <Trophy size={16} className="text-white" />
                      <span className="text-white font-bold text-sm">WINNER</span>
                    </div>
                  </motion.div>
                )}

                {/* Player Header */}
                <div className="flex items-center justify-between mb-6">
                  <h3 className={`text-xl font-bold ${
                    isWinner ? 'text-yellow-300' : 'text-white'
                  }`}>
                    {player.name}
                  </h3>
                  {isWinner && (
                    <motion.div
                      animate={{ rotate: [0, 10, -10, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      <Star className="text-yellow-400" size={24} fill="currentColor" />
                    </motion.div>
                  )}
                </div>

                {/* Player's Hole Cards */}
                <div className="mb-6">
                  <p className="text-gray-300 text-sm mb-3">Hole Cards</p>
                  <div className="flex gap-2 justify-center">
                    {player.hole_cards?.map((card, cardIndex) => (
                      <PlayingCard
                        key={`${player.name}-hole-${cardIndex}`}
                        card={card}
                        size="md"
                        isHighlighted={isWinner}
                        flipDelay={0} // Remove delay for hand comparison
                        showAnimation={false} // Disable animation in comparison to ensure visibility
                      />
                    ))}
                  </div>
                </div>

                {/* Best Hand Badge */}
                <div className="mb-4">
                  <div className={`
                    inline-block px-4 py-2 rounded-full text-sm font-bold
                    ${isWinner 
                      ? 'bg-gradient-to-r from-yellow-400 to-orange-500 text-white' 
                      : 'bg-gray-600/50 text-gray-200 border border-gray-500/30'
                    }
                  `}>
                    {player.best_hand?.toUpperCase()}
                  </div>
                </div>

                {/* Hand Description */}
                <p className={`text-lg font-semibold mb-4 ${
                  isWinner ? 'text-yellow-200' : 'text-gray-200'
                }`}>
                  {player.hand_description}
                </p>

                {/* Hand Strength Bar */}
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-300">Hand Strength</span>
                    <span className={isWinner ? 'text-yellow-300' : 'text-gray-300'}>
                      {strength.toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700/50 rounded-full h-3 overflow-hidden">
                    <motion.div
                      className={`h-full rounded-full ${
                        isWinner 
                          ? 'bg-gradient-to-r from-yellow-400 to-orange-500' 
                          : 'bg-gradient-to-r from-gray-500 to-gray-400'
                      }`}
                      initial={{ width: 0 }}
                      animate={{ width: `${strength}%` }}
                      transition={{ delay: 1 + index * 0.2, duration: 1.5, ease: "easeOut" }}
                    />
                  </div>
                </div>

                {/* Winning Glow Effect */}
                {isWinner && (
                  <motion.div
                    className="absolute inset-0 border-2 border-yellow-400/30 rounded-2xl pointer-events-none"
                    animate={{
                      boxShadow: [
                        "0 0 20px rgba(255,215,0,0.2)",
                        "0 0 40px rgba(255,215,0,0.4)",
                        "0 0 20px rgba(255,215,0,0.2)"
                      ]
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Community Cards Reference */}
        <motion.div 
          variants={itemVariants}
          className="mt-8 p-4 bg-black/20 rounded-2xl border border-gray-600/30"
        >
          <p className="text-gray-300 text-sm mb-3 text-center">
            Community Cards (available to both players)
          </p>
          <div className="flex gap-2 justify-center flex-wrap">
            {community_cards.map((card, index) => (
              <PlayingCard
                key={`community-ref-${index}`}
                card={card}
                size="sm"
                flipDelay={0}
                showAnimation={false} // Disable animation for community cards reference
              />
            ))}
          </div>
        </motion.div>

      </div>
    </motion.div>
  );
};

export default HandComparisonPanel;