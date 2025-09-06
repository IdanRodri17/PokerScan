import React from 'react';
import { motion } from 'framer-motion';
import PlayingCard from './PlayingCard';

const PokerTableView = ({ gameAnalysis, showAnimation = true }) => {
  if (!gameAnalysis) return null;

  const { community_cards = [], players = [], winner } = gameAnalysis;

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
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
      {/* Poker Table Background */}
      <div className="relative bg-gradient-to-br from-green-800 via-green-700 to-green-900 
                      rounded-2xl md:rounded-3xl p-4 md:p-8 shadow-2xl border-4 md:border-8 border-amber-600/30
                      backdrop-blur-sm">
        
        {/* Table Felt Texture Overlay */}
        <div className="absolute inset-0 opacity-10 rounded-3xl"
             style={{
               backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.15) 1px, transparent 0)`,
               backgroundSize: '20px 20px'
             }} />

        {/* Inner Table Border */}
        <div className="absolute inset-4 border-2 border-amber-600/20 rounded-2xl" />

        <div className="relative z-10 space-y-4 md:space-y-8">
          
          {/* Player 1 Area (Top) */}
          <motion.div 
            variants={itemVariants}
            className="flex flex-col items-center"
          >
            {players[0] && (
              <div className={`
                p-4 md:p-6 rounded-2xl backdrop-blur-sm border-2 transition-all duration-500
                ${winner?.name === players[0].name 
                  ? 'bg-gradient-to-br from-yellow-400/20 via-yellow-500/30 to-orange-500/20 border-yellow-400/50 shadow-lg shadow-yellow-400/25' 
                  : 'bg-black/20 border-gray-600/30'
                }
              `}>
                <h3 className={`text-lg md:text-xl font-bold mb-3 md:mb-4 text-center ${
                  winner?.name === players[0].name ? 'text-yellow-300' : 'text-white'
                }`}>
                  {players[0].name}
                  {winner?.name === players[0].name && (
                    <span className="ml-2 text-yellow-400">👑</span>
                  )}
                </h3>
                
                {/* Player 1 Cards */}
                <div className="flex gap-2 md:gap-3 justify-center">
                  {players[0].hole_cards?.map((card, index) => (
                    <PlayingCard
                      key={`p1-${index}`}
                      card={card}
                      size="md" // Smaller size for mobile
                      isHighlighted={winner?.name === players[0].name}
                      flipDelay={0.5 + index * 0.1}
                      showAnimation={showAnimation}
                    />
                  ))}
                </div>

                <div className="mt-3 text-center">
                  <p className={`text-sm ${
                    winner?.name === players[0].name ? 'text-yellow-200' : 'text-gray-300'
                  }`}>
                    {players[0].hand_description || players[0].best_hand}
                  </p>
                </div>
              </div>
            )}
          </motion.div>

          {/* Community Cards Area (Center) */}
          <motion.div 
            variants={itemVariants}
            className="flex flex-col items-center py-4 md:py-6"
          >
            <h3 className="text-xl md:text-2xl font-bold text-amber-300 mb-4 md:mb-6 text-center">
              Community Cards
            </h3>
            
            <div className="flex gap-2 md:gap-4 justify-center flex-wrap bg-black/20 p-4 md:p-6 rounded-2xl 
                            border-2 border-amber-600/30 backdrop-blur-sm">
              {community_cards.map((card, index) => (
                <PlayingCard
                  key={`community-${index}`}
                  card={card}
                  size="lg" // Smaller for mobile, still prominent
                  flipDelay={1.0 + index * 0.2}
                  showAnimation={showAnimation}
                />
              ))}
            </div>
          </motion.div>

          {/* Player 2 Area (Bottom) */}
          <motion.div 
            variants={itemVariants}
            className="flex flex-col items-center"
          >
            {players[1] && (
              <div className={`
                p-4 md:p-6 rounded-2xl backdrop-blur-sm border-2 transition-all duration-500
                ${winner?.name === players[1].name 
                  ? 'bg-gradient-to-br from-yellow-400/20 via-yellow-500/30 to-orange-500/20 border-yellow-400/50 shadow-lg shadow-yellow-400/25' 
                  : 'bg-black/20 border-gray-600/30'
                }
              `}>
                <h3 className={`text-lg md:text-xl font-bold mb-3 md:mb-4 text-center ${
                  winner?.name === players[1].name ? 'text-yellow-300' : 'text-white'
                }`}>
                  {players[1].name}
                  {winner?.name === players[1].name && (
                    <span className="ml-2 text-yellow-400">👑</span>
                  )}
                </h3>
                
                {/* Player 2 Cards */}
                <div className="flex gap-2 md:gap-3 justify-center">
                  {players[1].hole_cards?.map((card, index) => (
                    <PlayingCard
                      key={`p2-${index}`}
                      card={card}
                      size="md" // Smaller size for mobile
                      isHighlighted={winner?.name === players[1].name}
                      flipDelay={0.8 + index * 0.15} // Shorter delay for player 2
                      showAnimation={showAnimation}
                    />
                  ))}
                </div>

                <div className="mt-3 text-center">
                  <p className={`text-sm ${
                    winner?.name === players[1].name ? 'text-yellow-200' : 'text-gray-300'
                  }`}>
                    {players[1].hand_description || players[1].best_hand}
                  </p>
                </div>
              </div>
            )}
          </motion.div>

        </div>

        {/* Table Edge Lighting Effect */}
        <div className="absolute inset-0 rounded-3xl pointer-events-none"
             style={{
               boxShadow: 'inset 0 0 60px rgba(0,0,0,0.3), 0 0 40px rgba(139, 69, 19, 0.2)'
             }} />
      </div>
    </motion.div>
  );
};

export default PokerTableView;