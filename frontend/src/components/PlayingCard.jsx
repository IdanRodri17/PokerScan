import React from 'react';
import { motion } from 'framer-motion';

const PlayingCard = ({ 
  card, 
  size = 'md', 
  isHighlighted = false, 
  flipDelay = 0,
  showAnimation = true 
}) => {
  // Parse card string (e.g., "KS" -> King of Spades)
  const parseCard = (cardString) => {
    if (!cardString || cardString.length < 2) return { rank: '?', suit: '?', color: 'black' };
    
    const rank = cardString.slice(0, -1);
    const suitChar = cardString.slice(-1).toLowerCase();
    
    const suitMap = {
      'h': { symbol: '♥', name: 'hearts', color: 'red' },
      'd': { symbol: '♦', name: 'diamonds', color: 'red' },
      'c': { symbol: '♣', name: 'clubs', color: 'black' },
      's': { symbol: '♠', name: 'spades', color: 'black' }
    };
    
    const suit = suitMap[suitChar] || { symbol: '?', name: 'unknown', color: 'black' };
    
    return { rank, suit: suit.symbol, suitName: suit.name, color: suit.color };
  };

  const { rank, suit, color } = parseCard(card);

  // Size variants
  const sizeClasses = {
    sm: 'w-14 h-18 text-xs',
    md: 'w-16 h-24 text-sm',
    lg: 'w-20 h-28 text-base',
    xl: 'w-24 h-36 text-lg'
  };

  // Center suit symbol sizes for each card size
  const centerSuitSizes = {
    sm: 'text-lg',      // Much smaller for tiny cards
    md: 'text-2xl',     // Appropriate for medium cards
    lg: 'text-3xl',     // Good for large cards
    xl: 'text-4xl'      // Original size for extra large
  };

  // Corner text sizes for each card size
  const cornerTextSizes = {
    sm: 'text-xs',
    md: 'text-sm', 
    lg: 'text-base',
    xl: 'text-lg'
  };

  // Suit sizes for corners
  const cornerSuitSizes = {
    sm: 'text-xs',
    md: 'text-xs',
    lg: 'text-sm',
    xl: 'text-sm'
  };

  const cardVariants = {
    hidden: { 
      rotateY: -90, 
      scale: 0.8, 
      opacity: 0 
    },
    visible: { 
      rotateY: 0, 
      scale: 1, 
      opacity: 1,
      transition: {
        type: "spring",
        damping: 15,
        stiffness: 200,
        delay: flipDelay
      }
    },
    highlighted: {
      rotateY: 0, // Ensure card is not flipped
      scale: 1.05,
      y: -4,
      opacity: 1, // Ensure card is visible
      boxShadow: "0 10px 25px rgba(255,215,0,0.4), 0 0 20px rgba(255,215,0,0.3)",
      transition: {
        type: "spring",
        damping: 10,
        delay: flipDelay
      }
    }
  };

  return (
    <motion.div
      className="perspective-1000"
      variants={showAnimation ? cardVariants : undefined}
      initial={showAnimation ? "hidden" : { opacity: 1, rotateY: 0, scale: 1 }}
      animate={
        showAnimation 
          ? (isHighlighted ? "highlighted" : "visible")
          : (isHighlighted 
              ? { scale: 1.05, y: -4, opacity: 1, rotateY: 0, boxShadow: "0 10px 25px rgba(255,215,0,0.4), 0 0 20px rgba(255,215,0,0.3)" }
              : { opacity: 1, rotateY: 0, scale: 1 }
            )
      }
      whileHover={{ 
        scale: 1.05, 
        y: -2,
        transition: { type: "spring", damping: 10 }
      }}
    >
      <div className={`
        ${sizeClasses[size]}
        bg-white rounded-lg shadow-lg border-2 border-gray-300
        flex flex-col justify-between ${size === 'sm' ? 'p-0.5' : 'p-2'}
        font-bold relative overflow-hidden
        ${isHighlighted ? 'ring-2 ring-yellow-400 ring-opacity-75' : ''}
        transform-gpu backface-visibility-hidden
      `}>
        {/* Card Background Pattern */}
        <div className="absolute inset-0 opacity-5">
          <div className="w-full h-full bg-gradient-to-br from-gray-100 to-gray-200" />
        </div>

        {/* Top Left Corner */}
        <div className={`
          flex flex-col items-center leading-none relative z-10
          ${color === 'red' ? 'text-red-600' : 'text-gray-800'}
        `}>
          <div className={`text-current font-black ${cornerTextSizes[size]}`}>{rank}</div>
          <div className={`text-current ${cornerSuitSizes[size]}`}>{suit}</div>
        </div>


        {/* Bottom Right Corner (Rotated) */}
        <div className={`
          flex flex-col items-center leading-none transform rotate-180 relative z-10
          ${color === 'red' ? 'text-red-600' : 'text-gray-800'}
        `}>
          <div className={`text-current font-black ${cornerTextSizes[size]}`}>{rank}</div>
          <div className={`text-current ${cornerSuitSizes[size]}`}>{suit}</div>
        </div>

        {/* Highlight Glow Effect */}
        {isHighlighted && (
          <motion.div
            className="absolute inset-0 border-2 border-yellow-400 rounded-lg pointer-events-none"
            animate={{
              boxShadow: [
                "0 0 10px rgba(255,215,0,0.3)",
                "0 0 20px rgba(255,215,0,0.6)",
                "0 0 10px rgba(255,215,0,0.3)"
              ]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}
      </div>
    </motion.div>
  );
};

export default PlayingCard;