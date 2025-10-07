import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import Confetti from 'react-confetti';
import { Trophy, Star, Crown } from 'lucide-react';

const WinnerAnnouncement = ({ winner, isTie = false, tiedPlayers = [], isVisible = false }) => {
  const [showConfetti, setShowConfetti] = useState(false);
  const [windowDimension, setWindowDimension] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  const detectSize = () => {
    setWindowDimension({
      width: window.innerWidth,
      height: window.innerHeight,
    });
  };

  useEffect(() => {
    window.addEventListener('resize', detectSize);
    if (isVisible) {
      // Delay confetti start for dramatic effect
      setTimeout(() => setShowConfetti(true), 1500);
      // Stop confetti after 5 seconds
      setTimeout(() => setShowConfetti(false), 6500);
    }
    return () => {
      window.removeEventListener('resize', detectSize);
    };
  }, [isVisible]);

  // Show tie announcement if it's a tie
  if (isTie && tiedPlayers.length > 0) {
    return (
      <div className="relative overflow-hidden">
        {/* Confetti Effect */}
        {showConfetti && (
          <Confetti
            width={windowDimension.width}
            height={windowDimension.height}
            recycle={false}
            numberOfPieces={200}
            colors={['#60A5FA', '#34D399', '#A78BFA', '#FBBF24', '#F87171']}
          />
        )}

        {/* Tie Announcement Hero */}
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={isVisible ? { scale: 1, opacity: 1 } : { scale: 0, opacity: 0 }}
          transition={{
            duration: 1.5,
            type: "spring",
            bounce: 0.4,
            delay: 0.5
          }}
          className="relative bg-gradient-to-br from-blue-400/20 via-purple-600/30 to-blue-500/20
                     backdrop-blur-lg rounded-3xl p-12 border border-blue-400/30
                     shadow-2xl shadow-blue-400/25"
        >
          {/* Background Glow Effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-400/10 via-transparent to-purple-400/10
                          rounded-3xl blur-xl animate-pulse" />

          {/* Main Content */}
          <div className="relative z-10 text-center">
            {/* Handshake Icon */}
            <motion.div
              initial={{ y: -50, rotate: -10 }}
              animate={{ y: 0, rotate: 0 }}
              transition={{ delay: 1, duration: 1, type: "spring" }}
              className="flex justify-center mb-4 md:mb-6"
            >
              <div className="text-6xl md:text-8xl">
                🤝
              </div>
            </motion.div>

            {/* "IT'S A TIE!" Text */}
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2, duration: 0.8 }}
              className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black mb-3 md:mb-4"
            >
              <span className="bg-gradient-to-r from-blue-300 via-purple-400 to-blue-400
                             bg-clip-text text-transparent animate-pulse
                             drop-shadow-2xl filter px-4 py-2">
                IT'S A TIE!
              </span>
            </motion.h1>

            {/* Tied Players */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.5, duration: 0.6 }}
              className="mb-4 md:mb-6"
            >
              <motion.h2
                className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-2"
              >
                {tiedPlayers.map((p, idx) => (
                  <span key={p.id}>
                    {p.name}
                    {idx < tiedPlayers.length - 1 && ' vs '}
                  </span>
                ))}
              </motion.h2>
            </motion.div>

            {/* Same Hand */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.8, duration: 0.6 }}
              className="bg-black/30 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-blue-400/20"
            >
              <p className="text-blue-200 text-base md:text-lg mb-2 font-medium">Both Players Have</p>
              <p className="text-lg sm:text-xl md:text-2xl lg:text-3xl font-bold text-white">
                {tiedPlayers[0]?.hand || 'Same Hand'}
              </p>
            </motion.div>

            {/* Pulse Ring */}
            <motion.div
              className="absolute inset-0 border-4 border-blue-400/30 rounded-3xl pointer-events-none"
              animate={{
                scale: [1, 1.02, 1],
                borderColor: [
                  "rgba(96,165,250,0.3)",
                  "rgba(96,165,250,0.6)",
                  "rgba(96,165,250,0.3)"
                ]
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
          </div>
        </motion.div>
      </div>
    );
  }

  if (!winner) return null;

  return (
    <div className="relative overflow-hidden">
      {/* Confetti Effect */}
      {showConfetti && (
        <Confetti
          width={windowDimension.width}
          height={windowDimension.height}
          recycle={false}
          numberOfPieces={300}
          colors={['#FFD700', '#FFA500', '#FF6B35', '#4ECDC4', '#45B7D1']}
        />
      )}

      {/* Winner Announcement Hero */}
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={isVisible ? { scale: 1, opacity: 1 } : { scale: 0, opacity: 0 }}
        transition={{ 
          duration: 1.5, 
          type: "spring", 
          bounce: 0.4,
          delay: 0.5 
        }}
        className="relative bg-gradient-to-br from-yellow-400/20 via-yellow-600/30 to-orange-500/20 
                   backdrop-blur-lg rounded-3xl p-12 border border-yellow-400/30 
                   shadow-2xl shadow-yellow-400/25"
      >
        {/* Background Glow Effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-yellow-400/10 via-transparent to-yellow-400/10 
                        rounded-3xl blur-xl animate-pulse" />
        
        {/* Floating Stars */}
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(8)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute text-yellow-300"
              style={{
                left: `${10 + i * 12}%`,
                top: `${15 + (i % 3) * 20}%`,
              }}
              animate={{
                y: [-10, 10, -10],
                rotate: [0, 180, 360],
                scale: [0.8, 1.2, 0.8],
              }}
              transition={{
                duration: 3 + i * 0.5,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            >
              <Star size={16 + (i % 3) * 4} fill="currentColor" />
            </motion.div>
          ))}
        </div>

        {/* Main Content */}
        <div className="relative z-10 text-center">
          {/* Trophy Icon */}
          <motion.div
            initial={{ y: -50, rotate: -10 }}
            animate={{ y: 0, rotate: 0 }}
            transition={{ delay: 1, duration: 1, type: "spring" }}
            className="flex justify-center mb-4 md:mb-6"
          >
            <div className="relative p-4 md:p-6">
              <Trophy 
                size={80} // Mobile size
                className="text-yellow-400 drop-shadow-2xl filter md:w-[120px] md:h-[120px]" 
                fill="currentColor"
              />
              <motion.div
                className="absolute -top-1 -right-1 md:-top-2 md:-right-2"
                animate={{ rotate: [0, -15, 15, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Crown size={20} className="text-yellow-300 md:w-8 md:h-8" fill="currentColor" />
              </motion.div>
            </div>
          </motion.div>

          {/* "WINNER!" Text */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.8 }}
            className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black mb-3 md:mb-4"
          >
            <span className="bg-gradient-to-r from-yellow-300 via-yellow-400 to-orange-400 
                           bg-clip-text text-transparent animate-pulse 
                           drop-shadow-2xl filter px-4 py-2">
              WINNER!
            </span>
          </motion.h1>

          {/* Player Name */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1.5, duration: 0.6 }}
            className="mb-4 md:mb-6"
          >
            <motion.h2
              animate={{ 
                scale: [1, 1.05, 1],
                textShadow: [
                  "0 0 20px rgba(255,215,0,0.5)",
                  "0 0 30px rgba(255,215,0,0.8)", 
                  "0 0 20px rgba(255,215,0,0.5)"
                ]
              }}
              transition={{ 
                duration: 2, 
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-2"
            >
              {winner.name}
            </motion.h2>
          </motion.div>

          {/* Winning Hand */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.8, duration: 0.6 }}
            className="bg-black/30 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-yellow-400/20"
          >
            <p className="text-yellow-200 text-base md:text-lg mb-2 font-medium">Winning Hand</p>
            <p className="text-lg sm:text-xl md:text-2xl lg:text-3xl font-bold text-white">
              {winner.winning_hand}
            </p>
          </motion.div>

          {/* Victory Pulse Ring */}
          <motion.div
            className="absolute inset-0 border-4 border-yellow-400/30 rounded-3xl pointer-events-none"
            animate={{ 
              scale: [1, 1.02, 1],
              borderColor: [
                "rgba(255,215,0,0.3)",
                "rgba(255,215,0,0.6)", 
                "rgba(255,215,0,0.3)"
              ]
            }}
            transition={{ 
              duration: 3, 
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        </div>
      </motion.div>
    </div>
  );
};

export default WinnerAnnouncement;