import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  Zap, 
  Eye, 
  Trophy, 
  Image as ImageIcon, 
  CheckCircle,
  ArrowRight,
  Sparkles,
  Target,
  Clock
} from 'lucide-react';

const PokerLandingPage = ({ onImageUpload, isProcessing }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1
  });

  const handleAnalyze = () => {
    if (selectedFile && onImageUpload) {
      onImageUpload(selectedFile);
    }
  };


  const features = [
    {
      icon: <Upload className="w-8 h-8" />,
      title: 'Upload Image',
      description: 'Drop your poker table photo',
      color: 'text-blue-400'
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: 'AI Analysis',
      description: 'Advanced card detection & hand evaluation',
      color: 'text-yellow-400'
    },
    {
      icon: <Trophy className="w-8 h-8" />,
      title: 'See Winner',
      description: 'Instant results with winning hand',
      color: 'text-green-400'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black overflow-hidden">
      {/* Background Effects */}
      <div className="fixed inset-0 opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 20% 30%, rgba(34,197,94,0.2) 0%, transparent 50%),
                           radial-gradient(circle at 80% 70%, rgba(59,130,246,0.2) 0%, transparent 50%),
                           radial-gradient(circle at 50% 50%, rgba(255,215,0,0.1) 0%, transparent 70%)`
        }} />
      </div>

      {/* Animated Card Suits Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        {['♠', '♥', '♦', '♣'].map((suit, index) => (
          <motion.div
            key={suit}
            className={`absolute text-6xl opacity-5 ${
              index % 2 === 0 ? 'text-red-500' : 'text-white'
            }`}
            style={{
              left: `${10 + index * 25}%`,
              top: `${20 + index * 15}%`,
            }}
            animate={{
              y: [-20, 20, -20],
              rotate: [0, 360],
              scale: [0.8, 1.2, 0.8],
            }}
            transition={{
              duration: 8 + index * 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          >
            {suit}
          </motion.div>
        ))}
      </div>

      <div className="relative z-10 px-6 py-12">
        <div className="max-w-6xl mx-auto">
          
          {/* Hero Section */}
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
          >
            <motion.div
              className="inline-flex items-center gap-3 mb-6"
              whileHover={{ scale: 1.05 }}
            >
              <div className="p-3 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-2xl">
                <Eye className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-5xl md:text-7xl font-black">
                <span className="bg-gradient-to-r from-yellow-400 via-yellow-300 to-orange-400 
                               bg-clip-text text-transparent">
                  PokerVision
                </span>
              </h1>
            </motion.div>
            
            <motion.h2 
              className="text-2xl md:text-3xl text-gray-300 mb-6 font-light"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 0.8 }}
            >
              Instant Winner Detection
            </motion.h2>

            <motion.p 
              className="text-xl text-gray-400 max-w-2xl mx-auto mb-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7, duration: 0.8 }}
            >
              Upload any poker table image and instantly see who wins with AI-powered 
              card detection and hand analysis
            </motion.p>

            <motion.div
              className="flex flex-wrap justify-center gap-4 text-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1, duration: 0.6 }}
            >
              {['100% Accurate', 'Instant Results', 'All Poker Variants'].map((feature, index) => (
                <div key={feature} className="flex items-center gap-2 text-green-400">
                  <CheckCircle size={16} />
                  <span>{feature}</span>
                </div>
              ))}
            </motion.div>
          </motion.div>

          {/* Upload Section */}
          <motion.div 
            className="mb-16"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
          >
            <div className="max-w-2xl mx-auto">
              
              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`
                  relative border-3 border-dashed rounded-3xl p-12 text-center cursor-pointer
                  transition-all duration-300 backdrop-blur-sm
                  ${isDragActive 
                    ? 'border-yellow-400 bg-yellow-400/10 scale-105' 
                    : 'border-gray-600 bg-gray-800/30 hover:border-gray-500 hover:bg-gray-800/50'
                  }
                  ${isProcessing ? 'pointer-events-none opacity-50' : ''}
                `}
              >
                <input {...getInputProps()} />
                
                {previewUrl ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-4"
                  >
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="max-h-64 mx-auto rounded-2xl shadow-2xl"
                    />
                    <p className="text-green-400 font-semibold">
                      Image ready for analysis!
                    </p>
                  </motion.div>
                ) : (
                  <div className="space-y-6">
                    <motion.div
                      animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
                      transition={{ type: "spring", damping: 10 }}
                    >
                      <div className="w-20 h-20 mx-auto bg-gradient-to-br from-blue-500 to-purple-500 
                                      rounded-2xl flex items-center justify-center mb-4">
                        <ImageIcon className="w-10 h-10 text-white" />
                      </div>
                    </motion.div>
                    
                    <div>
                      <p className="text-xl font-semibold text-white mb-2">
                        {isDragActive ? 'Drop your image here!' : 'Upload Poker Table Image'}
                      </p>
                      <p className="text-gray-400">
                        Drag & drop or click to select • JPG, PNG, WEBP
                      </p>
                    </div>
                  </div>
                )}

                {/* Upload Button */}
                {selectedFile && !isProcessing && (
                  <motion.button
                    onClick={handleAnalyze}
                    className="mt-6 px-8 py-4 bg-gradient-to-r from-green-500 to-blue-500 
                               hover:from-green-400 hover:to-blue-400 text-white font-bold 
                               text-lg rounded-2xl transition-all transform hover:scale-105 
                               shadow-lg shadow-green-500/25 flex items-center gap-3 mx-auto"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Sparkles className="w-5 h-5" />
                    Analyze & Find Winner
                    <ArrowRight className="w-5 h-5" />
                  </motion.button>
                )}

                {/* Processing State */}
                {isProcessing && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-6 flex items-center justify-center gap-3 text-blue-400"
                  >
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full"
                    />
                    <span className="font-semibold">Analyzing poker game...</span>
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>

          {/* How It Works */}
          <motion.section 
            className="mb-16"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
          >
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-white mb-4">How It Works</h2>
              <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                Advanced AI technology analyzes your poker table in seconds
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  className="relative p-8 bg-gray-800/50 backdrop-blur-sm rounded-3xl 
                             border border-gray-600/30 text-center group hover:bg-gray-700/50 
                             transition-all duration-300"
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7 + index * 0.2, duration: 0.6 }}
                  whileHover={{ y: -5 }}
                >
                  <div className={`inline-flex p-4 rounded-2xl mb-6 ${feature.color} bg-current/10`}>
                    <div className={feature.color}>
                      {feature.icon}
                    </div>
                  </div>
                  
                  <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
                  <p className="text-gray-400">{feature.description}</p>

                  {/* Step Number */}
                  <div className="absolute -top-4 -right-4 w-8 h-8 bg-gradient-to-br from-yellow-400 to-orange-500 
                                  rounded-full flex items-center justify-center text-white font-bold text-sm">
                    {index + 1}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.section>

          {/* App Purpose & Photo Tips */}
          <motion.section 
            className="mb-16"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9, duration: 0.8 }}
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
                    transition={{ delay: 1.1 }}
                  >
                    <span className="text-yellow-400">The End</span> of Your Poker Arguments! 
                  </motion.h2>
                  
                  <motion.div
                    className="max-w-4xl mx-auto space-y-4 text-gray-300 text-lg md:text-xl leading-relaxed"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 1.3 }}
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
                      with <span className="text-green-400 font-semibold">100% accuracy</span>.
                    </p>
                  </motion.div>
                </div>

                {/* Photo Tips Section */}
                <motion.div
                  className="bg-black/30 backdrop-blur-sm rounded-2xl p-6 border border-yellow-400/30"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1.5 }}
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
                        💡 Keep cards clearly visible and well-lit for instant, accurate results!
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
                </motion.div>
              </div>
            </div>
          </motion.section>


          {/* Features Grid */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.7, duration: 0.8 }}
          >
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { icon: <Target className="w-6 h-6" />, text: "100% Accuracy", color: "text-green-400" },
                { icon: <Clock className="w-6 h-6" />, text: "Instant Results", color: "text-blue-400" },
                { icon: <Sparkles className="w-6 h-6" />, text: "AI Powered", color: "text-purple-400" },
                { icon: <Trophy className="w-6 h-6" />, text: "Winner Detection", color: "text-yellow-400" }
              ].map((feature, index) => (
                <motion.div
                  key={feature.text}
                  className="flex items-center justify-center gap-3 p-4 bg-gray-800/30 
                             backdrop-blur-sm rounded-2xl border border-gray-600/20"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.5 + index * 0.1, duration: 0.4 }}
                >
                  <div className={feature.color}>{feature.icon}</div>
                  <span className="text-white font-semibold">{feature.text}</span>
                </motion.div>
              ))}
            </div>
          </motion.section>

        </div>
      </div>
    </div>
  );
};

export default PokerLandingPage;