import React from 'react';
import { motion } from 'framer-motion';
import { Camera, AlertTriangle, ArrowRight } from 'lucide-react';

/**
 * Shown when the backend's photo_quality gate flags an upload as likely too far /
 * messy to analyze reliably. Offers a friendly retake with specific guidance, plus
 * an "analyze anyway" escape hatch so the user is never blocked.
 */
const RetakePrompt = ({ photoQuality, originalImage, onRetake, onAnalyzeAnyway }) => {
  const reasons = photoQuality?.reasons || [];
  const cardsDetected = photoQuality?.cards_detected ?? 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black
                    flex items-center justify-center px-4 py-12">
      <motion.div
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, type: 'spring', bounce: 0.3 }}
        className="max-w-xl w-full bg-gradient-to-br from-orange-500/15 via-yellow-600/10 to-orange-500/15
                   backdrop-blur-lg rounded-3xl p-6 md:p-8 border-2 border-orange-400/40
                   shadow-2xl shadow-orange-400/10"
      >
        <div className="flex justify-center mb-5">
          <motion.div
            animate={{ rotate: [0, -8, 8, -8, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            className="w-20 h-20 bg-orange-500/25 rounded-full flex items-center justify-center
                       border-2 border-orange-400/50"
          >
            <AlertTriangle size={38} className="text-orange-300" />
          </motion.div>
        </div>

        <h2 className="text-2xl md:text-3xl font-bold text-center text-orange-200 mb-2">
          Let&apos;s retake that photo
        </h2>
        <p className="text-center text-white/80 mb-6">
          This photo looks hard to read, so the result would probably be wrong. A quick
          retake usually fixes it{cardsDetected ? ` (only ${cardsDetected} cards came through)` : ''}:
        </p>

        {originalImage && (
          <img
            src={originalImage}
            alt="Your upload"
            className="max-h-40 mx-auto rounded-xl mb-6 opacity-80 border border-white/10"
          />
        )}

        <ul className="space-y-3 mb-7">
          {reasons.map((reason, i) => (
            <li key={i} className="flex items-start gap-3 text-white/90">
              <Camera size={18} className="text-orange-300 mt-0.5 flex-shrink-0" />
              <span>{reason}</span>
            </li>
          ))}
        </ul>

        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={onRetake}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-4
                       bg-gradient-to-r from-orange-500 to-yellow-500
                       hover:from-orange-600 hover:to-yellow-600 text-white font-bold
                       rounded-xl transition-all shadow-lg shadow-orange-500/30"
          >
            <Camera size={20} /> Retake photo
          </button>
          <button
            onClick={onAnalyzeAnyway}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-4
                       bg-gray-700/60 hover:bg-gray-600/60 text-white/90 font-semibold
                       rounded-xl transition-all border border-gray-500/40"
          >
            Analyze anyway <ArrowRight size={18} />
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default RetakePrompt;
