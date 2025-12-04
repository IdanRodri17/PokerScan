# 🚀 Deployment Instructions

## Summary of Changes

✅ **Fixed duplicate card detection bug**
✅ **Fixed community card selection logic** (now uses confidence instead of X position)
✅ **All changes are committed and ready to push**

---

## Step 1: Push to Hugging Face Spaces (Backend Deployment)

The `backend/PokerVision-backend` folder is already committed with the fixes.

### Commands:

```bash
# Navigate to PokerVision-backend folder
cd backend/PokerVision-backend

# Check status
git status
# Should show: "Your branch is ahead of 'origin/main' by 1 commit"

# Push to Hugging Face
git push origin main
```

**When prompted for credentials:**
- Username: `Rodri17` (or your Hugging Face username)
- Password: Use your **Hugging Face Access Token** (not your password!)
  - Get token at: https://huggingface.co/settings/tokens

**After pushing:**
- Hugging Face Spaces will automatically rebuild (takes ~2-3 minutes)
- Check build status at: https://huggingface.co/spaces/Rodri17/PokerVision-backend
- Once deployed, test the API at: https://rodri17-pokervision-backend.hf.space/docs

---

## Step 2: Push to GitHub (Main Repository)

The main repository changes are also committed and ready.

### Commands:

```bash
# Navigate back to main folder
cd /mnt/c/Users/zionn/Desktop/PokerScan

# Check status
git status
# Should show: "Your branch is ahead of 'origin/main' by 1 commit"

# Push to GitHub
git push origin main
```

**When prompted for credentials:**
- Username: `IdanRodri17` (your GitHub username)
- Password: Use your **GitHub Personal Access Token** (not your password!)
  - Create token at: https://github.com/settings/tokens
  - Required scopes: `repo` (full control of private repositories)

**After pushing:**
- Changes will appear on GitHub immediately
- Repository: https://github.com/IdanRodri17/PokerScan

---

## Alternative: Using Git Credential Helper

If you don't want to enter credentials every time:

```bash
# Store credentials (careful - stores in plain text)
git config --global credential.helper store

# Or use cache (stores for 15 minutes)
git config --global credential.helper cache
```

---

## Step 3: Verify Deployment

### Test Backend (Hugging Face)

1. Go to: https://rodri17-pokervision-backend.hf.space/docs
2. Try the `/health` endpoint
3. Upload a test poker image via `/upload`
4. Verify:
   - Shows **9/9 cards** (not 10/9)
   - Community cards have correct cards (3c, 6h) not wrong ones (Qd, 3s)
   - Player 2 shows **2 cards** (not 3)

### Test Frontend (Netlify)

Frontend will automatically use the updated backend API.

1. Go to: https://pokervision.netlify.app
2. Upload the same test image
3. Verify the same fixes work on production

---

## What Was Fixed

### 1. Duplicate Rank Removal
- Players can't have 2 cards with same rank (e.g., 2H + 2S)
- Keeps highest confidence card when duplicates found
- Example: If Player 2 has 2H (89%) and 2S (80%), keeps only 2H

### 2. Community Card Selection
- **Old logic**: Kept "5 most centered by X position" ❌
- **New logic**: Keeps "5 highest confidence" ✅
- This fixes the bug where correct cards (3c 90%, 6h 87%) were removed and wrong cards (Qd 73%, 3s 84%) were kept

### 3. Detection Filtering
- Backend now returns only cards accepted by game analyzer
- Filters out duplicates that were removed
- Frontend receives exactly 9 cards for a valid game

---

## Files Modified

### Backend Files:
1. **`backend/ml/poker_game_analyzer.py`**
   - Lines 176-216: Duplicate rank removal
   - Lines 247-260: Confidence-based community card selection

2. **`backend/services/image_processor.py`**
   - Lines 257-306: Filter results to match game analyzer
   - Lines 315-337: Helper method to determine card groups

3. **`DUPLICATE_CARD_FIX_SUMMARY.md`**
   - Complete technical documentation

### Copied to Hugging Face:
- `backend/PokerVision-backend/ml/poker_game_analyzer.py` ✅
- `backend/PokerVision-backend/services/image_processor.py` ✅

---

## Commit Messages

Both repositories have identical commit messages:

```
Fix duplicate card detection and community card selection

Critical fixes:
1. Added duplicate rank removal for player zones
2. Fixed community card selection - uses confidence instead of X position
3. Filter detection results to match game analyzer output

Fixes: 10/9 cards issue, Player 2 showing 3 cards, wrong community cards

🤖 Generated with Claude Code
```

---

## Troubleshooting

### "fatal: could not read Username"
→ Push requires manual authentication in terminal

### "Authentication failed"
→ Use Personal Access Token, not password

### Hugging Face build fails
→ Check build logs at: https://huggingface.co/spaces/Rodri17/PokerVision-backend/settings

### Frontend still shows old behavior
→ Clear browser cache or wait 5 minutes for Netlify CDN to update

---

## Need Help?

Check the detailed technical docs in `DUPLICATE_CARD_FIX_SUMMARY.md`

---

**Status:** ✅ Ready to deploy!

All changes are committed locally. Just need to push to remotes with authentication.
