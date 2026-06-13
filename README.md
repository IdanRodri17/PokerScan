# 🃏 PokerVision — AI poker card detection & winner analysis

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://pokervision.netlify.app)
[![Backend API](https://img.shields.io/badge/API-Hugging%20Face-yellow)](https://rodri17-pokervision-backend.hf.space)

**PokerVision** reads a photo of a heads-up (2-player) Texas Hold'em table, detects
the cards with a fine-tuned **YOLOv11** model, works out which cards are community
vs. each player's hole cards, evaluates both hands, and names the winner — with a
manual-correction step for when detection isn't perfect.

> ⚠️ **Honest status:** this is a **demo / portfolio project**, not a production
> product. Detection is strong on clean datasets, but **accuracy on real phone
> photos is a work in progress** (see [Measured accuracy](#-measured-accuracy)).
> There is **no "100% accuracy"** claim here — the whole app is designed around the
> assumption that detection will sometimes be wrong, which is why manual correction
> is a first-class feature.

_Screenshot / demo GIF: TODO._

---

## ✨ What it does

- 📸 Upload a poker-table photo (including iPhone **HEIC**) → detected cards + winner.
- 🧠 Fine-tuned **YOLOv11** (Ultralytics) detector for all 52 playing cards.
- 🧩 **Adaptive grouping** sorts cards into player 1 / community / player 2 from the
  layout itself — no hard-coded screen regions.
- 🏆 Custom hand evaluator picks the best 5-of-7 and names the winner
  (Royal Flush → High Card), including ties and the A-2-3-4-5 "wheel".
- ✏️ **Manual correction** modal — fix any misread card and re-evaluate instantly.
- 📱 Web (React) + mobile shells (Capacitor for iOS / Android).

---

## 🎮 How it works (the real pipeline)

```
photo ─▶ /upload ─▶ YOLOv11 detection ─▶ de-dup ─▶ adaptive row grouping ─▶ hand evaluation ─▶ winner
                                                                              ▲
                                          user fixes a misread card ──▶ /evaluate-winner
```

1. **Upload** (`POST /upload`) — the image is validated (size limit, HEIC/EXIF
   handled) and decoded.
2. **Detection** — the image runs through the fine-tuned **YOLOv11** model
   (`conf=0.25`, `imgsz=1280`). Results are de-duplicated by card identity and by
   spatial proximity (a poker deck has unique cards), then the count is sanity-capped.
3. **Adaptive row grouping** (`ml/poker_game_analyzer.py`) — cards are clustered into
   up to 3 rows by the **vertical gaps** between them: the row nearest the middle with
   3–5 cards is the community, the row above is one player, the row below the other.
   This replaces the old fixed `20% / 50% / 80%` bands, and emits an internal
   `layout_confidence` score for how cleanly the rows separated.
4. **Hand evaluation** (`ml/hand_evaluator.py`) — a single evaluator (shared by both
   the photo path and the manual path, so they can't disagree) finds each player's
   best 5-card hand.
5. **Winner** — hand strengths are compared and the response returns the community
   cards, both players, and the winner (or a tie).
6. **Manual correction (fallback)** — if the detection is off, the user fixes cards in
   the UI and `POST /evaluate-winner` recomputes the winner from the corrected cards.

The live model is `poker_yolov11_whole_cards_CLEAN.pt` (YOLOv11); the loader falls
back to an older `poker_cards_best.pt` (YOLOv8) if the primary model is missing.

---

## 📊 Measured accuracy

Accuracy is measured by a standalone harness in [`backend/eval/`](backend/eval/) that
runs the **full production pipeline** on real photos and compares against
hand-labelled ground truth. On a **62-image real-photo set** (phone photos of the
kind the app targets — not the clean training distribution):

| Metric | Result |
|---|---|
| Per-card detection (recall) | **65%** (precision 76%) |
| Grouping — given a correctly-read card | **98%** |
| Correct winner named, end-to-end | **61%** |

**What this tells us:** the grouping logic is solid; the bottleneck is **detection on
real photos** — the model was fine-tuned on cleaner data, so there's a domain gap.
Most errors are *missed* cards and *suit* slips. The winner rate is "only" 61%
precisely because every one of ~9 cards has to be right to call a hand confidently.

> These numbers are from a **small, non-held-out** set — treat them as a working
> baseline, not a benchmark. See [`backend/eval/README.md`](backend/eval/README.md)
> for how to add your own labelled photos and regenerate the report.

---

## 🛠 Tech stack

**Frontend** — React 18 · Vite · TailwindCSS · Framer Motion · Lucide · react-dropzone · axios · **Capacitor** (iOS / Android shells)

**Backend** — FastAPI · **Ultralytics YOLOv11** · PyTorch · OpenCV · Pillow + `pillow-heif` (HEIC) · NumPy · a custom pure-Python hand evaluator · `pytest`

**Deployment** — Frontend on **Netlify** (CI/CD) · Backend on **Hugging Face Spaces** (Docker)

---

## 📡 API endpoints

Interactive docs at `http://localhost:8000/docs`.

### `POST /upload`
Detect cards in an image and analyze the game.
- **Body** (`multipart/form-data`): `file` — the image.
- **Query** (optional): `create_visualization` (bool) — also render an annotated image.
- **Validation**: rejects oversized uploads (**413**) and unreadable/non-image files (**400**).
- **Returns**: `detection_results`, `cards_detected`, `processing_time`, and
  `game_analysis` `{ community_cards, players[], winner, tie }`.

### `POST /evaluate-winner`
Re-evaluate the winner from manually corrected cards (used by the correction modal).
```json
{
  "player1_cards": ["As", "Ks"],
  "community_cards": ["Qs", "Js", "10s", "9h", "2c"],
  "player2_cards": ["Ah", "Kh"]
}
```
Returns the same `game_analysis` shape with the recomputed winner. Validates that each
player has exactly 2 cards and the community has 3–5 (**400** otherwise).

### `GET /health`
Liveness check plus model status.

### `GET /version`
`{ version, model_file, model_loaded, device }` — handy for confirming which weights
are loaded and whether it's running on CPU/GPU.

---

## 🏗 Local development

### Prerequisites
- **Node.js** 18+
- **Python** 3.11
- The detection weights at `backend/ml/models/poker_yolov11_whole_cards_CLEAN.pt`
  (~40 MB). Large binaries like this are best tracked with **Git LFS** rather than
  committed directly — see [Known limitations](#-known-limitations--roadmap).

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # test deps (pytest, httpx)
python main.py                    # or: uvicorn main:app --reload --port 8000
```
Backend runs at `http://localhost:8000`. CORS origins are configurable via the
`CORS_ORIGINS` env var (defaults to local dev + the deployed frontend).

### Frontend
```bash
cd frontend
npm install
npm run dev                       # http://localhost:5173
```
The frontend defaults to `http://localhost:8000` for the API; override with a
`.env` containing `VITE_API_URL=...` if needed.

---

## 📁 Project structure

```
PokerScan/
├── frontend/                       # React + Vite + Capacitor
│   ├── src/
│   │   ├── components/             # PokerLandingPage, PokerResultsPage,
│   │   │                           # CardCorrectionModal, PokerTableView,
│   │   │                           # HandComparisonPanel, WinnerAnnouncement, ...
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
│
├── backend/                        # FastAPI
│   ├── main.py                     # app + endpoints
│   ├── models/schemas.py           # Pydantic request/response models
│   ├── services/image_processor.py # orchestration: detect → analyze
│   ├── ml/
│   │   ├── card_detector.py        # YOLOv11 wrapper + de-duplication
│   │   ├── poker_game_analyzer.py  # adaptive row grouping + winner
│   │   ├── hand_evaluator.py       # hand strength (single source of truth)
│   │   └── models/                 # *.pt weights (large — use Git LFS)
│   ├── eval/                       # real-photo accuracy harness
│   └── tests/                      # pytest suite
└── README.md
```

---

## 🧪 Tests

```bash
cd backend
pytest                  # hand evaluation, adaptive grouping, and API tests
```
Frontend linting: `cd frontend && npm run lint`.

---

## 🚢 Deployment

- **Frontend (Netlify):** base `frontend`, build `npm run build`, publish `frontend/dist`,
  set `VITE_API_URL` to the backend URL.
- **Backend (Hugging Face Spaces, Docker):** push `backend/` as a Docker Space; set
  `CORS_ORIGINS` to the frontend origin(s). Track model weights with Git LFS
  (`git lfs track "*.pt"`) so they don't bloat the repo.

---

## 🧭 Known limitations & roadmap

**Limitations**
- **Detection domain gap** — the model was fine-tuned on cleaner images, so real phone
  photos (glare, angles, small cards in a large frame) currently read at ~65% per card.
  This is the main thing holding back the end-to-end winner rate.
- **Grouping edge cases** — very tilted layouts, or non-standard arrangements, can split
  rows incorrectly; these tend to come with a low internal `layout_confidence`.
- **Heads-up only** — supports exactly two players (2 + 2 hole cards + 3–5 community).
- **Model weights are committed directly** (not yet Git LFS), which bloats the repo.

**Roadmap**
- Close the detection gap by **retraining on real photos**, using the in-app correction
  flow as a free source of labels (a data flywheel) + a held-out benchmark to gate releases.
- Surface `layout_confidence` in the UI to prompt a retake when a photo is ambiguous.
- **Bankroll tracking** across sessions.
- **Multi-player** support (3+ players).
- Migrate model weights to **Git LFS**.

---

## 📝 License

MIT — see [LICENSE](LICENSE).

---

## 📧 Contact

**Idan Rodriguez** · [LinkedIn](https://www.linkedin.com/in/idanrodrigez/) ·
idan101012@gmail.com · [Portfolio](https://idanportfolio.netlify.app/)

---

**Status:** demo / portfolio project — actively improving · **v1.0.0**

⭐ If you found this useful, a star is appreciated!
