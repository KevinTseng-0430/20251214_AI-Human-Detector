# AI vs Human Detector (Streamlit, sklearn)

A lightweight bilingual (EN/中文) text classification PoC that estimates how likely a paragraph is AI-generated vs human-written. Streamlit UI, tf-idf + Logistic Regression model, bundled with tiny handcrafted samples (AI-like / Human-like in both languages).

## Project Structure
- `app.py`: Main app with data construction, model training, inference, and visualizations.
- `requirements.txt`: Dependencies (`streamlit`, `scikit-learn`, `altair`, `pandas`).

## Architecture (logical view)
- **UI (Streamlit)**: text input → probability cards → Altair charts (probability bars, top n-grams) → language stats metrics.
- **Inference pipeline**: raw text → `TfidfVectorizer` (char 3–5 gram, 6k feats) → `LogisticRegression` → probabilities + top-weighted n-grams.
- **Data layer**: tiny in-memory bilingual samples; built at startup, no external storage.
- **Performance**: model trained once and cached via `@st.cache_resource`; inference is single-pass vectorize + linear classifier (fast on CPU).
- **Extensibility points**: replace vectorizer/encoder, swap classifier, load persisted model, or connect to real labeled corpus.

## Crisp-DM Mapping

### 1) Business Understanding
- **Goal**: Quickly flag whether a text sounds more AI-like or human-like, return AI% / Human%, and show key n-grams to assist manual review.
- **Constraints**: Demo-only; tiny built-in corpus means results are not production-grade.
- **PoC Success**: Input EN/中文 text, get instant probabilities, highlight representative n-grams, and show language stats.

### 2) Data Understanding
- **Source**: Small, handcrafted bilingual samples in `build_dataset()`, labeled AI-like vs Human-like.
- **Feature idea**:
  - Character n-grams (3–5) to cover both EN/中文 without tokenization issues.
  - Weight inspection: top n-grams shown as bar charts in the UI.
- **Risks**: Very small, narrow corpus; not representative; likely overfits example tones.

### 3) Data Preparation
- **Embedded data**: `build_dataset()` returns bilingual texts and labels.
- **Vectorizer**: `TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=6000)`; language-agnostic.
- **Labels**: `1` = AI-like, `0` = Human-like.

### 4) Modeling
- **Model**: `LogisticRegression(max_iter=1000)`.
- **Pipeline**: `Pipeline([("tfidf", ...), ("clf", ...)])`.
- **Training time**: Fits on startup, cached via `@st.cache_resource` to avoid recompute.
- **Swap options**: Use word-level features, transformer embeddings, or external pretrained classifiers.

### 5) Evaluation
- **Current**: No formal metrics (dataset too small). UI shows:
  - AI% / Human% and confidence gap
  - Representative n-gram weights
  - Language stats (word count, sentences, diversity, punctuation density, etc.)
- **Recommendation**: With real labeled data, report F1 / AUC / calibration via Train/Val/Test or cross-validation.

### 6) Deployment
- **Local run**:
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```
- **Input**: EN or 中文 paragraph; instant scoring.
- **Output**:
  - AI% / Human% with confidence gap
  - Probability bar chart (Altair)
  - Language stats cards
  - Top AI-leaning / Human-leaning n-grams (bar charts)

### 7) Maintenance & Next Steps
- **Data expansion**: Collect real labeled text, balanced across EN/中文 and domains.
- **Model upgrades**: Stronger features (transformer embeddings), regularization and calibration (Platt scaling / temperature scaling).
- **Persistence**: Serialize the trained model (joblib/pickle) to skip retraining on startup.
- **MLOps**: Add evaluation scripts, monitor input distribution drift, schedule retraining.

## Usage Notes
- Demo-only results; always pair with human judgment.
- For production, retrain on real data, validate, and monitor false positives/negatives.

## Quick Troubleshooting
- **Altair rendering**: Ensure versions match `requirements.txt` and open Streamlit in the browser.
- **Chinese tokenization**: Using char n-grams already supports mixed EN/中文; for finer Chinese features, add jieba/ckip tokenization and switch to word n-grams.

## Result
<img width="773" height="575" alt="image" src="https://github.com/user-attachments/assets/6faf3db7-e575-4bf9-9191-7c3449bd8987" />
<img width="731" height="590" alt="image" src="https://github.com/user-attachments/assets/f4bf5d7e-d011-413c-8cd5-bf3ad4f8740a" />


