
```markdown
#  Multi-Label Emotion Detection — GoEmotions (TF-IDF + MiniLM)

This project implements a **hybrid emotion classification system** that detects multiple emotions from text.  
It combines traditional lexical features (**TF-IDF**) with contextual semantic features from a **transformer (MiniLM)**, using per-label thresholds for balanced multi-label predictions.

---

## Project Structure

```

Multi-Label Emotion Detection/
├── notebooks/
│   └── Multi-Label Emotion Detection.ipynb      ← full training notebook
├── models/                                      ← saved artifacts for inference
│   ├── tfidf_vectorizer.joblib   # TF-IDF feature extractor
│   ├── tfidf_ovr.joblib          # TF-IDF Logistic Regression classifier
│   ├── emb_model_name.txt        # name of SentenceTransformer model used
│   ├── emb_ovr.joblib            # Embedding Logistic Regression classifier
│   ├── thresholds.json           # final per-label thresholds (fused model)
│   └── thresholds_tfidf.json     # (optional) TF-IDF-only thresholds
├── app.py                        # Streamlit app for live testing
├── requirements.txt              # dependencies
└── README.md                     # this file

````

---

## 🧠 Model Overview

| Branch | Features | Model | Purpose |
|--------|-----------|--------|----------|
| **TF-IDF** | Bag-of-words n-grams (1–2) | Logistic Regression (OvR) | Captures explicit emotional keywords |
| **Embeddings** | SentenceTransformer (MiniLM) | Logistic Regression (OvR) | Captures contextual meaning |
| **Fusion** | Average of both probabilities | — | Improves robustness and generalization |

---
## Usage 

```bash
pip install -r requirements.txt
streamlit run app.py
````

Then open the Streamlit interface (usually at `http://localhost:8501`) and enter any sentence to see detected emotions and confidence scores.

---

##  Dataset

**[GoEmotions](https://huggingface.co/datasets/go_emotions)**
58k Reddit comments annotated with **27 fine-grained emotions** + 1 neutral class.
Used here for English multi-label emotion classification.

---

##  Credits

* Dataset: Google Research (GoEmotions)
* Embeddings: SentenceTransformers (MiniLM-L6-v2)
* Implementation: Logistic Regression (scikit-learn)
* App: Streamlit (for interactive demo)

```
