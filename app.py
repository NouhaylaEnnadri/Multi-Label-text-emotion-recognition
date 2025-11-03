# app.py — GoEmotions Emotion Detector (CPU, lazy-loading, safe)
import os, re, json, traceback
import numpy as np
import joblib
import streamlit as st

# ---- Force CPU & avoid meta-tensor issues ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard-disable GPU for this app

st.set_page_config(page_title="GoEmotions — Multi-label (CPU)", layout="wide")
st.title("🎭 GoEmotions — Multi-label Emotion Detector (CPU)")
st.caption("Lazy-loaded models, thresholds per label. Runs on CPU.")

# ---- Paths (adjust if needed) ----
MODEL_DIR = "models"
TFIDF_VEC = f"{MODEL_DIR}/tfidf_vectorizer.joblib"
TFIDF_OVR = f"{MODEL_DIR}/tfidf_ovr.joblib"
EMB_OVR   = f"{MODEL_DIR}/emb_ovr.joblib"
EMB_NAMEF = f"{MODEL_DIR}/emb_model_name.txt"
THR_JSON  = f"{MODEL_DIR}/thresholds.json"

# ---- Simple cleaner (same as your notebook TF-IDF branch) ----
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
def clean_for_tfidf(t: str) -> str:
    t = URL_RE.sub("<URL>", t)
    t = USER_RE.sub("<USER>", t)
    return re.sub(r"\s+", " ", t).strip().lower()

# ---- Sidebar diagnostics ----
with st.sidebar:
    st.header("Diagnostics")
    st.write("Working dir:", os.getcwd())
    st.write("Models dir exists:", os.path.isdir(MODEL_DIR))
    st.write("TF-IDF files:", os.path.exists(TFIDF_VEC), os.path.exists(TFIDF_OVR))
    st.write("Emb files:", os.path.exists(EMB_NAMEF), os.path.exists(EMB_OVR))
    st.write("Thresholds:", os.path.exists(THR_JSON))

# ---- Cached loaders (lazy) ----
@st.cache_resource(show_spinner=False)
def load_thresholds_and_labels():
    thr_map = json.loads(open(THR_JSON, "r", encoding="utf-8").read())
    labels = list(thr_map.keys())
    thr_vec = np.array([thr_map[lbl] for lbl in labels], dtype=float)
    return labels, thr_vec

@st.cache_resource(show_spinner=False)
def load_tfidf_branch():
    vec = joblib.load(TFIDF_VEC)
    clf = joblib.load(TFIDF_OVR)
    return vec, clf

@st.cache_resource(show_spinner=False)
def load_embeddings_branch_cpu():
    # Import inside and force device='cpu' to avoid meta-tensor issues
    from sentence_transformers import SentenceTransformer
    name = open(EMB_NAMEF, "r", encoding="utf-8").read().strip()
    sbert = SentenceTransformer(name, device="cpu")
    clf = joblib.load(EMB_OVR)
    return sbert, clf, name

# ---- UI controls ----
colA, colB = st.columns([2, 1])
with colA:
    text = st.text_area(
        "Enter your sentence:",
        "Tomorrow’s the big day — can’t sleep, can’t wait!"
    )
with colB:
    topk = st.slider("Minimum results (fallback top-K)", 0, 5, 3)
    use_tfidf = st.checkbox("Use TF-IDF branch", value=True)
    use_emb   = st.checkbox("Use Embeddings branch", value=True)
    fuse = st.selectbox("Fusion", ["mean"], index=0)

# ---- Load button (so UI renders immediately) ----
if st.button("Load models"):
    with st.status("Loading models on CPU…", expanded=True) as status:
        try:
            labels, thr_vec = load_thresholds_and_labels()
            st.write(f"• Thresholds & labels loaded: {len(labels)}")

            vec = clf_tfidf = sbert = clf_emb = emb_name = None
            if use_tfidf:
                vec, clf_tfidf = load_tfidf_branch()
                st.write("• TF-IDF branch ready")
            if use_emb:
                sbert, clf_emb, emb_name = load_embeddings_branch_cpu()
                st.write(f"• Embeddings branch ready ({emb_name})")

            st.session_state.ready = dict(
                labels=labels, thr_vec=thr_vec,
                vec=vec, clf_tfidf=clf_tfidf,
                sbert=sbert, clf_emb=clf_emb
            )
            status.update(label="Models loaded ✅", state="complete")
        except Exception:
            st.error("Failed to load models.")
            st.code(traceback.format_exc())
            status.update(label="Load failed ❌", state="error")

# ---- Predict button ----
if "ready" in st.session_state:
    R = st.session_state.ready
    if st.button("Analyze"):
        try:
            probs_list = []

            if use_tfidf and R["vec"] is not None and R["clf_tfidf"] is not None:
                Xt = R["vec"].transform([clean_for_tfidf(text)])
                p_tfidf = R["clf_tfidf"].predict_proba(Xt)[0]
                probs_list.append(p_tfidf)

            if use_emb and R["sbert"] is not None and R["clf_emb"] is not None:
                Xe = R["sbert"].encode([text], convert_to_numpy=True)
                p_emb = R["clf_emb"].predict_proba(Xe)[0]
                probs_list.append(p_emb)

            if not probs_list:
                st.warning("No active branch. Enable at least one.")
            else:
                p = np.mean(np.stack(probs_list, axis=0), axis=0)  # mean fusion
                picked = [(R["labels"][j], float(p[j]))
                          for j in range(len(R["labels"]))
                          if p[j] >= R["thr_vec"][j]]
                if not picked and topk > 0:
                    idx = np.argsort(-p)[:topk]
                    picked = [(R["labels"][i], float(p[i])) for i in idx]
                picked.sort(key=lambda t: -t[1])

                st.subheader("Detected emotions")
                if not picked:
                    st.info("No label passed thresholds. Try increasing top-K.")
                else:
                    for lbl, score in picked:
                        st.write(f"**{lbl.capitalize()}** — {score:.3f}")

                # Bar chart of all probabilities
                import pandas as pd
                df = pd.DataFrame({"label": R["labels"], "prob": p}).sort_values("prob", ascending=False)
                st.subheader("All probabilities")
                st.bar_chart(df.set_index("label"))
        except Exception:
            st.error("Prediction error:")
            st.code(traceback.format_exc())
else:
    st.info("Click **Load models** first (downloads embeddings on first run; UI stays responsive).")
