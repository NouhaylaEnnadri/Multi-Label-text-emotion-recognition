# app.py — GoEmotions (CPU) · DRY + lazy-loading
import os, re, json, traceback
import numpy as np
import joblib
import streamlit as st

# ---------- hard-disable GPU (avoid meta tensor issues) ----------
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---------- page ----------
st.set_page_config(page_title="GoEmotions — Multi-label", layout="wide")
st.title("GoEmotions — Multi-label Emotion Detector")
st.caption("Hybrid TF-IDF + MiniLM. Lazy-loaded models.")

# ---------- paths ----------
MODEL_DIR = "models"
TFIDF_VEC = f"{MODEL_DIR}/tfidf_vectorizer.joblib"
TFIDF_OVR = f"{MODEL_DIR}/tfidf_ovr.joblib"
EMB_OVR   = f"{MODEL_DIR}/emb_ovr.joblib"
EMB_NAMEF = f"{MODEL_DIR}/emb_model_name.txt"
THR_JSON  = f"{MODEL_DIR}/thresholds.json"

# ---------- tiny cleaner (for TF-IDF branch only) ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
def clean_for_tfidf(t: str) -> str:
    t = URL_RE.sub("<URL>", t)
    t = USER_RE.sub("<USER>", t)
    return re.sub(r"\s+", " ", t).strip().lower()

# ---------- cached loaders ----------
@st.cache_resource(show_spinner=False)
def load_thresholds():
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
def load_emb_branch_cpu():
    # import inside; force CPU
    from sentence_transformers import SentenceTransformer
    name = open(EMB_NAMEF, "r", encoding="utf-8").read().strip()
    sbert = SentenceTransformer(name, device="cpu")
    clf = joblib.load(EMB_OVR)
    return sbert, clf, name

# ---------- helpers ----------
def fuse_probs(prob_list, method="mean"):
    if not prob_list:
        return None
    if method == "mean":
        return np.mean(np.stack(prob_list, axis=0), axis=0)
    # hooks for future (e.g., weighted)
    return np.mean(np.stack(prob_list, axis=0), axis=0)

def predict_once(text, use_tfidf, use_emb, fuse_method, labels, thr_vec,
                 vec=None, clf_tfidf=None, sbert=None, clf_emb=None,
                 topk_fallback=0):
    probs = []

    if use_tfidf and vec is not None and clf_tfidf is not None:
        Xt = vec.transform([clean_for_tfidf(text)])
        p_tfidf = clf_tfidf.predict_proba(Xt)[0]
        probs.append(p_tfidf)

    if use_emb and sbert is not None and clf_emb is not None:
        Xe = sbert.encode([text], convert_to_numpy=True)
        p_emb = clf_emb.predict_proba(Xe)[0]
        probs.append(p_emb)

    p = fuse_probs(probs, method=fuse_method)
    if p is None:
        return [], None  # no active branch

    picked = [(labels[j], float(p[j])) for j in range(len(labels)) if p[j] >= thr_vec[j]]
    if not picked and topk_fallback > 0:
        idx = np.argsort(-p)[:topk_fallback]
        picked = [(labels[i], float(p[i])) for i in idx]
    picked.sort(key=lambda t: -t[1])
    return picked, p

# ---------- sidebar: diagnostics + about ----------
with st.sidebar:
    st.header("Diagnostics")
    st.write("Working dir:", os.getcwd())
    st.write("Models dir:", os.path.isdir(MODEL_DIR))
    st.write("TF-IDF files:", os.path.exists(TFIDF_VEC), os.path.exists(TFIDF_OVR))
    st.write("Emb files:", os.path.exists(EMB_NAMEF), os.path.exists(EMB_OVR))
    st.write("Thresholds:", os.path.exists(THR_JSON))
    st.markdown("---")
    st.header("About")
    st.write(
        "TF-IDF captures lexical cues; MiniLM captures semantic context. "
        "We fuse probabilities and apply per-label thresholds."
    )

# ---------- UI controls ----------
colA, colB = st.columns([2, 1])
with colA:
    text = st.text_area("Enter a sentence:", "Tomorrow’s the big day — can’t sleep, can’t wait!")
with colB:
    topk = st.slider("Minimum results (top-K fallback)", 0, 5, 3)
    use_tfidf = st.checkbox("Use TF-IDF branch", True)
    use_emb   = st.checkbox("Use Embeddings branch", True)
    fuse_method = st.selectbox("Fusion", ["mean"], index=0)
    apply_thresholds = st.checkbox("Apply per-label thresholds", True)

# ---------- load models (lazy) ----------
if st.button("Load models"):
    with st.status("Loading models on CPU…", expanded=True) as status:
        try:
            labels, thr_vec = load_thresholds()
            st.write(f"• Thresholds & labels loaded: {len(labels)}")

            vec = clf_tfidf = sbert = clf_emb = None
            if use_tfidf:
                vec, clf_tfidf = load_tfidf_branch()
                st.write("• TF-IDF branch ready")
            if use_emb:
                sbert, clf_emb, emb_name = load_emb_branch_cpu()
                st.write(f"• Embeddings branch ready ({emb_name})")

            st.session_state.ready = dict(
                labels=labels, thr_vec=thr_vec,
                vec=vec, clf_tfidf=clf_tfidf,
                sbert=sbert, clf_emb=clf_emb
            )
            status.update(label="Models loaded ", state="complete")
        except Exception:
            st.error("Failed to load models.")
            st.code(traceback.format_exc())
            status.update(label="Load failed ", state="error")

# ---------- predict ----------
if "ready" in st.session_state:
    R = st.session_state.ready
    if st.button("Analyze"):
        try:
            labels, thr_vec = R["labels"], R["thr_vec"]
            # if user disables thresholds, use zeros so everything passes
            thr_used = thr_vec if apply_thresholds else np.zeros_like(thr_vec)

            picked, p = predict_once(
                text=text, use_tfidf=use_tfidf, use_emb=use_emb, fuse_method=fuse_method,
                labels=labels, thr_vec=thr_used,
                vec=R["vec"], clf_tfidf=R["clf_tfidf"],
                sbert=R["sbert"], clf_emb=R["clf_emb"],
                topk_fallback=topk
            )

            if p is None:
                st.warning("No active branch. Enable at least one (TF-IDF or Embeddings).")
            else:
                st.subheader("Detected emotions")
                if not picked:
                    st.info("No label passed thresholds. Increase top-K or disable thresholds.")
                else:
                    for lbl, score in picked:
                        st.write(f"**{lbl.capitalize()}** — {score:.3f}")

                # bar chart of all probabilities
                import pandas as pd
                df = pd.DataFrame({"label": labels, "prob": p}).sort_values("prob", ascending=False)
                st.subheader("All probabilities")
                st.bar_chart(df.set_index("label"))
        except Exception:
            st.error("Prediction error:")
            st.code(traceback.format_exc())
else:
    st.info("Click **Load models** first. (Embeddings download on first run; UI stays responsive.)")
