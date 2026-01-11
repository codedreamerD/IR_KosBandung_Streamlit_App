import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="IR Kos Bandung (TF-IDF + Cosine)", layout="wide")

DATASET_PATH = Path("data/Dataset_ReviewKos_v1.csv")
TOP_K_DEFAULT = 5
TOP_N_FEATURES_DEFAULT = 200

STOPWORDS = set([
    "yang","dan","di","ke","dari","untuk","pada","dengan","ini","itu","juga","karena","atau","agar","sebagai","adalah",
    "saya","kami","kamu","dia","mereka","nya","lah","pun","jadi","lebih","sudah","belum","tidak","ya","kok","banget",
    "dalam","akan","bisa","dapat","saat","ketika","seperti","karna","gak","nggak","tp","tapi"
])

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Zà-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    text = normalize_text(text)
    return [t for t in text.split() if t and t not in STOPWORDS]

def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [p.strip() for p in parts if p.strip()]

def identity(x):
    return x

@st.cache_resource(show_spinner=True)
def load_and_build():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH.resolve()}")

    df = pd.read_csv(DATASET_PATH)
    df["doc_id"] = df["doc_id"].astype(str).str.strip()

    # Validasi anti-human-error
    if df.shape[0] != 30:
        raise ValueError(f"Jumlah dokumen harus 30 (D1..D30). Sekarang: {df.shape[0]}")
    if df["doc_id"].nunique() != 30:
        raise ValueError("doc_id harus unik (tidak boleh duplikat).")
    if "review_text" not in df.columns:
        raise ValueError("Kolom 'review_text' wajib ada di CSV.")
    if df["review_text"].isna().any():
        raise ValueError("review_text tidak boleh NaN.")
    if (df["review_text"].astype(str).str.strip() == "").any():
        raise ValueError("review_text tidak boleh string kosong.")

    df["tokens"] = df["review_text"].astype(str).apply(tokenize)

    # Inverted index: term -> list (doc_id, tf)
    inv = defaultdict(list)
    for did, toks in zip(df["doc_id"], df["tokens"]):
        c = Counter(toks)
        for term, tf in c.items():
            inv[term].append((did, int(tf)))

    vectorizer = TfidfVectorizer(
        tokenizer=identity,
        preprocessor=identity,
        token_pattern=None,
        lowercase=False
    )
    X = vectorizer.fit_transform(df["tokens"])
    terms = np.array(vectorizer.get_feature_names_out())
    term_to_idx = {t: i for i, t in enumerate(terms)}

    idf_df = pd.DataFrame({"term": terms, "idf": vectorizer.idf_}).sort_values("idf", ascending=False).reset_index(drop=True)
    return df, inv, vectorizer, X, terms, term_to_idx, idf_df

def sentence_score(sentence: str, doc_vec, term_to_idx: dict):
    toks = tokenize(sentence)
    idxs = [term_to_idx[t] for t in toks if t in term_to_idx]
    if not idxs:
        return 0.0
    return float(np.sum([doc_vec[0, j] for j in idxs]))

def summarize(text: str, doc_vec, term_to_idx: dict, n_sent: int = 2):
    sents = split_sentences(text)
    if len(sents) <= n_sent:
        return " ".join(sents)
    scored = [(sentence_score(s, doc_vec, term_to_idx), i, s) for i, s in enumerate(sents)]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = sorted(scored[:n_sent], key=lambda x: x[1])
    return " ".join([b[2] for b in best])

def search(query: str, df: pd.DataFrame, vectorizer: TfidfVectorizer, X, term_to_idx: dict, top_k: int = 5, n_sent: int = 2):
    q_tokens = tokenize(query)
    q_vec = vectorizer.transform([q_tokens])
    sims = cosine_similarity(q_vec, X).flatten()

    res = df[["doc_id", "nama_kos", "lokasi_kecamatan", "review_text"]].copy()
    res["score"] = sims
    res = res.sort_values("score", ascending=False).reset_index(drop=True)
    res.insert(0, "rank", np.arange(1, len(res) + 1))

    top = res.head(top_k).copy()
    top["summary"] = [
        summarize(df.loc[df.index[df["doc_id"] == did][0], "review_text"], X[df.index[df["doc_id"] == did][0]], term_to_idx, n_sent=n_sent)
        for did in top["doc_id"].tolist()
    ]
    return q_tokens, top, res

st.title("Sistem Information Retrieval – Review Kos Bandung")
st.caption("TF-IDF (VSM) + Cosine Similarity + Inverted Index + Feature Selection + Extractive Summarization")

with st.spinner("Memuat dataset & membangun indeks..."):
    df, inv, vectorizer, X, terms, term_to_idx, idf_df = load_and_build()

left, right = st.columns([1, 1])

with left:
    st.subheader("Query Console")
    query = st.text_input("Masukkan query", value="murah dekat kampus")
    top_k = st.number_input("Top-K hasil", min_value=1, max_value=30, value=TOP_K_DEFAULT, step=1)
    n_sent = st.number_input("Jumlah kalimat ringkasan per dokumen", min_value=1, max_value=5, value=2, step=1)
    run = st.button("Cari", type="primary")

    st.markdown("---")
    st.subheader("Info Dataset")
    st.write(f"Jumlah dokumen: {df.shape[0]}")
    st.write(f"Jumlah term (vocabulary): {len(terms)}")

with right:
    st.subheader("Inverted Index (cek term)")
    term_input = st.text_input("Masukkan 1 term untuk melihat posting list", value="wifi")
    t = term_input.strip().lower()
    if t:
        postings = inv.get(t, [])
        st.write(f"df('{t}') = {len(postings)} dokumen")
        if postings:
            st.dataframe(pd.DataFrame(postings, columns=["doc_id", "tf"]).sort_values(["tf","doc_id"], ascending=[False, True]))
        else:
            st.info("Term tidak ditemukan (mungkin stopword/typo/berbeda bentuk kata).")

st.markdown("---")

if run:
    if query.strip() == "":
        st.error("Query tidak boleh kosong.")
    else:
        q_tokens, top, full = search(query, df, vectorizer, X, term_to_idx, top_k=int(top_k), n_sent=int(n_sent))
        st.subheader("Hasil Pencarian (Top-K)")
        st.write("Token query setelah preprocessing:", q_tokens)
        st.dataframe(top[["rank", "doc_id", "nama_kos", "lokasi_kecamatan", "score", "summary"]])

        with st.expander("Lihat full ranking"):
            st.dataframe(full[["rank", "doc_id", "nama_kos", "lokasi_kecamatan", "score"]])

st.markdown("---")
st.subheader("Feature Selection (Top-N berdasarkan IDF)")
top_n = st.number_input("Top-N fitur untuk ditampilkan", min_value=10, max_value=min(500, len(idf_df)), value=min(TOP_N_FEATURES_DEFAULT, len(idf_df)), step=10)
st.dataframe(idf_df.head(int(top_n)))
