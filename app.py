import os
import io
import time
import numpy as np
import streamlit as st

# Optional LLM (uses OpenAI if OPENAI_API_KEY is set in Streamlit Secrets)
USE_LLM = False
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        USE_LLM = True
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    USE_LLM = False

# Lightweight embedding + retrieval (local; no keys required)
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss

st.set_page_config(page_title="Brand Guideline RAG (MVP)", layout="wide")
st.title("Brand Guideline RAG (MVP)")
st.caption("Upload your brand guideline PDFs, ask questions, and get answers with citations.")

@st.cache_resource(show_spinner=False)
def load_embedder():
    # small, fast model good enough for MVPs
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def new_index(dim):
    return faiss.IndexFlatIP(dim)

def chunk_text(text, chunk_size=700, overlap=120):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def pdf_to_pages(file):
    reader = PdfReader(file)
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            pages.append((i+1, p.extract_text() or ""))
        except Exception:
            pages.append((i+1, ""))
    return pages

def build_corpus(files):
    docs = []
    meta = []
    for uploaded in files:
        pages = pdf_to_pages(uploaded)
        for pg, txt in pages:
            if not txt or not txt.strip():
                continue
            for ch in chunk_text(txt):
                docs.append(ch)
                meta.append({"filename": uploaded.name, "page": pg})
    return docs, meta

def embed_docs(embedder, docs):
    embs = embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embs.astype("float32")

def top_k(query, embedder, index, docs, meta, k=5):
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: 
            continue
        hits.append((float(score), docs[idx], meta[idx]))
    return hits

def answer_from_context(query, contexts):
    # If LLM available, synthesize; else return stitched extract
    if USE_LLM:
        sys = "You are a helpful assistant answering questions based only on the provided context. Cite filenames and page numbers."
        context_text = "\n\n".join([f"[{m['filename']} p.{m['page']}] {c}" for _, c, m in contexts])
        prompt = f"{sys}\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer succinctly with citations in brackets."
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(LLM error; falling back to extracts)\n\n" + fallback_answer(contexts)
    else:
        return fallback_answer(contexts)

def fallback_answer(contexts):
    # Simple extractive fallback: return the most relevant snippets with citations
    parts = []
    for score, chunk, m in contexts:
        parts.append(f"[{m['filename']} p.{m['page']}] {chunk[:600]}")
    return "\n\n".join(parts)

# Sidebar upload + settings
with st.sidebar:
    st.header("Upload PDFs")
    files = st.file_uploader("Brand guidelines or policies (PDF)", type=["pdf"], accept_multiple_files=True)
    k = st.slider("Retrieved chunks", 3, 10, 5, 1)
    st.divider()
    st.markdown("**LLM available:** " + ("✅ (OpenAI key set)" if USE_LLM else "❌ (using extractive fallback)"))
    st.markdown("Add OPENAI_API_KEY in **Settings → Secrets** to enable synthesis.")

if files:
    with st.spinner("Building index…"):
        embedder = load_embedder()
        docs, meta = build_corpus(files)
        if not docs:
            st.warning("No extractable text found. Check your PDFs.")
            st.stop()
        embs = embed_docs(embedder, docs)
        index = new_index(embs.shape[1])
        index.add(embs)
    st.success(f"Indexed {len(docs)} chunks from {len(files)} file(s).")

    query = st.text_input("Ask a question about your brand guidelines")
    if query:
        start = time.time()
        contexts = top_k(query, embedder, index, docs, meta, k=k)
        ans = answer_from_context(query, contexts)
        elapsed = (time.time() - start) * 1000

        st.subheader("Answer")
        st.write(ans)

        st.caption(f"Latency: {elapsed:.0f} ms • Retrieved chunks: {len(contexts)}")
        with st.expander("Sources"):
            for score, chunk, m in contexts:
                st.markdown(f"- **{m['filename']}**, p.{m['page']} (sim={score:.3f})")
else:
    st.info("Upload at least one PDF to begin.")
