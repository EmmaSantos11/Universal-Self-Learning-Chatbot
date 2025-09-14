"""
Cloud-native Self-Learning Chatbot
Runs on: Streamlit-Cloud, local, phone, colab
Learns: per-user memory + nightly web scrape + auto-search
Embeddings: Hugging-Face Inference API (free, no local model)
Vector store: FAISS-cpu (no Chroma, no torch)
LLM: Gemini (default) or OpenAI
Author: Ohamadike Chidera Emmanuel
"""
import os
import json
import time
import threading
import requests
from datetime import datetime
from typing import List

import streamlit as st
import faiss
import numpy as np
from ddgs import DDGS
from bs4 import BeautifulSoup
import openai

# ---------- ENV ----------
HF_TOKEN   = st.secrets.get("HF_TOKEN")   or os.getenv("HF_TOKEN")
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not HF_TOKEN:
    st.error("🔑 Add HF_TOKEN to .env or Streamlit Secrets (free: https://huggingface.co/settings/tokens)")
    st.stop()

# ---------- CLIENTS ----------
import google.generativeai as genai
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- CONFIG ----------
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
MEMORY_DIR   = "./faiss_index"
os.makedirs(MEMORY_DIR, exist_ok=True)

# ---------- UI ----------
st.set_page_config(page_title="∞ Cloud-Native Bot", layout="wide")
st.title("∞ Cloud-Native Self-Learning Bot")
st.caption("No torch, no Chroma – just cloud APIs and FAISS. Deploy anywhere.")

with st.sidebar:
    provider = st.radio("LLM provider", ["Gemini", "OpenAI"], index=0,
                        disabled=not bool(GEMINI_KEY or OPENAI_KEY))
    st.info("💡 Tip: paste a URL or ask anything – I’ll search & remember.")

# ---------- FAISS MEMORY ----------
INDEX_FILE = os.path.join(MEMORY_DIR, "index.faiss")
META_FILE  = os.path.join(MEMORY_DIR, "meta.json")

def load_or_create_index():
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf8") as f:
            meta = json.load(f)
    else:
        index = faiss.IndexFlatL2(384)          # MiniLM-L6 dimension
        meta = []
    return index, meta

def save_index(index, meta):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

index, meta = load_or_create_index()

def embed(text: str) -> np.ndarray:
    """Call HF Inference API (free)"""
    print(f"[DEBUG] HF_TOKEN='{HF_TOKEN[:10]}...'")   # ← shows first 10 chars only
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp = requests.post(HF_EMBED_URL, headers=headers, json={"inputs": text}, timeout=30)
    resp.raise_for_status()
    return np.array(resp.json(), dtype=np.float32)

def store_memory(text: str, src: str):
    vec = embed(text)
    index.add(vec.reshape(1, -1))
    meta.append({"text": text, "source": src, "time": datetime.utcnow().isoformat()})
    save_index(index, meta)

def retrieve_memory(query: str, k: int = 5) -> str:
    if index.ntotal == 0:
        return ""
    vec = embed(query)
    D, I = index.search(vec.reshape(1, -1), k)
    return "\n".join([meta[i]["text"] for i in I[0] if i < len(meta)])

# ---------- WEB SEARCH ----------
def search_web(query: str) -> List[str]:
    with DDGS() as ddgs:
        return [r["body"] for r in ddgs.text(query, max_results=5)]

def scrape_and_learn(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        text = BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)[:10_000]
        store_memory(text, url)
        return text[:500]
    except Exception as e:
        return f"Scrape error: {e}"

# ---------- LLM CALL ----------
def llm_complete(prompt: str, max_tokens: int = 2048, temp: float = 0.7) -> str:
    if provider == "Gemini":
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tokens
            )
        )
        return response.text.strip()
    else:  # OpenAI
        client = openai.OpenAI(api_key=OPENAI_KEY)
        return client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp
        ).choices[0].text.strip()

# ---------- CHAT ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask or paste a URL – the sky is the limit"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        mem = retrieve_memory(prompt)
        search = ""
        if any(k in prompt.lower() for k in ("search", "fact", "define", "what", "how", "why", "latest")):
            search = "\n".join(search_web(prompt))
        scrape = ""
        if prompt.startswith("http"):
            scrape = scrape_and_learn(prompt)

        system = (
            "You are a helpful, ever-learning assistant.\n"
            f"Relevant memory:\n{mem}\n"
            f"Search results:\n{search}\n"
            f"Scraped content:\n{scrape}\n"
        )
        mega_prompt = system + "\nUser: " + prompt + "\nAssistant:"
        reply = llm_complete(mega_prompt)
        store_memory(reply, "assistant")

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"⚠️ {e}")

# ---------- EXPORT ----------
if st.sidebar.button("Download memory JSON"):
    data = {"index": index.ntotal, "meta": meta}
    json_str = json.dumps(data, indent=2, default=str)
    st.sidebar.download_button("📥 memory.json", json_str, file_name="memory.json")