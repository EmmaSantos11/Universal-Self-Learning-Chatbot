"""
Cloud-native Self-Learning Chatbot
- Vector store: FAISS (no Chroma, no torch)
- Embeddings: Hugging-Face Inference API (free, no local model)
- Search: ddgs
- Memory: local FAISS index
- LLM: Gemini (default) or OpenAI
Author: Ohamadike Emmanuel Chidera
"""
import os, json, time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import requests
import faiss
import numpy as np
from ddgs import DDGS
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import openai

# ---------- ENV ----------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not GEMINI_KEY and not OPENAI_KEY:
    st.error("🔑 Add GEMINI_API_KEY (and/or OPENAI_API_KEY) to .env or Secrets")
    st.stop()

# ---------- CLIENTS ----------
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# ---------- CONFIG ----------
MODEL_NAME = "gemini-2.0-flash-exp"
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
MEMORY_DIR = "./faiss_index"
os.makedirs(MEMORY_DIR, exist_ok=True)

# ---------- UI ----------
st.set_page_config(page_title="∞ Cloud-Native Bot", layout="wide")
st.title("∞ Cloud-Native Self-Learning Bot")
st.caption("No torch, no Chroma – just cloud APIs and FAISS. Deploy anywhere.")

with st.sidebar:
    provider = st.radio("LLM provider", ["Gemini", "OpenAI"], index=0,
                        disabled=not bool(GEMINI_KEY))
    st.info("💡 Tip: paste a URL or ask anything – I’ll search & remember.")

# ---------- FAISS MEMORY ----------
INDEX_FILE = os.path.join(MEMORY_DIR, "index.faiss")
META_FILE = os.path.join(MEMORY_DIR, "meta.json")

def load_or_create_index():
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r") as f:
            meta = json.load(f)
    else:
        index = faiss.IndexFlatL2(384)  # MiniLM-L6 dimension
        meta = []
    return index, meta

def save_index(index, meta):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w") as f:
        json.dump(meta, f)

index, meta = load_or_create_index()

def embed(text: str) -> np.ndarray:
    headers = {"Authorization": f"Bearer {GEMINI_KEY or OPENAI_KEY}"}
    resp = requests.post(HF_EMBED_URL, headers=headers, json={"inputs": text})
    resp.raise_for_status()
    return np.array(resp.json(), dtype=np.float32)

def store_memory(text: str, src: str):
    vec = embed(text)
    index.add(vec.reshape(1, -1))
    meta.append({"text": text, "source": src, "time": datetime.utcnow().isoformat()})
    save_index(index, meta)

def retrieve_memory(query: str, k=5):
    if index.ntotal == 0:
        return ""
    vec = embed(query)
    D, I = index.search(vec.reshape(1, -1), k)
    return "\n".join([meta[i]["text"] for i in I[0] if i < len(meta)])

# ---------- WEB SEARCH ----------
def search_web(query: str):
    with DDGS() as ddgs:
        return [r["body"] for r in ddgs.text(query, max_results=5)]

def scrape_and_learn(url: str):
    try:
        r = requests.get(url, timeout=10)
        text = BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)[:10_000]
        store_memory(text, url)
        return text[:500]
    except Exception as e:
        return f"Scrape error: {e}"

# ---------- LLM CALL ----------
def llm_complete(prompt: str, max_tokens: int = 4_000, temp: float = 0.7) -> str:
    if provider == "Gemini":
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temp, max_output_tokens=max_tokens)
        )
        return response.text.strip()
    else:
        return openai.Completion.create(
            engine="text-davinci-003",
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
        system = f"You are a helpful, ever-learning assistant.\nRelevant memory:\n{mem}\nSearch results:\n{search}\nScraped content:\n{scrape}\n"
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