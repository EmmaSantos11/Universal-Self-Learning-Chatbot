"""
Universal Self-Learning Chatbot
Runs on: Streamlit-Cloud, local, phone, colab
Learns: per-user memory + nightly web scrape + auto-search
Gemini default (free), OpenAI optional
Author: Ohamadike Chidera Emmanuel
"""
import os, time, json, threading, schedule, requests
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from ddgs import DDGS
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import openai

# ------------------ ENV ------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not GEMINI_KEY and not OPENAI_KEY:
    st.error("🔑 Add GEMINI_API_KEY (and/or OPENAI_API_KEY) to .env or Secrets")
    st.stop()

# ------------------ CLIENTS ------------------
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# ------------------ CONFIG ------------------
MODEL_NAME = "gemini-2.0-flash-exp"
MAX_TOKENS = 30_000
MEMORY_DIR = "./long_term_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

# ------------------ UI ------------------
st.set_page_config(page_title="∞ Self-Learning Bot", layout="wide")
st.title("∞ Self-Learning Chatbot")
st.caption("Learns from you, surfs the web, remembers everything. Deploy once, use anywhere.")

with st.sidebar:
    provider = st.radio("LLM provider", ["Gemini", "OpenAI"], index=0,
                        disabled=not bool(GEMINI_KEY))
    st.info("💡 Tip: paste a URL or ask anything – I’ll search & remember.")

# ------------------ MODELS ------------------
@st.cache_resource(show_spinner=False)
def load_models():
    tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    dialo = pipeline("text-generation", model=mdl, tokenizer=tok,
                     pad_token_id=tok.eos_token_id, return_full_text=False)
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    return dialo, emb

dialo_chat, embedder = load_models()

# ------------------ CHROMA MEMORY ------------------
chroma_client = chromadb.PersistentClient(path=MEMORY_DIR)
collection = chroma_client.get_or_create_collection(name="universal_memory")

def store_memory(text: str, meta: dict):
    collection.add(documents=[text], metadatas=[meta], ids=[str(hash(text))])

def retrieve_memory(query: str, k=5):
    emb = embedder.encode(query).tolist()
    docs = collection.query(query_embeddings=[emb], n_results=k)["documents"]
    if not docs:
        return ""
    flat = [item for sublist in docs for item in sublist]
    return "\n".join(flat)

# ------------------ WEB SEARCH ------------------
def search_web(query: str):
    with DDGS() as ddgs:
        return [r["body"] for r in ddgs.text(query, max_results=5)]

# ------------------ WEB SCRAPE ------------------
def scrape_and_learn(url: str):
    try:
        rsp = requests.get(url, timeout=10)
        soup = BeautifulSoup(rsp.text, "lxml")
        text = soup.get_text(" ", strip=True)[:10_000]
        store_memory(text, {"source": url, "time": datetime.utcnow().isoformat()})
        return text[:500]
    except Exception as e:
        return f"Scrape error: {e}"

# ------------------ BACKGROUND SCRAPER ------------------
def nightly_scrape():
    urls = ["https://en.wikipedia.org/wiki/Main_Page", "https://news.ycombinator.com"]
    for u in urls:
        scrape_and_learn(u)
    store_memory("Background scrape finished.", {"source": "scheduler"})

def run_scheduler():
    schedule.every().day.at("03:00").do(nightly_scrape)
    while True:
        schedule.run_pending()
        time.sleep(60)

threading.Thread(target=run_scheduler, daemon=True).start()

# ------------------ LLM CALL ------------------
def llm_complete(prompt: str, max_tokens: int = MAX_TOKENS, temp: float = 0.7) -> str:
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

# ------------------ CHAT ------------------
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
        store_memory(reply, {"role": "assistant", "time": datetime.utcnow().isoformat()})

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"⚠️ {e}")

# ------------------ EXPORT MEMORY ------------------
if st.sidebar.button("Download memory JSON"):
    data = collection.get()
    json_str = json.dumps({"memory": data}, indent=2, default=str)
    st.sidebar.download_button("📥 memory.json", json_str, file_name="memory.json")