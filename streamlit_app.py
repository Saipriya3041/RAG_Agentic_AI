import os
import re

import requests
import streamlit as st


st.set_page_config(
    page_title="Agentic AI | Document Chat",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
SAMPLE_PROMPTS = [
    "What is the core definition of Agentic AI?",
    "What are the main architectural components?",
    "What industry use cases does the eBook discuss?",
    "What role does memory play in Agentic AI workflows?",
]

st.markdown(
    """
    <style>
    :root {
        --ink: #172321;
        --muted: #64736e;
        --line: #dce5df;
        --paper: #f5f7f3;
        --panel: #ffffff;
        --forest: #176b52;
        --forest-deep: #104b3b;
        --lime: #d9edaa;
        --coral: #c7654f;
    }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #edf2ed; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }
    p, label, button, input, textarea { font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; }
    .brand-mark {
        display: inline-flex; width: 34px; height: 34px; border-radius: 9px;
        align-items: center; justify-content: center; background: var(--forest);
        color: white; font: bold 17px Georgia, serif; margin-bottom: 1.25rem;
    }
    .eyebrow { color: var(--forest); font-size: 0.72rem; font-weight: 800;
        letter-spacing: 0.12em; text-transform: uppercase; }
    .hero-title { font: 44px/1.07 Georgia, 'Times New Roman', serif; color: var(--ink);
        margin: 0.4rem 0 0.55rem; }
    .hero-copy { color: var(--muted); font-size: 1rem; max-width: 690px; }
    .status-line { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
        padding: 0.75rem 0; margin: 1.5rem 0 1.2rem; color: var(--muted); font-size: 0.86rem; }
    .status-dot { color: var(--forest); font-size: 1.2rem; vertical-align: -1px; padding-right: 0.35rem; }
    [data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: 10px;
        background: var(--panel); padding: 1rem 1.15rem; }
    [data-testid="stChatMessage"] p { color: var(--ink); line-height: 1.6; }
    [data-testid="stChatInput"] { border-color: #bdcec4; background: white; }
    [data-testid="stChatInput"]:focus-within { border-color: var(--forest); }
    div.stButton > button { border: 1px solid #cad8cf; color: var(--forest-deep);
        background: white; border-radius: 8px; min-height: 42px; text-align: left; }
    div.stButton > button:hover { border-color: var(--forest); color: var(--forest-deep); background: #f0f6ed; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line);
        padding: 0.85rem 1rem; border-radius: 8px; }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: white; }
    .sidebar-note { color: var(--muted); font-size: 0.88rem; line-height: 1.55; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="brand-mark">A</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Document intelligence</div>', unsafe_allow_html=True)
    st.title("Agentic AI")
    st.markdown(
        '<p class="sidebar-note">Ebook-Agentic-AI.pdf<br>Answers grounded in retrieved passages.</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    api_url = st.text_input("API endpoint", value=API_URL)
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="eyebrow">The Agentic AI field guide</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Ask the eBook.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Search the document in plain language. Answers include page citations and the passages used.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="status-line"><span class="status-dot">&bull;</span> Ebook-Agentic-AI.pdf '
    '<span style="padding: 0 0.6rem; color: #b2bdb6;">/</span> Retrieval-backed answers</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def submit_query(query: str) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/chat",
            json={"query": query},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        st.session_state.messages.append({"role": "assistant", "payload": payload})
    except requests.RequestException as exc:
        st.session_state.messages.append(
            {"role": "error", "content": f"Could not reach the RAG API: {exc}"}
        )


if not st.session_state.messages:
    st.markdown('<div class="eyebrow">Start with a question</div>', unsafe_allow_html=True)
    prompt_columns = st.columns(2)
    for index, sample in enumerate(SAMPLE_PROMPTS):
        with prompt_columns[index % 2]:
            if st.button(sample, key=f"sample-{index}", use_container_width=True):
                st.session_state.pending_query = sample
                st.rerun()

for message_index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "error":
        st.error(message["content"])
    else:
        payload = message["payload"]
        with st.chat_message("assistant"):
            st.markdown(payload["final_answer"])
            confidence = float(payload.get("confidence_score", 0.0))
            st.progress(max(0.0, min(confidence, 1.0)), text=f"Evidence confidence - {confidence:.2f}")
            chunks = payload.get("retrieved_context_chunks", [])
            if chunks:
                with st.expander(f"Retrieved evidence - {len(chunks)} passages"):
                    for chunk in chunks:
                        page_match = re.match(r"\[Page ([^\]]+)\]\s*", chunk)
                        page = page_match.group(1) if page_match else "Source"
                        text = chunk[page_match.end():] if page_match else chunk
                        st.markdown(f"**Page {page}**")
                        st.write(text)

pending_query = st.session_state.pop("pending_query", None)
if pending_query:
    with st.spinner("Searching the eBook..."):
        submit_query(pending_query)
    st.rerun()

query = st.chat_input("Ask a question about the eBook")
if query:
    with st.spinner("Searching the eBook..."):
        submit_query(query)
    st.rerun()