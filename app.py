from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from src.db import init_duckdb
from src.rag import build_vectorstore, load_vectorstore
from src.agent import answer

# Choose embeddings (free local option)
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Hybrid Financial Analyst Bot", layout="wide")
st.title("Hybrid Financial Analyst Chatbot")

if "state" not in st.session_state:
    st.session_state.state = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_con():
    return init_duckdb("data/financial_data.csv")

@st.cache_resource
def get_vectordb():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # build once if store not present
    import os
    if not os.path.exists("chroma_store"):
        return build_vectorstore("docs", emb, "chroma_store")
    return load_vectorstore(emb, "chroma_store")

con = get_con()
vectordb = get_vectordb()

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about financial metrics or strategic outlook…")
if user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    result = answer(user_q, st.session_state.state, con, vectordb)

    with st.chat_message("assistant"):
        st.markdown(result["final"])
        with st.expander("Traceability"):
            st.json(result["trace"])

    st.session_state.messages.append({"role":"assistant","content":result["final"]})
