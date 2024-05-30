from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms.ollama import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms.anthropic import Anthropic
import streamlit as st
import asyncio

def ensure_asyncio_event_loop():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

def get_llm(llm_name, temperature, config_data, llm_category):
    try:
        ensure_asyncio_event_loop()
        if "gemini" in llm_category:
            llm = ChatGoogleGenerativeAI(google_api_key=st.session_state.google_gemini_api, model=llm_name, temperature=temperature)
        elif "huggingface" in llm_category:
            llm = HuggingFaceEndpoint(repo_id=llm_name, huggingfacehub_api_token=st.session_state.huggingfacehub_api_token, temperature=temperature)
        elif "openai" in llm_category:
            llm = ChatOpenAI(model=llm_name, temperature=temperature, api_key=st.session_state.openai_api_key)
        elif "groq" in llm_category:
            llm = ChatGroq(api_key=st.session_state.groq_api_key, temperature=temperature, model=llm_name)
        elif "antrophic" in llm_category:
            llm = Anthropic(model_name=llm_name, temperature=temperature, anthropic_api_key=st.session_state.antrophic_api_token)
        elif "ollama" in llm_category:
            llm = Ollama(model=llm_name, temperature=temperature)
        return llm
    
    except Exception as e:
        raise {e}