import streamlit as st
import os

def initialize_secrets():
    st.session_state.tavily_api_key = st.secrets['TAVILY_API_KEY']['api_key']
    st.session_state.bing_api_key = st.secrets['BING_API_KEY']['api_key']

def initialize_states():
    vars=['embeddings','vectorstoreretriever','loaded_vstore','uploaded_files',"current_page","huggingfacehub_api_token","google_gemini_api","openai_api_key","anthropic_api_key","groq_api_key"]
    for var in vars:
        if var not in st.session_state:
            if var in vars[5:]:
                st.session_state[var]=st.secrets[var.upper()]['api_token' if var=="huggingfacehub_api_token" else "api_key"]
            elif var=='current_page':
                st.session_state[var]="INSTRUCTIONS"
            elif var in vars[0:4]:
                st.session_state[var]=None