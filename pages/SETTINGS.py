import os
import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

st.set_page_config(page_title="Settings", layout='wide')

class Settings:
    def __init__(self):
        # self.huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        # self.google_gemini_api = os.getenv("GOOGLE_GEMINI_API")
        # self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.groq_api_key = os.getenv("GROQ_API_KEY")
        # self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        self.huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]['api_token']
        self.google_gemini_api = st.secrets["GOOGLE_GEMINI_API"]['api_key']
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]['api_key']
        self.groq_api_key = st.secrets["GROQ_API_KEY"]['api_key']
        self.anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]['api_key']
        st.session_state.huggingfacehub_api_token = self.huggingfacehub_api_token
        st.session_state.google_gemini_api = self.google_gemini_api
        st.session_state.openai_api_key = self.openai_api_key
        st.session_state.groq_api_key = self.groq_api_key
        st.session_state.anthropic_api_key = self.anthropic_api_key



    def changesettings(self):  
        with st.expander('provide api keys'):
            # If user provides their own key, use that instead
            self.huggingfacehub_api_token_input = st.text_input("HuggingFace Hub API Token")
            self.google_gemini_api_input = st.text_input("Google Gemini API Key")
            self.openai_api_key_input = st.text_input("OpenAI API Key")
            self.groq_api_key_input = st.text_input("Groq API Key")
            self.anthropic_api_key_input = st.text_input("ANTHROPIC api key")

            if st.button('apply'):

                st.session_state.huggingfacehub_api_token = self.huggingfacehub_api_token_input if self.huggingfacehub_api_token_input else self.huggingfacehub_api_token
                st.session_state.google_gemini_api = self.google_gemini_api_input if self.google_gemini_api_input else self.google_gemini_api
                st.session_state.openai_api_key = self.openai_api_key_input if self.openai_api_key_input else self.openai_api_key
                st.session_state.anthropic_api_key = self.anthropic_api_key_input if self.anthropic_api_key_input else self.anthropic_api_key
                st.session_state.groq_api_key = self.groq_api_key_input if self.groq_api_key_input else self.groq_api_key


        with st.expander('choose embeddings'):
            embeddings={'HuggingFaceHubEmbeddings':0,'GooglePalmEmbeddings':1,'HuggingFaceInferenceAPIEmbeddings':2}
            embedding=st.radio(label='Choose among 3 embeddings that best suit performance on your data',options=list(embeddings.keys()), index=[st.session_state.embeddings if st.session_state.embeddings is not None else 2][0])
            if st.button('apply embedding'):
                st.session_state.embeddings=embeddings[embedding]



settings=Settings()
settings.changesettings()
   