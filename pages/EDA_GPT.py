import streamlit as st
from pages.src.setup import DataFrameEnvironment, DataManager
from pages.src.unstructured_data import unstructured_Analyzer
from pages.src.Instructions.instructions import instructions
from pages.src.Tools.scraper import extractollamamodels
from pages.src.structured_data import EDAGPT
from pages.src.Tools.secrets import initialize_secrets, initialize_states
import os, json, time
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Analyze data", layout='wide')

config_file = os.path.join('pages', 'src', 'Database', 'config.json')
with open(config_file, 'r') as file:
    config_data = json.load(file)

prompts_config_file = os.path.join('pages', 'src', 'Database', 'Prompts','prompts.json')
with open(prompts_config_file, 'r') as file:
    prompt_data = json.load(file)
st.session_state.structured_analyzer=EDAGPT(datahandler=DataManager(config_data, config_file_path=config_file),dataframe_environment=DataFrameEnvironment(), config_data=config_data, config_file=config_file, prompt_data=prompt_data)
st.session_state.unstructured_analyzer=unstructured_Analyzer(config_data,prompt_data)
messages = {
    "Instructions": '🤖 Hey its EDA GPT. I can help you with Data Analysis.',
    "Structured": '🤖 Analyze structured data here.Create EDA report, analyze data, derive insights, talk to EDA GPT and much more',
    "Unstructured": '🤖 Analyze unstructured data here. Summarize, analyze, derive key insights from pdf and internet using llms.'
}
pages=['Instructions','Structured','Unstructured']
if __name__ == "__main__":
    initialize_states()
    initialize_secrets()
    st.session_state.current_page=option_menu(menu_title=None,icons=['question','database','folder'],options=pages,menu_icon="robot",orientation="horizontal", styles={"container":{"background":"transparent", "font-size":"1rem", "border":"1px solid gray"}, "icon":{"color":"gray"}})
    st.write(f"**{messages[st.session_state.current_page]}**", unsafe_allow_html=True)
    # st.session_state.current_page=select_dcategory
    if st.session_state.current_page.lower()=='instructions':
        instructions()   
    if st.session_state.current_page.lower()=='structured':
        extractollamamodels()
        st.session_state.structured_analyzer.workflow()
    elif st.session_state.current_page.lower()=='unstructured':
        st.session_state.unstructured_analyzer.run()

