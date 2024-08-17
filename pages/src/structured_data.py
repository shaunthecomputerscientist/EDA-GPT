import streamlit as st
from pages.src.EDA import EDAAnalyzer
import streamlit.components.v1 as components
from streamlit_extras import colored_header
import os
import pandas as pd
import logging
import json
import io
from io import StringIO
from streamlit_option_menu import option_menu
from pages.src.automl.fengineering import Automl


class EDAGPT:
    def __init__(self,datahandler,dataframe_environment, config_data, **kwargs):
        
        
        # Initialize session state variables
        variables_to_initialize = {
        'edastatus': False,
        'analysis_done': False,
        'analyze_data': st.session_state,
        'chat_interface': st.session_state,
        'regenerate': st.session_state,
        'datahandler': None,
        'data': None,
        'table_name': None,
        'chat_history': [],
        'initialEDA':None,
        'data':pd.DataFrame,
        'description':None,
        'questions':[],
        'extra_data':None,
        }
        self.dataframe_placeholder=st.empty
        self.extra_data=None
        self.prompt_data=kwargs['prompt_data']

        for variable, default_value in variables_to_initialize.items():
            if variable not in st.session_state:
                st.session_state[variable] = default_value
        self.config_data=config_data
        self.datahandler=datahandler
        self.config_file=kwargs['config_file']
        self.df_env=dataframe_environment
        

    @st.fragment
    def model_interface_initializer(self):
        colored_header.colored_header("LLM Category",description=None, color_name='blue-green-90')
        llm_category=st.selectbox(label='choose llm category',options=self.config_data["llm_category"], label_visibility='collapsed')
        if llm_category=='gemini models':
            self.supported_llms = self.config_data["supported_llms"]['gemini_llms']
        elif llm_category=='ollama models':
            self.supported_llms=self.config_data["supported_llms"]['opensource_llms']
        elif llm_category=='huggingface models':
            self.supported_llms=self.config_data["supported_llms"]['huggingface_llms']
        elif llm_category=='openai models':
            self.supported_llms=self.config_data["supported_llms"]['openai_llms']
        elif llm_category=='groq models':
            self.supported_llms=self.config_data["supported_llms"]['groq_llms']
        elif llm_category=='antrophic models':
            self.supported_llms=self.config_data["supported_llms"]['antrophic_llms']

        col1, col2= st.columns(2)

        with col1:
            self.selected_llm = st.selectbox("LLM", self.supported_llms)
        with col2:
            self.model_temperature = st.slider('Model Temperature', min_value=0.1, max_value=1.0, value=0.5, step=0.01)

        return llm_category
    

    def _initializer(self, llm_category):
        self.data = self.datahandler.get_data()
        self.df_env.load_data(self.data)
        if self.data is not None:
            show_dataframe=option_menu(None,options=['DATAFRAME VIEW','INTERACTIVE IFRAME VIEW'], orientation="horizontal", styles={"container":{"background":"transparent", "font-size":"0.7rem", "width":"90%", "border":"1px solid black"}, "icon":{"color":"gray"}})
            if 'dataframe' in show_dataframe.lower():
                self.df_env.display(self.data)
            elif 'interactive iframe'  in show_dataframe.lower():
                st.session_state.eda_analyzer.data_interface()
            

        logging.info('Inside initializer')
        self.eda_analyzer = EDAAnalyzer(data=self.data,llm_name=self.selected_llm, table_name=self.datahandler.table_name, temperature=self.model_temperature, config_data=self.config_data, config_file=self.config_file, prompt_data=self.prompt_data, llm_category=llm_category)
        st.session_state.df_env=self.df_env
        st.session_state.datahandler=self.datahandler
        st.session_state.eda_analyzer=self.eda_analyzer
        st.session_state.data=self.data


    @st.fragment
    def create_context(self):
        st.session_state.description=st.text_area('Give brief description about the data')
        question=st.text_input('What questions do you want answered from the analysis?')
        if st.button('add question'):

            st.session_state.questions.append(question)
            for index,question in enumerate(st.session_state.questions):
                st.write(f"{index}: {question}")
        if st.button('load data'):
            self.extra_data={'description':question,'questions':st.session_state.questions}
            st.session_state.extra_data=self.extra_data
            return self.extra_data
        return None

    @st.fragment
    def plot_data(_self, data):
        with st.spinner('Analyzing, Please wait'):

            _self.eda_analyzer.perform_eda()
            colored_header.colored_header("PLOTS",description=None, color_name='blue-green-90')
            _self.eda_analyzer.streamlitplots_numerical(data)
            _self.eda_analyzer.streamlitplots_categorical(data)
            colored_header.colored_header("HEATMAPS",description=None, color_name='blue-green-90')
            with st.expander('Visualize heatmaps'):
                st.write('***Heatmaps are used to visualize the correlation between the features. Identify patterns and outliers. All through color.***')
                _self.eda_analyzer.heatmap_section(_self.data)
            
        with st.spinner('generating eda gpt plots'):
            _self.eda_analyzer.eda_sweetviz(_self.data)
            with open(os.path.join('pages','src','Database','charts','sweetviz.html')) as f:
                html_content = f.read()
            colored_header.colored_header("EDA GPT VISUAL REPORT",description=None, color_name='blue-green-90')
            with st.expander('Analysis Visuals by EDA GPT'):
                components.html(html_content, scrolling=True, height=2000, width=1000)
    
    def statistical_tests(_self,data):
        colored_header.colored_header("Statistical Tests",description=None, color_name='blue-green-90')
        _self.eda_analyzer.statistical_tests()
        st.subheader("Hypothesis Testing")
        with st.expander('Hypothesis Testing'):
            _self.eda_analyzer.hypothesis_testing_display()



    def generate_analysis_report(self, data):
        with st.spinner(text='Generating Report, Please wait'):
            @st.cache_data
            def generate_report(_eda_analyzer,data):

        

                _eda_analyzer.perform_eda()
                if st.session_state.extra_data is not None:
                    logging.info(st.session_state.extra_data)
                    eda_report = _eda_analyzer.EDAAgent(extra_questions=st.session_state.extra_data)
                else:
                    eda_report = _eda_analyzer.EDAAgent()

                return eda_report

            eda_report = generate_report(self.eda_analyzer,data)
            colored_header.colored_header("Analysis Report",description=None, color_name='blue-green-90')

            with st.expander('Analysis'):
                st.write_stream(StringIO(eda_report))
            st.session_state.analysis_done = True
            st.session_state.eda_analyzer = self.eda_analyzer
            st.session_state.initialEDA=eda_report
            st.session_state.extra_data=None

    @st.fragment
    def messagesinterface(self):
        def checkgraphpresence(tupledata):
            for ele in tupledata:
                if isinstance(ele,str) and "graphed" in ele:
                    return True
            return False
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                    
                
                if  message['role']=="user":
                    st.write(message['content'],unsafe_allow_html=True)
                elif message['role']=="assistant":
                    response=message['code']
                    metadatas=message['metadata']

                    if isinstance(metadatas,str) and metadatas.lower()=="sentence":
                        st.write(response)
                    elif checkgraphpresence(metadatas):
                            if message==st.session_state.chat_history[-1]:
                                plots_path = os.path.join('pages','src','Database','Plots')
                                svg_file = os.listdir(plots_path)
                                for index, svgs in enumerate(svg_file):
                                    svg_path = os.path.join(plots_path, svg_file[index])
                                    st.image(svg_path, caption=f"graph {index+1}")

                            for metadata in metadatas:
                                st.write(metadata)
                                # logging.info('\ngraphed\n') 
                            st.code(response)
                       
                    elif not isinstance(metadatas,tuple) and "try again"==metadatas.lower():
                        st.error(metadatas)
                    else:
                        for metadata in metadatas:
                            if isinstance(metadata,pd.DataFrame):
                                st.dataframe(metadata, use_container_width=True)
                            else:
                                st.write(metadata, unsafe_allow_html=True)
                        st.code(response)
                


    @st.fragment
    def chat_interface(self):
        self.messagesinterface()
       
        if user_input := st.chat_input("Talk to EDA GPT:" , key='pandasaichatool'):
            if user_input=='':
                st.warning('Enter Valid Question')
            else:
                # Append user input to chat history
                st.session_state.chat_history.append({"role" : "user" , "content": user_input , "metadata":"user"})
                
                # Get response from analyzer
                response = self.eda_analyzer.pandasaichattool(user_input, st.session_state.initialEDA)
                if response is not None:
                    if len(st.session_state.chat_history)>10:
                        st.session_state.chat_history=st.session_state.chat_history[5:]
                    st.session_state.chat_history.append({"role" : "assistant" , "code": response[1], 'metadata':response[0]})
                    # print('-------------------------------------')
                    # print(response[0])
                    # print('-------------------------------------')
                    # print(response[1])
                    st.rerun()


    # @st.fragment
    # def dataframe_viewer(self):
    #     if self.data is not None:
    #         show_dataframe=option_menu(None,options=['DATAFRAME VIEW','INTERACTIVE IFRAME VIEW'], orientation="horizontal", styles={"container":{"background":"transparent", "font-size":"0.7rem", "width":"90%", "border":"1px solid black"}, "icon":{"color":"gray"}})
    #         if 'dataframe' in show_dataframe.lower():
    #             self.df_env.display(self.data)
    #         elif 'interactive iframe'  in show_dataframe.lower():
    #             st.session_state.eda_analyzer.data_interface()
    
    def workflow(self):
            llm_category=self.model_interface_initializer()
            self._initializer(llm_category=llm_category)
            if st.session_state.data is not None:
                if st.checkbox('Do you want to provided additional context about the data for better analysis?'):
                    self.create_context()
                if st.button('Analyze'):
                        self.plot_data(self.data)
                        self.statistical_tests(self.data)
                        self.generate_analysis_report(self.data)
                        st.session_state.analysis_done = True
                if st.session_state.analysis_done:
                    if st.button('Regenerate result'):
                        self.plot_data(self.data)
                        self.statistical_tests(self.data)
                        # self.generate_analysis_report.clear()
                        self.generate_analysis_report(self.data)
                    if len(st.session_state.chat_history) > 3:
                        st.write("<i style=color:cyan>Beta Feature</i>", unsafe_allow_html=True)
                        if st.toggle('Do you have enough understanding of the data to start feature engineering?'):
                            automl=Automl(self.data)
                            with st.expander("Feature Engineering"):
                                automl.main()
                    self.chat_interface()