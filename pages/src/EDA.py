import numpy as np
import pandas as pd
import re
import os, sys, io
from langchain.prompts import PromptTemplate, ChatPromptTemplate,HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from queue import Queue
import plotly.express as px
import plotly.graph_objects as go
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from bs4 import BeautifulSoup
import logging
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
logging.basicConfig(level=logging.INFO)
import sweetviz as sv
from .vstore import VectorStore
from dotenv import load_dotenv
load_dotenv()
from textwrap import dedent
from joblib import load
from pages.src.Tools.llms import get_llm
import streamlit.components.v1 as stc
from pygwalker.api.streamlit import init_streamlit_comm,StreamlitRenderer
from pages.src.Tools.tasks import structured_tasks
from scipy.stats import shapiro, kstest, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, f_oneway, kruskal
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import sys
from crewai import Agent,Task,Crew

class CodeFormatter(BaseModel):
    """formats llm output into proper code block removing extra text or characters"""
    code:str=Field(description="formatted code without extra leading or trailing characters like '```'. Code Should start with import statement and end with result variable.")

openaifunctions=convert_pydantic_to_openai_function(CodeFormatter)
class EDAAnalyzer:
    def __init__(self, data : pd.DataFrame, table_name : str , llm_name='gemini-pro',**kwargs):
        self.config_data=kwargs['config_data']
        self.config_file=kwargs['config_file']
        self.prompt_data=kwargs['prompt_data']
        self.table_name=table_name
        self.data = data
        self.head=None
        self.df_info = None
        self.summary_statistics = None
        self.missing_values = []
        self.initialEDA : str = None
        self.unique_values = []
        self.skewness=[]
        self.kurtosis=[]
        self.formatted_data=None
        self.correlation_matrix = pd.DataFrame()
        self.value_counts = {}
        self.result_queue=Queue()
        self.vector_store=VectorStore(directory=self.config_data['relational_vstore'])
        # if 'loaded_vstore' not in st.session_state:
        #     st.session_state.loaded_vstore=None
        self.llm_category=kwargs['llm_category']
        init_streamlit_comm()
        # self.initialEDA= None
        # logging.info(self.initialEDA)

        self.formatted_data= None
        self.llm_name=llm_name
        if 'temperature' in kwargs:
            self.temperature=kwargs['temperature']
            logging.info(self.temperature)
        else:
            self.temperature=0.5
            logging.info(self.temperature)

        self.llm=self._get_llm()
        # logging.info("llm data",self.llm_name, self.llm_category, self.llm)
        



        classification_model_path=self.config_data['Classification_models']
        st.session_state.tfidf2=load(os.path.join(classification_model_path,'tfidf2_pretrained.joblib'))
        st.session_state.codedes=load(os.path.join(classification_model_path,'code_decision_pretrained.joblib'))
        
    def _get_llm(self, **kwargs):
        return get_llm(self.llm_name,self.temperature, self.config_data, self.llm_category)

    def perform_eda(self):
        
        # Store basic information about the DataFrame
            missing_values=['?',' ?','? ','NaN', 'N/A', '?', '-', '', ' ', '   ', '  ?', '?  ', '-  ', ' - ', ' -', '? ', '??']
            missing_values_pattern=r'^\s*(nan|na|Na|NaN|Nan|N/A|\?|-)*\s*$'
            def replace_missing_with_nan(value):
                if re.match(missing_values_pattern, str(value)):
                    return np.nan
                return value
            
            self.data=self.data.apply(replace_missing_with_nan)
            self.data.replace(missing_values,np.nan,inplace=True)

            self.df_info = self.data.info()
            self.head=self.data.head()
            
            # Store summary statistics for numerical columns
            self.summary_statistics = self.data.describe().T

            logging.info(self.summary_statistics)
            
            # Store number of missing values per column
            self.missing_values = self.data.isnull().sum()
            logging.info(self.missing_values)
            
            # Store number of unique values per column
            self.unique_values = self.data.nunique()
            
            
            # Store correlation matrix for numerical columns
            numerical_cols = self.data.select_dtypes(include=[np.number, np.int64, np.float64]).columns
            if len(numerical_cols) > 1:
                self.correlation_matrix = self.data[numerical_cols].corr()
                self.skewness=self.data[numerical_cols].skew()
                self.kurtosis=self.data[numerical_cols].kurtosis()
            
            # Store value counts for categorical columns
            categorical_cols = self.data.select_dtypes(exclude=[np.number, np.int64, np.float64] , include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                self.value_counts = {}
                for col in categorical_cols:
                    self.value_counts[col] = self.data[col].value_counts()
            self.fetch_formated_data()


    def fetch_formated_data(self):
        categorical_cols=self.data.select_dtypes(include=['object','category']).columns.to_list()
        if self.missing_values is not None:
            f_missing_values=[(column,self.missing_values[column]) for index,column in enumerate(self.data.columns) if self.missing_values[column]>0]
        if isinstance(self.skewness, (pd.DataFrame, pd.Series)) and len(self.data.select_dtypes(include=[np.number]).columns.to_list())>0:
            logging.info('self.skewness,',self.skewness==None)
            f_skewness=[(column,self.skewness[column]) for column in self.skewness.index if column in self.data.select_dtypes(include=[np.number]).columns.to_list()]
        else:
            f_skewness="none"
        if self.unique_values is not None:
            f_uniqueValues=[(column,self.unique_values[column]) for column in self.unique_values.index if self.unique_values[column]>0]
        if self.value_counts is not None:
            f_valueCounts=[(column,self.value_counts[column]) for column in self.value_counts.keys() if column in categorical_cols and self.value_counts is not None]
        self.formatted_data = f'''skewness of each column is given as follows (column_name , skewness) : {f_skewness}\nmissing_values are as follows (column_name ,missingvalue) : {f_missing_values}\nunique values count in each column is given as follows (column_name , count) : {f_uniqueValues}\nvalue counts in each categorical column (column_name , valuecounts) : {f_valueCounts}\nHere is the correlation matrix of the table : {self.correlation_matrix}\nsummary statistics of the table : {self.summary_statistics}\ndf.info data about the coluumns and it's dtype : {self.df_info}\ncategorical columns : {self.data.select_dtypes(exclude=np.number).columns.to_list()} numerical columns {self.data.select_dtypes(include=[np.number]).columns.to_list()}'''

    def _execute_generated_code(self,generated_code):
        try:
            generated_code = self._format_code(generated_code)
            class CapturePrints:
                def __init__(self):
                    self.outputs = []

                def write(self, msg):
                    if msg.strip():  # Ignore empty messages
                        self.outputs.append(msg.strip())

                def flush(self):
                    pass  # No need to implement this for capturing

            # Override the print function to capture each print statement
            def custom_print(*args, **kwargs):
                output = ' '.join(map(str, args))
                captured_prints.write(output)

            # Setup restricted globals and locals
            restricted_globals = {'df': self.data, 'print': custom_print}
            restricted_locals = {}
            
            # Capture prints
            captured_prints = CapturePrints()
            
            # Execute the generated code
            exec(generated_code, restricted_globals, restricted_locals)
            
            result_value = restricted_locals.get('result')
            
            if not isinstance(result_value, tuple):
                result_value = (result_value,)
            
            # Combine the result variable and captured prints into a single tuple
            answer = result_value + tuple(captured_prints.outputs)

            logging.info(f'formatted_generated_code , answer: {generated_code}, {answer}')
            return (True, answer, generated_code)

        except Exception as e:
            logging.error(f"Error executing generated code: {e}")
            return (False, str(e))



    def handle_single_code_block(self,matches):
        first_part = matches[0]


        pattern = r"result\s*=\s*.*$"
        lastresult = re.findall(pattern, matches[1], re.MULTILINE)[-1]
        redundant_part=(matches[1]).split(lastresult)[-1]
        # print(redundant_part)
        # print('-------------')
        last_part=(matches[1]).replace(redundant_part,'')
        # print(last_part)
        # print('--------')
        formatted_code='\n'.join([first_part,last_part])
        if 'return result' in formatted_code:
            formatted_code.replace('return result','')
        
        return f'''{formatted_code}'''

    def _format_code(self,code : str):

        if 'python' in code:
            code=((code.replace('python','')).strip('```')).strip()
        lines = code.split("\n")
        lines = [line for line in lines if "df = pd" not in line and "df=pd" not in line]
        semi_formatted_code = "\n".join(lines)

        if semi_formatted_code.count('result=')>1 or semi_formatted_code.count('result =')>1:
            return f'''{semi_formatted_code}'''
        if 'import' not in semi_formatted_code and 'result' in semi_formatted_code:
            return f'''{semi_formatted_code}'''
        # print(semi_formatted_code)
        code_block_pattern=r"```(.*?)```"
        code_block_pattern=r"(import[\s\S]*?)(result\s*=\s*[\s\S]+?)(?=(import|result|$))"
        matches=re.findall(code_block_pattern,semi_formatted_code)

        # code_blocks = [block for block in re.findall(code_block_pattern,semi_formatted_code)]
        result=[]
        print(len(matches))
        # print(matches[0])
        for code_blocks in matches:
            result.append(self.handle_single_code_block(code_blocks))
        # return handle_single_code_block(matches=matches[0])

        formatted_code='\n'.join(result)

        
        

        return f'''{formatted_code}'''


    def _promptformatter(self):
        system_variables = ['table']
        human_variables = ['input', 'table_head','column_names', 'context']
        system_prompt_template=self.prompt_data['Structured Prompts']['eda_gpt(pandasai_chattool)']['system_prompt']
        human_prompt_template =self.prompt_data['Structured Prompts']['eda_gpt(pandasai_chattool)']['human_prompt']

        system_message_template = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=system_variables, template=system_prompt_template))

        # Create a new HumanMessagePromptTemplate with the modified prompt
        human_message_template = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=human_variables, template=human_prompt_template))

        # Create a new ChatPromptTemplate with the modified HumanMessagePromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages([system_message_template,human_message_template])
        # logging.info(chat_prompt_template)
        return chat_prompt_template
        
    def _feedback_instructions(self):
        system_prompt_template=self.prompt_data['Structured Prompts']['eda_gpt(pandasai_chattool)']['feedback_llm']
        system_message_template = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['code','error','columns'],template=system_prompt_template))
        chat_prompt_template = ChatPromptTemplate.from_messages([system_message_template])
        return chat_prompt_template



    
    def _texttypedecision(self,sentence) -> str:
        sentence_vectorized=st.session_state.tfidf2.transform([sentence])
        prediction=st.session_state.codedes.predict(sentence_vectorized)
        print(prediction)
        return prediction[0]



    def pandasaichattool(self,query, initialeda):
        logging.info('query',query)
        self._clean_charts()
        query=query+f"NOTE: FOLLOW INSTRUCTIONS/GUIDELINES FOR CODE BLOCK/ SENTENCE GENERATION.Do not return variables as string inside result. df is dataframe."
        try:
            retrieval_qa_chat_prompt = self._promptformatter()
            combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=retrieval_qa_chat_prompt
            )
            # if self.loaded_vstore is None:
            #     vector_embeddings_retriever=(self.vector_store.loadvectorstores()).as_retriever(search_kwargs={'k': 2, 'fetch_k': 10})
            # else:
            try:
                if st.session_state.loaded_vstore is not None:
                    vector_embeddings_retriever=st.session_state.loaded_vstore.as_retriever(search_kwargs={'k': 2})
            except Exception as e:
                raise e
            retrieval_chain = create_retrieval_chain(vector_embeddings_retriever, combine_docs_chain)

            valid_code_generated = False
            attempts = 0
            max_attempts = 5
            temp_query=query
            while not valid_code_generated and attempts <= max_attempts:
                if attempts>2:
                    feedback_llm=SmartLLMChain(prompt=self._feedback_instructions(), llm=self.llm, verbose=True, n_ideas=1)
                    code=feedback_llm.invoke({'input':temp_query,'code':wrong_code,'error':error_code, 'columns':self.data.columns})['resolution']
                else:
                    code = retrieval_chain.invoke({'input':temp_query,'table':self.table_name,'table_head':self.data.head()[0:4], 'column_names':self.data.columns})['answer']
               
                

                logging.info("generated code",code)


                
                
                if 'sentence' in self._texttypedecision(code).lower():
                    if 'result=' in code:
                        formatcode_to_sentence=((code.split('result=')[-1]).strip("'")).replace('result=','')
                        return ("Sentence",[code if formatcode_to_sentence is None else formatcode_to_sentence][0])
                    elif 'result =' in code:
                        formatcode_to_sentence=((code.split('result =')[-1]).strip("'")).replace('result =','')
                        return ("Sentence",[code if formatcode_to_sentence is None else formatcode_to_sentence][0])

                
                    logging.info('type is sentence')
                    return ("Sentence",code)
                    
                
                
                
                #format code
                # code=self._format_code(code)
                #execute code
                response = self._execute_generated_code(code)
                # logging.info(response)
                if response:
                    if response[0] is True:
                        valid_code_generated = True
                        return (response[1],response[2])
                    elif not response[0] or response[1]==None:
                        attempts += 1
                        temp_query = f"\nYOU ARE AN ADVANCED CODE CORRECTOR AI, YOU RETURN ONLY SINGLE CORRECTED CODE BLOCK FROM GIVEN ERROR AND CODE WITHOUT ADDITIONAL TEXT. HERE IS WRONG CODE THAT YOU GENERATED PREVIOUSLY : {code}\n with error : {response[1]}\n"
                        wrong_code=code
                        error_code=response[1]

            return ('Try Again','No Code Generated')
        except Exception as e:
            raise e
        

    def EDAAgent(self, extra_questions=None):
        template = self.prompt_data['Structured Prompts']['eda_gpt_analysis']['role']
        if extra_questions is not None:
            descriptiondata=f"Here is a brief qualitative overview about the data (metadata)--->{extra_questions['description']}"
            questions="Here is a list of questions that the user needs answer of"+"\n".join(extra_questions['questions'])
            template=f"{template}\n{descriptiondata}"
        else:
            questions=""
            descriptiondata=""

        
        EdaAGENT = Agent(
            role=template,
            backstory=f'''You can analyze complex and large volumes of relational data.''',
            verbose=True,
            allow_delegation=False,
            memory=True,
            llm = self.llm,
            goal=dedent('come up with extensive analysis report from given dataframe and metadata after analyzing it thoroughly. You have been given calculator if you need to calculate.')
            )

        task = structured_tasks(agent=EdaAGENT,table_name=self.table_name,formatted_data=self.formatted_data,questions=questions)
        crew = Crew(
        agents=[EdaAGENT],
        tasks=task,
        verbose=0,
        
        )
        with st.spinner('Generating analysis report takes some time. Please have a ☕ break...'):
            crew.kickoff()
        result= "\n\n".join([tasks.output.raw_output for tasks in task])

        self.initialEDA=result
        with st.spinner('Almost Done...'):
            with open(os.path.join(self.config_data['relational_vstore'],"EDAanalysis.txt"),'w', encoding='utf-8') as f:
                f.write(result)
            self.vector_store.directory=self.config_data['relational_vstore']
            st.session_state.loaded_vstore=self.vector_store.makevectorembeddings(embedding_num=st.session_state.embeddings)
            logging.info(st.session_state.loaded_vstore)

        return result

    def _save_edadata(self, eda):
        file_path=os.path.join('pages','src','Database','userdata','eda.txt')
        with open(file_path,'w') as edafile:
            edafile.write("Table Name: {} \n".format(self.table_name))
            edafile.write("EDA Description: {} \n".format(self.initialEDA))
            edafile.write("----------------------------------------------------------------")
    
    def _clean_charts(self):
        files = os.listdir(os.path.join('pages','src','Database','Plots'))
        if files:
            for file in files:
                os.remove(os.path.join('pages','src','Database','Plots',file))
    
    def _clear_EDA_and_chats(self):
        folder=os.path.join('pages','src','Database','userdata')
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'w') as file:
                pass
    
    def saveNdownload_eda(self, eda):
        # Inject additional data into the HTML report
        self._save_edadata(eda)

    
    @st.cache_data
    def eda_sweetviz(_self, data):
        report = sv.analyze(_self.data)       
        html_path = _self.config_data["html_charts_sweet"]
        files = os.listdir(html_path)
        for file in files:
            os.remove(os.path.join(html_path, file))
        report_html_path = os.path.join(html_path, "sweetviz.html")
        report.show_html(report_html_path, open_browser=False)
        
        with open(report_html_path, 'r') as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        specific_div = soup.find('div', {'class': 'pos-logo-group'})
        if specific_div:
            specific_div.decompose()
        
        new_div = soup.new_tag('div', attrs={'class': 'EDA_Agent'})
        
        h1_tag=soup.new_tag('h1')
        h1_tag.string='Welcome to Visuals with EDA GPT'
        new_div.append(h1_tag)
        soup.body.insert(0, new_div)
        with open(report_html_path, 'w', encoding='utf-8') as file:
            file.write(str(soup))

    
    @st.fragment
    def streamlitplots_numerical(_self, data):

        # Check if the dataframe is empty
        if _self.data.empty:
            st.write("DataFrame is empty.")
            return

        # Plotting numerical data
        st.subheader("Numerical Data Analysis")
        numerical_columns = _self.data.select_dtypes(include='number').columns.tolist()

        if len(numerical_columns) < 3:
            st.write('No numerical plots present for numerical columns less than 3')
        else:
            selected_columns = st.multiselect("Select numerical columns to visualize", numerical_columns, default=numerical_columns[:1])

            for column in selected_columns:
                if column in numerical_columns:

                    with st.expander(f"{column} Analysis"):
                        # Histogram
                        st.write("Histogram:")
                        bins = st.slider(f"Number of bins for {column} histogram", min_value=10, max_value=100, value=50)
                        fig = px.histogram(_self.data, x=column, nbins=bins)
                        st.plotly_chart(fig)

                        # Box Plot
                        st.write("Box Plot:")
                        fig = px.box(_self.data, y=column)
                        st.plotly_chart(fig)

            # Scatter Plot
            with st.expander('Scatter Plot'):
                st.subheader("Scatter Plot:")
                scatter_columns = st.multiselect("Select columns for scatter plot matrix", selected_columns, default=selected_columns)
                if scatter_columns:
                    fig = px.scatter_matrix(_self.data[scatter_columns])
                    st.plotly_chart(fig)

            # Pairplot for numerical features
            with st.expander('Pairplot'):
                st.subheader("Pairplot for Numerical Features")
                pairplot_columns = st.multiselect("Select columns for pairplot", _self.data.columns.to_list(), default=numerical_columns)
                if pairplot_columns:
                    sns_pairplot = sns.pairplot(_self.data[pairplot_columns], diag_kind='kde')
                    st.pyplot(sns_pairplot)



    @st.fragment
    def streamlitplots_categorical(_self, data):

        # Check if the dataframe is empty
        if _self.data.empty:
            st.write("DataFrame is empty.")
            return
        else:

            # Plotting categorical data
            st.subheader("Categorical Data Analysis")
            categorical_columns = _self.data.select_dtypes(include=['object', 'category']).columns.tolist()

            if len(categorical_columns) < 1:
                st.warning('Less than one categorical column is present. No analysis done for this.')
            else:
                selected_cat_columns = st.multiselect("Select categorical columns to visualize", categorical_columns, default=categorical_columns[:1])

                for column in selected_cat_columns:
                    if column in categorical_columns:

                        if _self.data[column].nunique() <= 1000000:  # Limit to 1000000 unique categories for better visualization
                            with st.expander(f"{column} Analysis"):
                                # Count Plot
                                st.write("Count Plot:")
                                fig = px.bar(_self.data, y=_self.data[column].value_counts().values, x=_self.data[column].value_counts().index, labels={'x': column, 'y': 'Count'})
                                st.plotly_chart(fig)

                                # Pie Chart
                                st.write("Pie Chart:")
                                fig = px.pie(values=_self.data[column].value_counts().values, names=_self.data[column].value_counts().index)
                                st.plotly_chart(fig)




    @st.fragment
    def statistical_tests(self):
        st.subheader("Normality tests")
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        with st.expander('Kolmogorov-Smirnov Test And Shapiro-Wilk Test'):
            results = []
            for col in numerical_columns:
                data = self.data[col].dropna()  # Remove NaN values
                print(data)
                if len(data) < 3:
                    results.append({
                        'Column': col,
                        'Shapiro-Wilk Stat': 'N/A',
                        'Shapiro-Wilk p-value': 'N/A',
                        'K-S Stat': 'N/A',
                        'K-S p-value': 'N/A',
                        'Notes': 'Insufficient data'
                    })
                    continue

                # Shapiro-Wilk test
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                except Exception as e:
                    shapiro_stat, shapiro_p = 'Error', str(e)

                # Kolmogorov-Smirnov test
                try:
                    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
                except Exception as e:
                    ks_stat, ks_p = 'Error', str(e)

                # Format p-values
                shapiro_p_formatted = f"{shapiro_p:.2e}" if isinstance(shapiro_p, float) else shapiro_p
                ks_p_formatted = f"{ks_p:.2e}" if isinstance(ks_p, float) else ks_p

                results.append({
                    'Column': col,
                    'Shapiro-Wilk Stat': shapiro_stat if isinstance(shapiro_stat, float) else 'N/A',
                    'Shapiro-Wilk p-value': shapiro_p_formatted,
                    'K-S Stat': ks_stat if isinstance(ks_stat, float) else 'N/A',
                    'K-S p-value': ks_p_formatted,
                    'Sample Size': len(data),
                    'Notes': 'Large sample size, interpret with caution' if len(data) > 5000 else ''
                })

            results_df = pd.DataFrame(results)
            st.write("Normality Test Results:")
            st.dataframe(results_df)

            st.write("""
            Note:
            - 'N/A' indicates that the test could not be performed, usually due to insufficient data.
            - Very small p-values are displayed in scientific notation (e.g., 1e-10).
            - For large sample sizes (>5000), even small deviations from normality can result in very small p-values.
            - Interpret results with caution, especially for large datasets.
            """)

    @st.fragment
    def heatmap_section(self,data):
        def heatmap_section_cache(data):
            all_columns = data.columns.tolist()

            # Allow user to select columns to include in the heatmap
            selected_columns = st.multiselect("Select columns for heatmap", all_columns, default=all_columns[:2])

            # Filter dataframe based on selected columns
            filtered_df = data[selected_columns]

            # Encode categorical columns
            df_encoded = pd.get_dummies(filtered_df, drop_first=True)

            # Combine numerical and encoded categorical columns
            combined_df = pd.concat([filtered_df.select_dtypes(include='number'), df_encoded], axis=1)
            return combined_df
        
        
        
        
        # Generate correlation matrix
        select_style=st.selectbox('Color Scales',options=['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'],
        )
        corr_matrix = heatmap_section_cache(data).corr()
        fig = px.imshow(corr_matrix,
                        labels=dict(x="Features", y="Features", color="Correlation"),
                        color_continuous_scale=select_style,
                        zmin=-1, zmax=1)

        fig.update_layout(title='Customizable Heatmap for Numerical and Categorical Data')
        st.plotly_chart(fig)

    @st.fragment
    def data_interface(self):
        
        st.subheader('EDA Playgroud', help='Analyze data using drag and drop tools before putting ai to work.')
        if "pygwalker_spec" not in st.session_state:
            st.session_state.pygwalker_spec=""
            st.session_state.pygwalker_table=""
        else:
            st.session_state.pygwalker_spec=self.config_data['pygwalker_config']
            st.session_state.pygwalker_table=self.config_data['pygwalker_table']


       
        
        pyg_app_renderer=StreamlitRenderer(self.data, spec_io_mode='rw' , spec=[st.session_state.pygwalker_spec if st.session_state.pygwalker_spec is not None and st.session_state.pygwalker_table==self.table_name else ''][0])
        size=st.slider('width-height',min_value=500,max_value=1300)
        pyg_app_renderer.explorer(width=size,height=size, scrolling=True)


    @st.fragment
    def hypothesis_test(self,data, columns, null_hypothesis, alternate_hypothesis, alpha):
        result = {}
        
        if len(columns) == 2:
            col1, col2 = columns
            if pd.api.types.is_numeric_dtype(data[col1]) and pd.api.types.is_numeric_dtype(data[col2]):
                # Two numerical columns: t-test
                stat, p_value = ttest_ind(data[col1], data[col2])
                result['test'] = 't-test'
            elif pd.api.types.is_categorical_dtype(data[col1]) and pd.api.types.is_categorical_dtype(data[col2]):
                # Two categorical columns: chi-square test
                contingency_table = pd.crosstab(data[col1], data[col2])
                stat, p_value, dof, expected = chi2_contingency(contingency_table)
                result['test'] = 'chi-square test'
            elif (pd.api.types.is_categorical_dtype(data[col1]) and pd.api.types.is_numeric_dtype(data[col2])) or (pd.api.types.is_numeric_dtype(data[col1]) and pd.api.types.is_categorical_dtype(data[col2])):
                # One categorical and one numerical column: ANOVA
                if pd.api.types.is_categorical_dtype(data[col1]):
                    formula = f'{col2} ~ C({col1})'
                else:
                    formula = f'{col1} ~ C({col2})'
                model = ols(formula, data=data).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                stat = anova_table['F'][0]
                p_value = anova_table['PR(>F)'][0]
                result['test'] = 'ANOVA'
            else:
                raise ValueError("Unsupported combination of column types for hypothesis testing.")
        elif len(columns) > 2:
            # Multiple numerical columns: ANOVA
            formula = f'{columns[-1]} ~ ' + ' + '.join([f'C({col})' if pd.api.types.is_categorical_dtype(data[col]) else col for col in columns[:-1]])
            model = ols(formula, data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            stat = anova_table['F'][0]
            p_value = anova_table['PR(>F)'][0]
            result['test'] = 'ANOVA'
        else:
            raise ValueError("Unsupported number of columns for hypothesis testing.")
        
        result['statistic'] = stat
        result['p_value'] = p_value
        result['null_hypothesis'] = null_hypothesis
        result['alternate_hypothesis'] = alternate_hypothesis
        result['alpha'] = alpha
        result['reject_null'] = p_value < alpha
        
        return result

    def display_hypothesis_test_result(self,result):
        st.write("Hypothesis Testing Result:")
        st.write(f"Test: {result['test']}")
        st.write(f"Statistic: {result['statistic']}")
        st.write(f"P-value: {result['p_value']}")
        st.write(f"Null Hypothesis: {result['null_hypothesis']}")
        st.write(f"Alternate Hypothesis: {result['alternate_hypothesis']}")
        st.write(f"Alpha: {result['alpha']}")
        if result['reject_null']:
            st.write("Conclusion: Reject the null hypothesis")
        else:
            st.write("Conclusion: Fail to reject the null hypothesis")

    @st.fragment
    def hypothesis_testing_display(self):
            # Hypothesis Testing Inputs
            st.header("Hypothesis Testing")
            columns_for_testing = st.multiselect("Columns for Hypothesis Testing", self.data.columns)
            null_hypothesis = st.text_input("Null Hypothesis")
            alternate_hypothesis = st.text_input("Alternate Hypothesis")
            alpha = st.slider("Alpha", min_value=0.001, max_value=0.1)
            
            if st.button("Run Hypothesis Test"):
                try:
                    result = self.hypothesis_test(self.data, columns_for_testing, null_hypothesis, alternate_hypothesis, alpha)
                    self.display_hypothesis_test_result(result)
                except ValueError as e:
                    st.error(f"Error: {e}")