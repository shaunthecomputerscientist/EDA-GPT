import streamlit as st
import json, os
from datetime import datetime
from .vstore import VectorStore
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate,PromptTemplate
import sys , re
# sys.path.insert(0,'/vendor/dependencies/crewAI')
# from vendor.dependencies.crewAI.crewai.agent import Agent
# from vendor.dependencies.crewAI.crewai.task import Task
# from vendor.dependencies.crewAI.crewai.crew import Crew
# sys.path.pop(0)
from crewai import Agent,Task,Crew
from langchain.tools import tool
import os, inspect, types
from streamlit_extras import colored_header
from pages.src.Tools.llms import get_llm
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from streamlit_extras.dataframe_explorer import dataframe_explorer
from pages.src.Tools.langroupchain_custom import LangGroupChain
import pandas as pd
import assemblyai as aai
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pymongo
from streamlit_option_menu import option_menu
from bson import ObjectId
from .Tools.tools import (
    SEARCH_API,
    Scraper,
    Vision,
    arxiv,
    wikipedia,
    datetimee,
    YoutubeVideoTranscript
)
from textwrap import dedent
from joblib import load
from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)


#tools initialization
@tool('Analyzer', return_direct=False)
def datanalystbotwrapper(query:str):
    '''Analyzes unstructured pdf data based on user question. User question is fed into query variable.'''
    result=st.session_state.unstructured_analyzer.datanalystbot(query=query)
    return result

@tool('Vision')
def vision(query:str):
    """This tool answers questions based on the current screen. But remember the context of the chat to answer follow up questions. Provide question to this helper agent to get answer."""
    vision=Vision(model=st.session_state.vision_model)
    return vision.vision(query=query)

@tool("Search Agent")    
def SearchAgent(query):
    """Searches internet for answers"""
    lgchain=LangGroupChain(llm=st.session_state.unstructured_analyzer._get_llm(), tools=[SEARCH_API, Scraper, datanalystbotwrapper, arxiv, wikipedia])
    return lgchain.generate_response3(query)
    

class unstructured_Analyzer:
    def __init__(self, config_data, prompt_data):

        self.config_data=config_data
        self.prompt_data=prompt_data       
        self.unstructured_directory=self.config_data['unstructured_data']
        self.image_path=self.config_data['QnA_img']
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store=VectorStore()
        st.session_state.can_upload=True
        if "messages" not in st.session_state:
            st.session_state['messages']=[]
        if 'docs_found' not in st.session_state:
            st.session_state['docs_found']=[]
        if 'internet_access' not in st.session_state:
            st.session_state['internet_access']=False
        if 'LangGroupChain' not in st.session_state:
            st.session_state['LangGroupChain']=False

        classification_model_path=self.config_data['Classification_models']
        st.session_state.tfidf=load(os.path.join(classification_model_path,'tfidf_pretrained.joblib'))
        st.session_state.rf=load(os.path.join(classification_model_path,'randomtree_decision_pretrained.joblib'))
        st.session_state.response_sentiment=load(os.path.join(classification_model_path,'response_sentiment.joblib'))
        


    def _upload_pdf(self):
        uploaded_files = st.file_uploader("Upload pdf file", type=["pdf", "mp3", "mp4", 'mpeg'],accept_multiple_files=False, key=1, label_visibility='hidden')
        if uploaded_files:
            logging.info('file :',uploaded_files, uploaded_files.type)
        
        st.session_state['uploaded_files']=uploaded_files
        return uploaded_files
    def _upload_image(self):
        # for file in os.listdir(self.image_path):
        #     os.remove(os.path.join(self.image_path,file))

        if 'vision_model' not in st.session_state:
            st.session_state['vision_model']=''

        st.session_state.vision_model=st.selectbox('Select LVM',self.config_data['supported_llms']['vision_models'])

        if st.session_state.vision_model:
            uploaded_image=st.file_uploader("Upload an image",type=["png", "jpg" , "jpeg"],key=2, accept_multiple_files=False, label_visibility="hidden")
            logging.info("name :",uploaded_image)
        return uploaded_image

    def _IsGenerator(self, obj):
        return inspect.isgeneratorfunction(obj) or isinstance(obj, types.GeneratorType)
    def _decision(self,sentence):
        sentence_vectorized=st.session_state.tfidf.transform([sentence])
        prediction=st.session_state.rf.predict(sentence_vectorized)
        return prediction[0]
    def response_sentiment(self,response):
        vectorizer=load(os.path.join(self.config_data['Classification_models'],'response_sentiment_vectorizer_pretrained.joblib'))
        response_vectorized=vectorizer.transform([response])
        prediction_proba=st.session_state.response_sentiment.predict_proba(response_vectorized)[0][0]
        logging.info(prediction_proba)
        return prediction_proba



    @st.cache_resource
    def _vstore_embeddings(_self, uploaded_files=None, mongo=False, _mongo_data=None):
        if 'vectorstoreretriever' not in st.session_state:
            st.session_state['vectorstoreretriever']=None
        for file in os.listdir(_self.unstructured_directory):
            os.remove(os.path.join(_self.unstructured_directory,file))
        if uploaded_files:
            if uploaded_files.type.split('/')[1] in ['pdf']:
                with open(os.path.join(_self.unstructured_directory, uploaded_files.name), 'wb') as f:
                    f.write(uploaded_files.getbuffer())
                    # logging.info('saved pdf')
            elif uploaded_files.type.split('/')[1] in ['mp3', 'mp4', 'mpeg4', 'mpeg']:
                    logging.info('audio file')
                    aai.settings.api_key = st.secrets['ASSEMBLYAI_API_KEY']['api_key']
                    with st.spinner('collecting transcripts...'):
                        audio_dir = _self.config_data['audio_dir']
                        audio_file_path = os.path.join(audio_dir, uploaded_files.name)
                        with open(audio_file_path, "wb") as f:
                            f.write(uploaded_files.getbuffer())
                        transcriber = aai.Transcriber()
                        transcript = transcriber.transcribe(audio_file_path)
                        
                    # Save the transcript to a text file
                    with open(os.path.join(_self.unstructured_directory, 'transcript.txt'), 'w') as f:
                        logging.info(transcript.text)
                        f.write(transcript.text)
                    for ele in os.listdir(audio_dir):
                        os.remove(os.path.join(audio_dir,ele))
        # logging.info('saved transcript')
                    # logging.info('saved transcript')
            with st.spinner('Generating Embeddings. May take some time...'):
                st.session_state.vectorstoreretriever=st.session_state.vector_store.makevectorembeddings(embedding_num=st.session_state.embeddings)
        elif mongo:
            file_path = os.path.join(_self.unstructured_directory, 'mongo_data.txt')

            # Write the JSON data to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(str(_mongo_data))
                file.write('\n')
            with st.spinner('Generating Embeddings. Wait for sometime...'):
                st.session_state.vectorstoreretriever=st.session_state.vector_store.makevectorembeddings(embedding_num=st.session_state.embeddings,key="mongo")
        return st.session_state.vectorstoreretriever
                    

        
    def check_for_url(self,text):
        pattern=r'https://\S+'
        matches=re.findall(pattern,text)
        if len(matches)>0:
            return True
        else:
            return False
    def _promptformatter(self):
        input_variables = ['context', 'input' , 'memory','extra_documents', 'date']
        variables = """\nQUESTION: {input},\n
        Retrieved CONTEXT: {context},\n
        Retrieved DOCS: {extra_documents},\n
        MEMORY: {memory}\n
        DATE & TIME '%Y-%m-%d %H:%M:%S' : {date}\n
        Answer:"""
        template = '\n'.join([self.prompt_data['Unstructured Prompts']['analystool']['prompt'],variables])
        # Create a new HumanMessagePromptTemplate with the modified prompt
        human_message_template = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=input_variables, template=template))

        # Create a new ChatPromptTemplate with the modified HumanMessagePromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages([human_message_template])
        # logging.info(chat_prompt_template)
        return chat_prompt_template
    
    def _get_llm(self, **kwargs):
        return get_llm(st.session_state.selected_llm,st.session_state.model_temperature, self.config_data, self.llm_category)
    
   
    def datanalystbot(self, query : str, context=" "):
        llm=self._get_llm()
        combine_docs_chain = create_stuff_documents_chain(
        llm=llm, prompt=self._promptformatter()
        )
        if st.session_state.vectorstoreretriever is None:
            vector_embeddings_retriever=(self._vstore_embeddings(uploaded_files=st.session_state['uploaded_files']))[0].as_retriever(search_kwargs={'k' : 5})
            st.write(st.session_state.vectorstoreretriever)
            
            
        else:
            vector_embeddings_retriever=st.session_state.vectorstoreretriever[0].as_retriever(search_kwargs={'k' : 5})
        
        st.write(st.session_state.vectorstoreretriever)

        
        #docs

        multi_query_retriever_from_llm = MultiQueryRetriever.from_llm(retriever=st.session_state.vectorstoreretriever[0].as_retriever(search_kwargs={'k' : 5}), llm=llm, prompt=PromptTemplate(input_variables=[], template=self.prompt_data['Unstructured Prompts']['analystool']['mqr_prompt']+f"\nDescription : {context}"+f"\nuser question : {query}"), include_original=True)
        # queries=multi_query_retriever_from_llm.generate_queries(query)
        # filtered_queries=[ele for ele in queries if ele!='']
        # mqdocs=multi_query_retriever_from_llm.retrieve_documents(filtered_queries)
        try:
            with st.spinner('extracting documents (multiquery)'):
                mqdocs=multi_query_retriever_from_llm.invoke(query)
                if len(mqdocs)>5:
                    mqdocs=mqdocs[:5]
        except Exception as e:
            mqdocs=[Document('')]
        #--------------------------------------------------------------
        # with st.expander("mqdocs"):
        #     st.write(mqdocs)
        ensemble_retriever = EnsembleRetriever(retrievers=[st.session_state.vectorstoreretriever[1], vector_embeddings_retriever], weights=[0.7, 0.3])
        with st.spinner('extracting documents (ensemble retriever)'):
            ensemble_docs=ensemble_retriever.invoke(input=query)
            if len(ensemble_docs)>=5:
                ensemble_docs=ensemble_docs[:5]
        #--------------------------------------------------------------
        with st.spinner('searching for docs with high similarity threshold (0.7)'):
            extra_data=(st.session_state.vectorstoreretriever)[0].as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold': 0.7, 'k' : 5}).invoke(input=query)
            extradata=''
            for ele in extra_data:
                extradata+=extradata+ele.page_content+'\n'
        #--------------------------------------------------------------
        logging.info('mqdocs',len(mqdocs))
        logging.info('emsemble docs',len(ensemble_docs))
        logging.info('extradata',len(extra_data))
        #--------------------------------------------------------------
        with st.spinner('extracting relevant documents'):
            combinedocs=mqdocs+[Document(page_content=extradata)]+ensemble_docs
            retrieval_chain = create_retrieval_chain(vector_embeddings_retriever, combine_docs_chain)
        with st.spinner('generating final answer'):
            try:
                result=retrieval_chain.invoke({'input':query,'context': [context if context is not None else " "][0], 'memory':st.session_state.messages[::-1][0:int([3 if len(st.session_state.messages)>3 else len(st.session_state.messages)][0])], 'date':datetime.today().strftime('%Y-%m-%d %H:%M:%S'), 'extra_documents':combinedocs})
            except Exception as e:
                return e
        #--------------------------------------------------------------
        logging.info(result)
        st.session_state.docs_found=result['context']+combinedocs
        return result['answer']

    def Multimodalagent(self, query):

        Multimodal = Agent(
            role=dedent(f"{self.prompt_data['Unstructured Prompts']['crewaiagent']['role']} DATE TIME {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"),
            backstory=dedent(self.prompt_data['Unstructured Prompts']['crewaiagent']['backstory']),
            verbose=True,
            allow_delegation=False,
            memory=True,
            max_iter=7,
            llm = self._get_llm(),
            goal=dedent(self.prompt_data['Unstructured Prompts']['crewaiagent']['goal']),
            tools=[vision, datanalystbotwrapper,SEARCH_API, Scraper, arxiv, wikipedia, datetimee,YoutubeVideoTranscript]
            )
        task = Task(
            description=dedent(f"user question: {query}\n\n INSTRUCTIONS : {self.prompt_data['Unstructured Prompts']['crewaiagent']['description']}\n\n CONVERSATION HISTORY : {st.session_state.messages[::-1][0:int([3 if len(st.session_state.messages)>3 else len(st.session_state.messages)][0])]}"),
            agent=Multimodal,
            async_execution=False,
            expected_output=dedent(self.prompt_data['Unstructured Prompts']['crewaiagent']['expected_output']),
            result=dedent(self.prompt_data['Unstructured Prompts']['crewaiagent']['result']),
            )
        crew = Crew(
        agents=[Multimodal],
        tasks=[task],
        verbose=0,
        )

        try:
        
            result = crew.kickoff()
            return result
        
        except Exception:

            return "Try again"

    


    def fetch_mongodb_data(self, uri, database_name, collection_name):
        client = pymongo.MongoClient(uri)
        db = client[database_name]
        collection = db[collection_name]
        docs = collection.find()
        
        data = []
        for doc in docs:
            # Extract all keys dynamically from the document
            doc_data = {"_id": str(doc["_id"])}
            for key, value in doc.items():
                doc_data[key] = value
            data.append(doc_data)
        
        return data
    
    
    @st.experimental_fragment
    def mongoviewer(self):
        uri = st.text_input("MongoDB URI", "mongodb://localhost:27017/")
        database_name = st.text_input("Database Name")
        collection_name = st.text_input("Collection Name")

        if st.button("Fetch Data and generate embeddings"):
            if uri and database_name and collection_name:
                    data = self.fetch_mongodb_data(uri, database_name, collection_name)
                    logging.info('generating', type(data))

                    vstore=self._vstore_embeddings(mongo=True,_mongo_data=data)
                    st.session_state.vectorstoreretriever=vstore
                    st.success(f'Embeddings generated {st.session_state.vectorstoreretriever}')
                    if data:
                        st.subheader(f"Number of docs is {len(data)}. Here is preview!")
                        for doc in [data if len(data)<20 else data[0]]:
                            st.write(doc)
                    
                    
                    else:
                        st.write("No documents found.")
                        
                        
            else:
                st.warning("Please enter all the required credentials.")

            

        
    


    
    def sidebarcomponents(self):
        if st.session_state.internet_access:
            with st.sidebar.title('Analyze images'):
                    with st.expander('Upload Images'):
                        st.session_state.uploaded_image=self._upload_image()

                        if st.session_state.uploaded_image is not None and st.session_state.uploaded_image!=[]:

                            logging.info('image uploader')
                            for ele in os.listdir(self.image_path):
                                os.remove(os.path.join(self.image_path,ele))

                            with open(os.path.join(self.image_path,(st.session_state.uploaded_image).name), 'wb') as file:
                                file.write(st.session_state.uploaded_image.read())

        if st.session_state.vectorstoreretriever is not None:
            with st.sidebar.title('Description'):
                    if 'description' not in st.session_state:
                        st.session_state.description=' '
                    
                    with st.popover('Describe',help='provide description about the data provided to aid better retrieval.', use_container_width=True):
                        st.session_state.description=st.text_area('Describe Data')

    def generateresponse(self, prompt):
        predict=self._decision(prompt)
        url_check=self.check_for_url(prompt)
        if url_check:
            predict="search"
        st.session_state.messages.append({'role':'user','content':prompt})
        

        if st.session_state.internet_access:

            if 'analysis' in predict:
                with st.spinner('your anawers will be ready soon......'):
                    message=self.datanalystbot(prompt, st.session_state.description)
                    if self.response_sentiment(message)>0.40:
                        with st.spinner('This might take a while...'):
                            st.session_state.docs_found=[]
                            if st.session_state.LangGroupChain:
                                message=SearchAgent(prompt)
                            else:
                                message : str=self.Multimodalagent(prompt)


            else:
                if 'vision' in predict and st.session_state.uploaded_image:
                    st.session_state.docs_found=[]
                    if st.session_state.uploaded_image:
                        st.image(st.session_state.uploaded_image)
                    with st.spinner('analyzing picture...'):
                        message : str=vision(prompt)
                else:
                    st.session_state.docs_found=[]
                    with st.spinner('Searching...'):
                        if st.session_state.LangGroupChain:
                            message=SearchAgent(prompt)
                        else:
                            message : str=self.Multimodalagent(prompt)
        else:
            with st.spinner('generating answer...'):
                message=self.datanalystbot(prompt, st.session_state.description)


        return message


    def workflow(self):
        
        colored_header.colored_header("Supported LLMs","choose your llm", color_name='blue-green-90')
        self.llm_category=st.selectbox(label='choose llm category',options=self.config_data["llm_category"], label_visibility='collapsed')
        if self.llm_category=='gemini models':
            self.supported_llms = self.config_data["supported_llms"]['gemini_llms']
        elif self.llm_category=='ollama models':
            self.supported_llms=self.config_data["supported_llms"]['opensource_llms']
        elif self.llm_category=='huggingface models':
            self.supported_llms=self.config_data["supported_llms"]['huggingface_llms']
        elif self.llm_category=='openai models':
            self.supported_llms=self.config_data["supported_llms"]['openai_llms']
        elif self.llm_category=='groq models':
            self.supported_llms=self.config_data["supported_llms"]['groq_llms']
        elif self.llm_category=='antrophic models':
            self.supported_llms=self.config_data["supported_llms"]['antrophic_llms']
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.selected_llm = st.selectbox("LLMS", self.supported_llms)
        with col2:
            st.session_state.model_temperature = st.slider('Model Temperatures', min_value=0.1, max_value=1.0, value=0.5, step=0.01)

        if st.toggle('Activate Internet Access',help="This will enable llm to search internet for queries"):
            st.session_state.internet_access=True
        else:
            st.session_state.internet_access=False

        if st.toggle('Activate LangGroupChain Experimental?',help="Works with internet accesss.Experimental chain that breaks down a problem into graph, analyzes subproblems in topological order and then responds. Might be unstable for some llms. Currently supports only gemini models."):
            st.session_state.LangGroupChain=True
        else:
            st.session_state.LangGroupChain=False

        
        select_upload_option=option_menu(None,['Upload pdf/audio','MONGO DB'], orientation="horizontal")
        if select_upload_option=='MONGO DB':
            self.mongoviewer()
        elif select_upload_option=='Upload pdf/audio':
            # st.session_state.messages=[]
            files=self._upload_pdf()
            st.session_state.vectorstoreretriever=self._vstore_embeddings(uploaded_files=files)
            # st.write(self._vstore_embeddings(uploaded_files=files),files)
            # if st.session_state['uploaded_files'] is not None and st.session_state.vectorstoreretriever is None:

            #     self._vstore_embeddings(uploaded_files=st.session_state['uploaded_files'])
        st.write(st.session_state.vectorstoreretriever)
        if st.session_state.vectorstoreretriever is not None:
                for message in st.session_state.messages:
                    with st.chat_message(message['role']):
                        if self._IsGenerator(message['content']):
                            st.write(message['content'], unsafe_allow_html=True)
                        elif isinstance(message['content'],pd.DataFrame):
                            st.dataframe(message['content'])

                        else:
                            st.write(message['content'], unsafe_allow_html=True)
                if st.session_state.docs_found:
                    with st.expander('retrived docs'):
                        st.write(st.session_state.docs_found, unsafe_allow_html=True)
                            
                if prompt := st.chat_input('Ask questions', key='data_chat'):
                    logging.info(prompt)
                    message=self.generateresponse(prompt=prompt)
                    st.session_state.messages.append({'role':'assistant','content':message})
                    logging.info(st.session_state.messages)
                    st.rerun()
            




    def run(self):
        self.workflow()
        self.sidebarcomponents()





            

            

                

            
