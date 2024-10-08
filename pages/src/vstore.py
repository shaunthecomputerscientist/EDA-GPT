from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
# from langchain_community.embeddings.voyageai import VoyageEmbeddings
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.retrievers import BM25Retriever
from unstructured.partition.auto import partition_pdf
import string
import time
from concurrent.futures import ThreadPoolExecutor
import nltk
nltk.download("stopwords")
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langchain_experimental.text_splitter import SemanticChunker
import math
import json
import pickle
import camelot.io as camelot
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import PyPDF2
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import tempfile
import pdfplumber
load_dotenv()

class VectorStore():
    def __init__(self, directory=None, **kwargs):
        self.config_file_path=os.path.join('pages','src','Database','config.json')
        with open(self.config_file_path, 'r') as file:
            config_data = json.load(file)
        self.config_data=config_data
        self.unstructured_directory_path=self.config_data['unstructured_data']
        self.vector_stores=None
        self.directory=directory
        self.data=None
        self.key=None
        self.chroma_db=None
        self.vector_stores_list=[]
        if self.directory is not None:
            self.unstructured_directory_path=self.directory


         
    def makevectorembeddings(self,key=None, **kwargs):
        if self.directory is None:
            if key is None:
                with st.spinner('searching for structure in the data'):
                    self._preprocess_data_in_directory()
        
        with st.spinner('loading data as text'):
            loader = DirectoryLoader(os.path.join(self.unstructured_directory_path), glob="*.txt", show_progress=True)
            if key is None:
                loader2  = DirectoryLoader(os.path.join(self.unstructured_directory_path), glob="**/[!.]*.csv", show_progress=True)
                merged_data_loader = MergedDataLoader([loader,loader2])
                self.data = merged_data_loader.load()
            else:
                self.data = loader.load()
        # embedding_list=[HuggingFaceHubEmbeddings(huggingfacehub_api_token=st.secrets['HUGGINGFACEHUB_API_TOKEN']['api_token']),
        #                 GooglePalmEmbeddings(google_api_key=st.secrets['GOOGLE_GEMINI_API']['api_key']),
        #                 GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
        #                 HuggingFaceInferenceAPIEmbeddings(api_key=st.secrets['HUGGINGFACEHUB_API_TOKEN']['api_token'],model_name="BAAI/bge-base-en-v1.5")]
        embedding_list=[HuggingFaceHubEmbeddings(huggingfacehub_api_token=st.secrets['HUGGINGFACEHUB_API_TOKEN']['api_token']),
                        GooglePalmEmbeddings(google_api_key=st.secrets['GOOGLE_GEMINI_API']['api_key']),
                        HuggingFaceInferenceAPIEmbeddings(api_key=st.secrets['HUGGINGFACEHUB_API_TOKEN']['api_token'],model_name="BAAI/bge-base-en-v1.5")]
        
        if kwargs['embedding_num'] is not None:
            embeddings = embedding_list[kwargs['embedding_num']]
            print("embs",embeddings)
        else:
            embeddings = embedding_list[1]
            print(embeddings)

        if self.directory is None:
            # text_splitter = SemanticChunker(
            #    embeddings , breakpoint_threshold_type="percentile"
            # )
            ##########################
            #CHROMA INITIALIZATION
            self.chroma_db=Chroma(embedding_function=embeddings, collection_name=f"{time.time()}")
            with st.spinner('creating chunks'):
                chunks = self.create_chunks_for_parallel_processing(self.data)
                list_of_docs=self.documents_from_chunks(text_chunks=chunks)

            with st.spinner('creating embeddings'):
            
                self._parallel_embeddings(db=self.chroma_db,list_of_docs=list_of_docs)
            with st.spinner('storing embeddings to Chroma'):
                bm25_retriever=BM25Retriever.from_documents(list_of_docs)
            self.vector_stores=self.chroma_db
            return (self.vector_stores,bm25_retriever)

        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=100,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents=self.data)
            print(embeddings)
            print('structured chroma')
            print(f"The vector store contains {self.vector_stores} documents.")
            self.vector_stores=Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name=f"{time.time()}")
            print(f"The vector store contains {self.vector_stores} documents.")

            return self.vector_stores


    def _create_documents_in_parallel(self,text_chunk, text_splitter):
        # print(text_chunk,text_splitter)
        return text_splitter.create_documents([text_chunk])
    def documents_from_chunks(self,text_chunks):
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=len(text_chunks[0]),
                chunk_overlap=20,
                length_function=len
            )
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(self._create_documents_in_parallel, text_chunk, text_splitter) for text_chunk in text_chunks]
                chunks = [chunk for future in futures for chunk in future.result()]
        return chunks
    
    def create_chunks_for_parallel_processing(self,data):
        text=''
        for doc in data:
            text+=doc.page_content+'\n'
        processed_chunks=[]
        print(len(text))
        if len(text)>=1000000:
            chunk_size=len(text)//1000
        elif len(text)>=100000000:
            chunk_size=len(text)//100000
        elif len(text)>=10000000:
            chunk_size=len(text)//10000
        elif len(text)<1000000 and len(text)>=10000:
            chunk_size=len(text)//100
        elif len(text)>10 and len(text)<10000:
            chunk_size=len(text)//10
        elif len(text)<10:
            chunk_size=len(text)//1
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=20,
                length_function=len
            )
        splitted_text_list=text_splitter.split_text(text)
        # len(splitted_texts)
        chunk_size = math.floor(len(splitted_text_list) / os.cpu_count())
        if chunk_size==0:
            remainder_text_list=0
            chunk_size=1
        else:
            remainder_text_list=len(splitted_text_list)%chunk_size
        text_chunks = [splitted_text_list[i:i + chunk_size] for i in range(0, len(splitted_text_list)-remainder_text_list, chunk_size)]
        text_chunks[-1].append(" ".join(splitted_text_list[len(splitted_text_list)-remainder_text_list:]))
        for chunks in text_chunks:
            processed_chunks.append(" ".join(chunks))


        return processed_chunks
    def _create_batches(self,list_of_docs):
        batch_size = math.floor(len(list_of_docs)) // os.cpu_count()
        if batch_size==0:
            remainder=0
            batch_size=1
        else:
            remainder = len(list_of_docs)%os.cpu_count()
        batches = [list_of_docs[i:i+batch_size] for i in range(0,len(list_of_docs)-remainder,batch_size)]
        batches[-1]+=list_of_docs[len(list_of_docs)-remainder:]
        return batches
    
    def _create_vstore_in_parallel(self,db : Chroma,chunks):
       return db.add_documents(chunks)
    




    def _parallel_embeddings(self,db,list_of_docs):
        batches=self._create_batches(list_of_docs)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._create_vstore_in_parallel,db,chunks) for chunks in batches]
            vstores = [future.result() for future in futures]




    def _savevectorstores(self):
        serialized_data = self.vector_stores.serialize_to_bytes()
        serialized_path=os.path.join(self.unstructured_directory_path,'serialized_index.pkl')
        with open(serialized_path, "wb") as file:
            pickle.dump(serialized_data, file)

        return type(serialized_data)


    def loadvectorstores(self):
        serialized_path=os.path.join(self.unstructured_directory_path,'serialized_index.pkl')
        with open(serialized_path, "rb") as file:
            serialized_faiss = pickle.load(file)
        embeddings = GooglePalmEmbeddings(google_api_key=st.secrets['GOOGLE_GEMINI_API']['api_key'])
        self.vector_stores = FAISS.deserialize_from_bytes(serialized_faiss , embeddings=embeddings)

        return self.vector_stores





    def _preprocess_text(self,text):
        text = text.lower()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]

        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    
    def split_pdf_in_memory(self, input_pdf_path):
        pdf_reader = PyPDF2.PdfReader(input_pdf_path)
        num_pages = len(pdf_reader.pages)
        num_cores = os.cpu_count()

        # Calculate the number of pages per part
        pages_per_part = num_pages // num_cores
        remainder_pages = num_pages % num_cores

        pdf_parts = []
        pdf_writer = None
        current_part_pages = 0

        for i in range(num_pages):
            if len(pdf_parts) < num_cores:
                if current_part_pages == 0:
                    pdf_writer = PyPDF2.PdfWriter()

                pdf_writer.add_page(pdf_reader.pages[i])
                current_part_pages += 1

                # Check if current part is filled or it's the last page
                if current_part_pages == pages_per_part or i == num_pages - 1:
                    # Create a temporary file for each part
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        pdf_writer.write(temp_pdf)
                        pdf_parts.append(temp_pdf.name)
                    current_part_pages = 0  # Reset current part page count

        return pdf_parts

    def process_pdf_part(self, pdf_path):
        # try:
        #     tables = camelot.read_pdf(pdf_path, pages='all')
        #     data_frames = [table.df for table in tables]

        # except Exception:
        data_frames = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                            data_frames.append(pd.DataFrame(table))
        finally:
            os.remove(pdf_path)

        return data_frames

    def extract_tables_from_pdf_parallel_processing(self, pdf_path):
        pdf_parts = self.split_pdf_in_memory(pdf_path)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(self.process_pdf_part, pdf_parts))

        combined_results = []
        for result in results:
            combined_results.extend(result)

        return combined_results

    def _preprocess_data_in_directory(self):
        poppler_bin_path = r'\poppler-24.02.0\Library\bin'
        os.environ['PATH'] += os.pathsep + poppler_bin_path

        print(os.environ['PATH'])

        
        directory_path=self.unstructured_directory_path
        print('inside preprocess')


        preprocessed_texts = []
        audio=False
        elements=[]
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Read text from PDF files
            if filename.endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    element=partition_pdf(file_path)
                    elements.append(ele.text for ele in element)
                
                data_frames=self.extract_tables_from_pdf_parallel_processing(pdf_path=file_path)
                current_columns=[]
                i=0
                for table in data_frames:
                        df=table
                        # df.columns=df.iloc[0]
                        if not list(df.iloc[0])==list(current_columns):
                            current_columns=df.iloc[0]
                            i+=1
                            try:
                                with open(os.path.join(directory_path, f'table_{i}.csv'), 'w', encoding='utf-8') as file:
                                    file.write(df.to_csv(header=False, index=False))
                            except Exception as e:
                                file.write('')
                        elif list(current_columns)==list(df.iloc[0]):
                            # df=df.iloc[1:]
                            df=df.iloc[1:]
                            try:
                                with open(os.path.join(directory_path, f'table_{i}.csv'), 'a', encoding='utf-8') as file:
                                    file.write(df.to_csv(header=False, index=False))
                            except Exception as e:
                                file.write('')

            elif filename.endswith('.txt'):
                audio=True

        if not audio:
            os.makedirs(self.unstructured_directory_path, exist_ok=True)
            output_file_path=os.path.join(self.unstructured_directory_path,"output.txt")
            print('This is pdf file processor')

            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                # for text in preprocessed_texts:
                #     outfile.write(text + '\n')
                for element in elements:
                    print(element)
                    for ele in element:

                        outfile.write(json.dumps(self._preprocess_text(ele))+'\n')



