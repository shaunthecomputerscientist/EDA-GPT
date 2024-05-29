from langchain.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
# from langchain_community.utilities.brave_search import BraveSearchWrapper
import os, re
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import json
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.pydantic_v1 import BaseModel, Field
import streamlit as st
import math



@tool('Wikipedia', return_direct=False)
def wikipedia(query: str)->str:
    "A wrapper around Wikipedia. Useful for when you need to answer people, places, companies, facts" "Input should be a search query. Not good for latest data"
    with st.spinner('searching wikipedia'):
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
        wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
        return wiki.run(query)

@tool("DateTime")
def datetimee(**kwargs):
    """Provides current user date time"""
    return datetime.today().strftime('%Y-%m-%d %H:%M:%S')


@tool('Arxiv', return_direct=False)
def arxiv(query: str)->str:
    "Only provides research papers based on query"
    with st.spinner('searching arxiv'):
        arxiv_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=20000)
        arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
        return arxiv.run(query)


@tool('Calculator',return_direct=False)
def Calculator(expression : str):
    """Calculator takes variable `expression` as mathematical string and evaluates it for answer"""
    try:
        return eval(expression)
    except Exception as e:
        return e

@tool('YoutubeVideoTranscript', return_direct=False)
def YoutubeVideoTranscript(url : str):
    """Provides Trascript from video url for youtube"""
    si_regex = r'/([a-zA-Z0-9_-]{11})\?si='
    def is_english(text):
        return all(ord(char) < 128 for char in text)
    
    # Check if the URL contains 'si='
    if 'si=' in url:
        match = re.search(si_regex, url)
        # If a match is found, return the video ID
        if match:
            video_id=match.group(1)
        else:
            video_id=None
    elif 'v=' in url:
        video_id=url.split('v=')[1]
    else:
        video_id=url.split('/')[-1]

    try:
        with st.spinner('searching youtube transcript'):
            transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id,languages=['en','hi'])
        text = ''
        for line in transcript:
            text += line['text'] + ' '
        return text
    except Exception as e:
        return f"Transcript extraction failed: {str(e)}"

        
    
class BingSchema(BaseModel):
    """schema For Bing Search"""
    snippet : str = Field(description='snippet of result')
    title : str= Field(description='title of result')
    link : str = Field(description='url of article')



@tool("SEARCH_API",return_direct=False)
def SEARCH_API(query: str, num_results=3)->str:
    """searches for latest news, technology, events, politics, etc.Takes in query and num_reults. Returns json of results with title, sources and main_content as fields."""
    max_search_results=math.ceil(num_results/3)
    fields=['\ntitle','\nsources','\nmain_content']
    results={f'RESULT {i+1}':"" for i in range(max_search_results*3)}
    for key in results:
        results[key]={field:"" for field in fields}
    count=0
    tavily=TavilySearchAPIWrapper(tavily_api_key=st.secrets['Tavily_api_key']['api_key'])
    bing=BingSearchAPIWrapper(bing_subscription_key=st.secrets['BING_API_KEY']['api_key'],bing_search_url="https://api.bing.microsoft.com/v7.0/search")
    duckduckgonews =DuckDuckGoSearchAPIWrapper()
    with st.spinner('searching duckduckgo'):
        result_ddgs_news=duckduckgonews._ddgs_news(query=query, max_results=max_search_results)
    with st.spinner('searching bing'):
        result_bing=bing.results(query=query, num_results=max_search_results)
    with st.spinner('searching tavily'):
        result_tavily=tavily.results(query=query, max_results=max_search_results)

    for i in range(len(result_bing)):
        current_ele_bing=BingSchema.parse_raw(str(result_bing[i]).replace("'",'"')).dict()
        count=count+1
        results[f'RESULT {count}']['\ntitle']=current_ele_bing['title']
        results[f'RESULT {count}']['\nsources']=current_ele_bing['link']
        results[f'RESULT {count}']['main_content']=current_ele_bing['snippet']
    
    for ele in result_tavily:
        count=count+1
        results[f'RESULT {count}']['\nsources']=ele['url']
        results[f'RESULT {count}']['\nmain_content']=ele['content']

    for ele in result_ddgs_news:
        count=count+1
        results[f'RESULT {count}']['\nsources']=ele['url']
        results[f'RESULT {count}']['\nmain_content']=ele['body']
        results[f'RESULT {count}']['\ntitle']=ele['title']


    return results


@tool("Scraper")
def Scraper(url: str)->str:
    """Takes valid url as input and scrapes content from the url. Do not provide invalid url, it will throw error. Only use if you know the url."""
    pattern=r'https://\S+'
    matches=re.findall(pattern,url)
    try:
        with st.spinner(f'scraping the url {url}'):
            websiteob=WebBaseLoader(web_paths=matches)
            result=websiteob.load()
    except Exception as e:
        return f"Error Occured while scraping the URL {e}, Invalid url, please use a proper real url on next iteration"
    return result





class Vision:
    def __init__(self, model):
        config_file = os.path.join('pages', 'src', 'Database', 'config.json')
        with open(config_file, 'r') as file:
            self.config_data = json.load(file)
        self.model=model



    def _input_image_setup(self,file_loc):
        if not (img := Path(file_loc)).exists():
            raise FileNotFoundError(f"Could not find image: {img}")
        image_parts = [
            {
                "mime_type": "image/png",
                "data": Path(file_loc).read_bytes()
                }
            ]
        return image_parts
    def _answer_image(self,context_or_sentence, image_content):
        input_prompt = """You are an image analyzer who can answer questions about the image given. You have general understanding about the world and are self aware which must reflect in your answer."""
        model = genai.GenerativeModel(model_name=self.model)
        prompt_parts = [input_prompt,image_content[0],context_or_sentence]
        response = model.generate_content(prompt_parts)
        # response=model.generate_content([context_or_sentence,image_content],)

        # print(response)
        response.resolve()
        return response.text
    def vision(self,query : str )->str:

        for img in os.listdir(self.config_data['QnA_img']):

            file_path=os.path.join(self.config_data['QnA_img'],img)
        image_content=self._input_image_setup(file_path)
        result = self._answer_image(query,image_content=image_content)
        os.remove(file_path)

        return result