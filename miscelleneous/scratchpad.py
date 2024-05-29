

from langchain.agents import Agent, AgentExecutor, create_self_ask_with_search_agent, create_react_agent


# from .pages.src.Tools.scraper import calheadlesschromiumforsearchagent



from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain.chains.llm import LLMChain
def promptformatter():
        input_variables = ['input','tools','tool_names'] 
        template ="""As an advanced AI you search internet for answers. You have been given question from user, use tools to answer questions. If you know the answer, do not use tools.\n
        Try to be friendly and helpful. Always follow the Action/Thought/Observation format. All the tools given to you are valid.
        \n\n
        \nQuestion: {input},\n\n

        Here are the tools you may need to use:\n\n
        tools: {tools}
        tools_names: {tool_names}
        """

        # Create a new HumanMessagePromptTemplate with the modified prompt
        human_message_template = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=input_variables, template=template))
        system_message_template = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['agent_scratchpad','tool_names'],template="""
        
        Follow this format:
        Thought:{agent_scratchpad}
        Action:\n
        Action Input: \n
        Observation: \n
        """))

        # Create a new ChatPromptTemplate with the modified HumanMessagePromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages([(human_message_template),(system_message_template)])
        # logging.info(chat_prompt_template)
        return chat_prompt_template

import json
# config_file = os.path.join('pages', 'src', 'Database', 'Prompts','prompts.json')
# with open(config_file, 'r') as file:
#     config_data = json.load(file)

# print(config_data['Unstructured Prompts']['crewaiagent'])

# llm=GoogleGenerativeAI(google_api_key=os.environ.get('GOOGLE_GEMINI_API'), model="gemini-pro")
# tools=[Tavilysearchapi,duckduckgo,Scraper]
# tool_names=['Tavilysearchapi','duckduckgo','Scraper']
# agent=create_react_agent(llm=llm,prompt=promptformatter(),tools=tools)
# from langchain.agents import AgentExecutor

# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# print(agent_executor.invoke({"input": "What year is it?", 'tools':tools, 'tool_names':tool_names}))

# import os
# from joblib import load
# config_file = os.path.join('pages', 'src', 'Database', 'config.json')
# with open(config_file, 'r') as file:
#     config_data = json.load(file)

# classification_model_path=config_data['Classification_models']
# def decision(sentence):
#         sentence_vectorized=load(os.path.join(classification_model_path,'tfidf_pretrained.joblib')).transform([sentence])
#         prediction=load(os.path.join(classification_model_path,'randomtree_decision_pretrained.joblib')).predict(sentence_vectorized)
#         return prediction[0]

# print(decision('hello how are you'))