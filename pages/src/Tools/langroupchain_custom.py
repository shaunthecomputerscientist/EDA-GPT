from langchain.agents import create_react_agent,AgentExecutor
from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from datetime import datetime
import time
import json
from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
import re
from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
import time
from crewai import Task,Agent,Crew
import json
from typing import Dict, List, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field
from textwrap import dedent
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

class Node(BaseModel):
    question: str

class Edge(BaseModel):
    source: int
    target: int

class Graph(BaseModel):
    """Structure of graph where questions are nodes and edges are relationships."""
    nodes: List[Node] = Field(description='List of questions, index of question represent node number')
    edges: List[Edge] = Field(description="List of edges where each edge is a list of two node number")

class QAChain(BaseModel):
   response: Tuple[str] = Field(description="tuple of answers, each answer correspond to given question.")
class Queue:
    def __init__(self):
        self.queue=[]
    def addq(self,v):
        self.queue.append(v)
    def delq(self):
        v=None
        if not self.isempty():
            v=self.queue[0]
            self.queue=self.queue[1:]
        return v

    def isempty(self):
        return self.queue==[]

    def __str__(self):
        return str(self.queue)

class LangNode:
    def __init__(self, question: str, serial_number: int,group_id: int = -1):
        self.question = question
        self.serial_number = serial_number
        self.answer = None
        self.context = None
        self.group_id = group_id
        self.next_node = None

    def add_answer(self,answer):
      self.answer=answer

    def add_context(self,context):
      self.context=context

class LangModelGroup:
    def __init__(self, group : List[LangNode], group_id: int):
        self.group = group
        self.group_id = group_id
        self.next_group = None
        self.cumulative_group_context=None
        self.group_context=None
    def extract_group_context(self, group : List[LangNode]):
      context={}
      for nodes in group:
        context[nodes.question]=[nodes.answer if nodes.answer is not None else ''][0]

      return context
    def add_context(self,context):
       self.group_context=context

class TupleOutputParser:
    def parse(self, text: str):
        try:
            parsed_data = eval(text)
            if isinstance(parsed_data, tuple):
                return parsed_data
            else:
                raise ValueError("Parsed data is not a tuple")
        except Exception as e:
            raise ValueError(f"Failed to parse output as tuple: {e}")

class LLMGraph:
  def __init__(self, edges : List[List[int]], nodes : Dict[int,Node]):
    self.edges=[[int(x) for x in edge] for edge in edges]
    self.nodes={int(key) : value for key , value in nodes.items()}
    self.graph={}
    self.groups=None
  def _construct_alist(self):
    for i in range(len(self.nodes)):
      self.graph[i]=[]
    for edge in self.edges:
        source, dest = edge
        # if source not in graph:
        #     graph[source] = []
        if dest not in self.graph[source] and source not in self.graph[dest]:
            self.graph[int(source)].append(int(dest))
    for key in self.graph.keys():
      if self.graph[key]==[key]:
        self.graph[key]=[]
    return self.graph



  def _toposortlist(self,Alist):
      (indegree,toposortlist, group)=({},[],[])
      for u in Alist.keys():
          indegree[u]=0

      for u in Alist.keys():#O(n)
          for v in Alist[u]:
              indegree[v]=indegree[v]+1


      zerodegreeq = Queue()
      subgroup=[]
      for u in Alist.keys():

          if indegree[u]==0:#O(n)
              zerodegreeq.addq(u)
              subgroup.append(u)

      group.append(subgroup)





      while not zerodegreeq.isempty():
          j=zerodegreeq.delq()
          toposortlist.append(j)
          indegree[j]=indegree[j]-1
          subgroup=[]
          for k in Alist[j]:#O(n)
              indegree[k]=indegree[k]-1

              if indegree[k]==0:
                  subgroup.append(k)
                  zerodegreeq.addq(k)
          if subgroup!=[]:
              group.append(subgroup)

      return (toposortlist,group, indegree)

  def _ordertopologically(self):
      alist=self._construct_alist()
      return self._toposortlist(alist)

  def construct_group_structure(self):
        _, groups, _ = self._ordertopologically()
        group_objects = []
        print(groups)

        for group_id, group in enumerate(groups):
            # print('current group',group_id)
            nodes_in_group = [LangNode(question=self.nodes[node_id],serial_number=node_id,group_id=group_id) for node_id in group]#list[langnode]
            # print(f"node in group with group id {group_id}",nodes_in_group)
            # for node in nodes_in_group:
            #     node.group_id = group_id
            group_objects.append(LangModelGroup(nodes_in_group, group_id))

        for i in range(len(group_objects) - 1):
            group_objects[i].next_group = group_objects[i + 1].group_id
        self.groups={group.group_id : group for group in group_objects}

        return self.groups

class LangGroupChain:
  def __init__(self,llm,temperature=0.5, tools=[]):
    self.llm=llm
    self.temperature=temperature
    self.llmgraph=None
    self.langmodelgroups=None
    self.tools=tools

    
  def format_prompt(self,data:str):
     return self.prompt+data
     
  def generate_group(self,query:str):
    try:
      nodes={}
      edges=[]
      max_attempts=3
      attempts=0
      while nodes=={}:
        prompt = ChatPromptTemplate.from_template("""
        User question: what subquestions do I need to solve this question -> {query}
        TASK: Given the User question, decompose it into a series of subquestions that lead to the desired output.Every Node corresponds to a subquestion. Every Edge relates two nodes in a directed graph.
        This should form a directed graph such that if a question has dependency, that dependency will be solved before indicating the dependency has a directed edge to the other node.
        Ensure that:
        1. Every node is populated with a meaningful subquestion.
        2. Edges clearly define the topological relationship between these nodes, meaning edges are directed from node which needs to be performed before to one after.
        3. Always ask right set of questions that give maximum information.
        4. If original question is self-sufficient, return that question as node.
        5. questions should be such that it does not demand user input.
        Output Format:
        nodes is a List (e.g., {{"question 1","question 2", ...}})
        where question 1 is node 0, question 2 is node 1, so on...
        edges is a List[List[nodes]] (e.g., [[0, 1], [1, 2], ...])
        Do not leave nodes or edges empty. Define atmost 10 nodes, at min 1 node.
        nodes: {{nodes}}
        edges: {{edges}})""")

        chain =  prompt | self.llm.bind(functions=[convert_to_openai_function(Graph)]) | JsonOutputFunctionsParser()
        result = chain.invoke({"query":query})
        print(result['nodes'], result['edges'])
        print(type(result))
        nodes={index:value for index,value in enumerate(result['nodes'])}
        print(nodes)
        edges=result['edges']

      
      if nodes=={}:
         nodes={0:query}
         edges=[[0,0]]

      self.llmgraph=LLMGraph(edges=edges,nodes=nodes)
      self.langmodelgroups=self.llmgraph.construct_group_structure()
      return self.langmodelgroups
    except Exception as e:
      raise e

  def _promptformatter(self):
    hwchase17="""You are given questions in format -> List[question 1, question 2, ... , question N]. Here are current questions :{input}
    INSTRUCTION: {instruction}
    Answer every question given in the list serially.
    You have access to the following tools:\n{tools}\n
    ERROR OUTPUT: If you do not know the answer, return "TERMINATE" as string.
    Context : May contain useful data else empty ---> {prev_context} \n\n
    Strictly use the following format:\n
    Question: serially choose next question from list\n
    Thought: Always think about next Action to use\n
    Action: the action to take, should be one of [{tool_names}]. Do not use same tool more than 2 times.\n
    Action Input: input to the Action\n
    Observation: result of Action\n... (this Thought/Action/Action Input/Observation can repeat N times)\n
    Thought: I now know the final answer\n
    Final Answer: final answer to all questions from list
    Begin!\n
    Thought:{agent_scratchpad}"""
    template=hwchase17
    input_variables=['tools','tool_names','agent_scratchpad','input', 'prev_context']
    human_message_template = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=input_variables, template=template))
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_template])
    return chat_prompt_template

  def generate_response2(self, query):
    self.generate_group(query)
    agent = create_react_agent(
    tools=self.tools,
    llm=self.llm,
    prompt=self._promptformatter(),
    )
    agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True, max_iterations=8)

    cumulative_data=[]
    for key, value in self.langmodelgroups.items():
      for node in value.group:
          answer=agent_executor.invoke({"input": f"{node.question}", "prev_context":f"{cumulative_data}"})
          node.add_answer(answer['output'])
          time.sleep(5)

      cumulative_data.append(value.extract_group_context(value.group))

    return cumulative_data
  def escape_special_characters(self, text: str) -> str:
        special_characters_pattern = r'[\x00-\x1f\\"]'
        escaped_text = re.sub(special_characters_pattern, lambda x: '\\' + x.group(0), text)

        return escaped_text

  def clean_and_escape_json(self, data: str) -> str:
        cleaned_data = data.strip("```").strip()
        cleaned_data = self.escape_special_characters(cleaned_data)
        cleaned_data = re.sub(r'([{,])(\s*)(\w+)(\s*):', r'\1"\3":', cleaned_data)
        cleaned_data=cleaned_data.replace('\\','').replace("'",'').strip("json").strip()
        return cleaned_data

  def convert_to_valid_json(self,data: str):
        try:
            cleaned_data = self.clean_and_escape_json(data)
            print(cleaned_data)
            return json.loads(cleaned_data.strip("'").replace('\n', ''))
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            raise e
  def generate_response(self, query):
    self.generate_group(query)
    agent = create_react_agent(
    tools=self.tools,
    llm=self.llm,
    prompt=self._promptformatter(),
    )
    agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True, max_iterations=10, early_stopping_method='generate')
    cumulative_data=[]
    for key, value in self.langmodelgroups.items():
      questions=[node.question+f" This is question_{index} " for index,node in enumerate(value.group)]
      instructions = f"You have a list of {len(questions)} questions. You need to return {len(questions)} answers where each answer correspond to each question."
      answer=agent_executor.invoke({"input": f"{questions}", "prev_context":f"{cumulative_data}", "instruction":instructions})
      print('--------------------------------------------------------------------------------')
      print(answer['output'], type(answer['output']))
      result=answer['output']
      if "terminate" in result.lower() or "am sorry" in result.lower():
         return "Try again"
      
      else:
        formatted_context=f"Here is list of previous question : {[node.question for index,node in enumerate(value.group)]}\n Here are answers found : {result}\n"
        cumulative_data.append(formatted_context)
        value.add_context(formatted_context)
        print(value.group_context)
         
    return cumulative_data
  

  def generate_response3(self,query):
      self.generate_group(query)
      max_iter=5
      Searcher = Agent(
      role=dedent(f"You are advanced ai named Logician with capabilities like logic, understanding,reasoning. Here is current date-time : {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"),
      backstory=dedent("You have ability to reason and think critically. Through logic you deduce answers. Tools are only given for finding data but you need to reason yourself."),
      verbose=True,
      max_iter=max_iter,
      llm = self.llm,
      goal=dedent(f"Given List of user query, answer the questions serially. You have {max_iter} tries to answer all questions."),
      tools=self.tools
      )

      cumulative_data=[]
      for key, value in self.langmodelgroups.items():
        questions=[node.question+f" This is question no {index} " for index,node in enumerate(value.group)]
        task = Task(
        description=dedent(f"Original question:{query} ---> Generated User subquestion List : {questions}\n. GOAL : Answer These Sub Questions in Sequence.\n Useful Data : {cumulative_data}\n.Tools given give you access to the internet. If you can reason the answer then do not use tools."),
        agent=Searcher,
        async_execution=False,
        expected_output=dedent("Answer all questions in sequence"),
        )
        crew = Crew(
        agents=[Searcher],
        tasks=[task],
        verbose=0,
        )
        result=crew.kickoff()
        formatted_context=f"List of previous question : {[node.question for index,node in enumerate(value.group)]}\n Answers found for these questions : {result}\n"
        cumulative_data.append(formatted_context)
        value.add_context(formatted_context)

        print(value.group_context)
         
      return cumulative_data