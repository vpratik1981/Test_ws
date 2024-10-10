import os
print(os.environ['OPENAI_API_KEY'])

# os.environ[]




# from openai import OpenAI
# import httpx
# http_client = httpx.Client(verify="./.venv/lib/python3.12/site-packages/certifi/cacert.pem")

# client = OpenAI(
    
#     http_client = http_client,
    
# ) 

# print(os.environ['GROQ_API_KEY'])

# from openai import OpenAI
# client = OpenAI()

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Write a haiku about recursion in programming."
#         }
#     ]
# )

# print(completion.choices[0].message)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain





llm = ChatOpenAI(api_key='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')

# llm.invoke("how can langsmith help with testing?")

prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm



output_parser = StrOutputParser()

chain=prompt|llm|output_parser

# print(chain.invoke({"input": "how can langsmith help with testing?"}))

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()

embeddings = OpenAIEmbeddings()

text_spitter = RecursiveCharacterTextSplitter()
documents = text_spitter.split_documents(docs)

vector = FAISS.from_documents(documents,embeddings)

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm,prompt)

# document_chain.invoke({
#     "input": "how can langsmith help with testing?",
#     "context": [Document(page_content="langsmith can let you visualize test results")]
# })

retriever = vector.as_retriever()

# retrieval_chain = create_retrieval_chain(retriever,document_chain)

# response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print(response["answer"])

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='chat_history'),
    ('user','{input}'),
    ('user','Given the above conversation, generate a search query to look up to get information relevant to the conversation')
])

retrieval_chain= create_history_aware_retriever(llm,retriever,prompt)


from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content='Can LangSmith help test my LLM applications?'),AIMessage(content='Yes!')]

# response =retrieval_chain.invoke({
#     "chat_history": chat_history,
#     "input":"Tell me how"
# })

# print(response)
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",

)

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()

tools = [retriever_tool,search]

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

prompt = hub.pull("hwchase17/openai-functions-agent")

llm= ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
agent = create_openai_functions_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

print(agent_executor.invoke({"input": "how can langsmith help with testing?"}))








# print(st)




