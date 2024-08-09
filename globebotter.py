
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.memory import ConversationBufferMemory 
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


st.set_page_config(page_title="GlobeBotter", page_icon="🌐")
st.header('🌐 안녕하세요. 저는 글로브보터입니다. 인터넷에 접속할 수 있는 여행 도우미입니다. 다음 여행은 무엇을 계획 중이신가요?')



load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']
serpapi_api_key = os.environ['SERPAPI_API_KEY']

search = SerpAPIWrapper()
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

raw_documents = PyPDFLoader('italy_travel.pdf').load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

memory = ConversationBufferMemory(
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output"
)

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="현재 일어나고 있는 일에 관한 질문에 답할 때 유용합니다."
    ),
    create_retriever_tool(
        db.as_retriever(), 
        "italy_travel",
        "이탈리아에 관한 문서를 검색하고 반환합니다."
    )
    ]

agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

user_query = st.text_input(
    "**휴가 때 어디에 갈 계획이신가요?**",
    placeholder="무엇이든 물어보세요!"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]
if "memory" not in st.session_state:
    st.session_state['memory'] = memory


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


def display_msg(msg, author):

    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

if user_query:
    display_msg(user_query, 'user')
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

if st.sidebar.button("채팅 기록 초기화"):
    st.session_state.messages = []
