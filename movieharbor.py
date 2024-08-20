import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import LanceDB
import lancedb
import pandas as pd
from langchain.chains import RetrievalQA

st.set_page_config(page_title="MovieHarbor", page_icon="🎬")
st.header('🎬 MovieHarbor에 오신 것을 환영합니다. 최고의 영화 추천 시스템입니다!')

load_dotenv()

#os.environ["HUGGINGFACEHUB_API_TOKEN"]
openai_api_key = os.environ['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings()
uri = "data/sample-lancedb"
db = lancedb.connect(uri)

table = db.open_table('movies')
docsearch = LanceDB(connection=db, embedding=embeddings, table_name="movies")

# 영화 데이터셋 불러오기
md = pd.read_pickle('movies.pkl')

# 사용자 입력을 위한 사이드바 생성
st.sidebar.title("영화 추천 시스템")
st.sidebar.markdown("아래에 정보를 입력해주세요:")

# 사용자에게 나이, 성별 및 선호 영화 장르를 묻기
age = st.sidebar.slider("나이를 선택하세요", 1, 100, 25)
gender = st.sidebar.radio("성별을 선택하세요", ("Male", "Female", "Other"))
genre = st.sidebar.selectbox("선호하는 영화 장르를 선택하세요", md.explode('genres')["genres"].unique())


# 사용자 입력을 기반으로 영화를 필터링
df_filtered = md[md['genres'].apply(lambda x: genre in x)]


template_prefix = """You are a movie recommender system that help users to find movies that match their preferences. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}"""

user_info = """This is what we know about the user, and you can use this information to better tune your research:
Age: {age}
Gender: {gender}"""

template_suffix= """Question: {question}
Your response:"""

user_info = user_info.format(age = age, gender = gender)

COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ template_suffix
print(COMBINED_PROMPT)

# 체인 설정
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'data': df_filtered}), return_source_documents=True)


query = st.text_input('질문을 입력하세요:', placeholder = '어떤 액션 영화를 추천해 주시겠어요?')
if query:
    result = qa.invoke({"query": query})
    st.write(result['result'])
