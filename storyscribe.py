import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper




st.set_page_config(page_title="StoryScribe", page_icon="📙")
st.header('📙 SNS 포스팅 생성기')

load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']

# 사용자 입력을 위한 사이드바 생성
st.sidebar.title("SNS 포스팅 생성기")
st.sidebar.markdown("아래에 세부 정보와 선호 사항을 입력하세요:")

llm = ChatOpenAI(model="gpt-4o-mini")

# 사용자에게 주제, 장르, 대상 연령을 묻는다
topic = st.sidebar.text_input("주제가 무엇인가요?", '해변에서 달리는 개')
genre = st.sidebar.text_input("장르는 무엇인가요?", '드라마')
audience = st.sidebar.text_input("시청자는 누구인가요?", '청소년')
social = st.sidebar.text_input("어떤 소셜 미디어에 게시할까요?", '인스타그램')

# 이야기 생성기
story_template = """당신은 이야기꾼입니다. 주제, 장르, 대상 독자가 주어지면 이야기를 생성하세요.

주제: {topic}
장르: {genre}
대상 독자: {audience}
이야기: 위 주제와 장르, 대상 독자에 맞는 이야기를 생성하세요:"""
story_prompt_template = PromptTemplate(input_variables=["topic", "genre", "audience"], template=story_template)
story_chain = LLMChain(llm=llm, prompt=story_prompt_template, output_key="story")

# 소셜 미디어 게시물 생성기
social_template = """당신은 인플루언서입니다. 주어진 이야기를 바탕으로, 해당 이야기를 홍보할 소셜 미디어 게시물을 생성하세요.
스타일은 사용된 소셜 미디어 유형을 반영해야 합니다.

이야기: 
{story}
소셜 미디어: {social}
위 작품에 대한 뉴욕 타임즈 연극 비평가의 리뷰:"""
social_prompt_template = PromptTemplate(input_variables=["story", "social"], template=social_template)
social_chain = LLMChain(llm=llm, prompt=social_prompt_template, output_key='post') 

# 이미지 생성기

image_template = """Generate a detailed prompt to generate an image based on the following social media post:

Social media post:
{post}

The style of the image should be oil-painted.

"""

prompt = PromptTemplate(
    input_variables=["post"],
    template=image_template,
)
image_chain = LLMChain(llm=llm, prompt=prompt, output_key='image')

# 전체 체인

overall_chain = SequentialChain(input_variables = ['topic', 'genre', 'audience', 'social'], 
                chains=[story_chain, social_chain, image_chain],
                output_variables = ['story','post', 'image'], verbose=True)


if st.button('게시물 생성하기!'):
    result = overall_chain({'topic': topic,'genre':genre, 'audience': audience, 'social': social}, return_only_outputs=True)
    image_url = DallEAPIWrapper().run(result['image'][:1000])
    st.subheader('이야기')
    st.write(result['story'])
    st.subheader('소셜 미디어 게시물')
    st.write(result['post'])
    st.image(image_url)
