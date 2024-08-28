import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities.dalle_image_generator import DallEAPIWrapper




st.set_page_config(page_title="StoryScribe", page_icon="📙")
st.header('📙 SNS 포스팅 생성기')

load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']

# 사용자 입력을 위한 사이드바 생성
st.sidebar.title("SNS 포스팅 생성기")
st.sidebar.markdown("아래에 세부 정보와 선호 사항을 입력하세요:")

llm = OpenAI()

# 사용자에게 주제, 장르, 대상 연령을 묻는다
topic = st.sidebar.text_input("주제가 무엇인가요?", '해변에서 달리는 개')
genre = st.sidebar.text_input("장르는 무엇인가요?", '드라마')
audience = st.sidebar.text_input("시청자는 누구인가요?", '청소년')
social = st.sidebar.text_input("어떤 소셜 미디어에 게시할까요?", '인스타그램')

# 이야기 생성기
story_template = """You are a storyteller. Given a topic, a genre and a target audience, you generate a story.

Topic: {topic}
Genre: {genre}
Audience: {audience}
Story: This is a story about the above topic, with the above genre and for the above audience:"""
story_prompt_template = PromptTemplate(input_variables=["topic", "genre", "audience"], template=story_template)
story_chain = LLMChain(llm=llm, prompt=story_prompt_template, output_key="story")

# 소셜 미디어 게시물 생성기
social_template = """You are an influencer that, given a story, generate a social media post to promote the story.
The style should reflect the type of social media used.

Story: 
{story}
Social media: {social}
Review from a New York Times play critic of the above play:"""
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
