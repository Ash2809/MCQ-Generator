import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import pandas as pd
import os
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(temperature = 0.7,model = "gemini-pro")

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

template = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
{response_json}

"""

prompt = PromptTemplate(template = template,input_variables = ["text","number","subject","tone","response_json"])
generate_chain = LLMChain(prompt = prompt,llm= llm,output_key = "quiz")


template2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

prompt_evaluation = PromptTemplate(input_variables = ["subject","quiz"],template = template2)
review_chain = LLMChain(prompt = prompt_evaluation,llm = llm,output_key = "review")


main_chain = SequentialChain(chains = [generate_chain,review_chain],
                             input_variables = ["text","number","subject","tone","response_json"],
                             output_variables = ["quiz","review"])


path = r"C:\Projects\MCQ-Generator\data.txt"
with open(path,"r") as file:
    text = file.read()
print(text[:200])
json.dumps(RESPONSE_JSON)

st.set_page_config(page_title="MCQ GENERATION",
                   page_icon=":book:",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.header("GENERATE MCQs :book:")
number = st.text_input("Enter Number of MCQ questions")
tone = st.radio("Tone:", ("Professional","Easy","Medium","Beginner"))

submit = st.button("GENERATE")
if submit:
    st.balloons()
    input_prompt = prompt.format(text=text,number=number,subject = "politics",tone=tone,response_json=RESPONSE_JSON)
    response = llm.invoke(input_prompt)
    st.write(response.content)