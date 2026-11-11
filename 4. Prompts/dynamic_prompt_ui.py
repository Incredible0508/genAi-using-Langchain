from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
#langchain_core.prompts is a module that provides classes and functions for working with prompts in LangChain.
#load_prompt is a function that loads a prompt from a file or a string. 
# It takes a string as input and returns a PromptTemplate object, which can be used to generate prompts for the model.

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-20b", temperature=1.5)

st.header("Research Tool")


paper_input = st.selectbox("Select Research Paper", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )
style_input = st.selectbox("Select Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox("Select Length", ["Short", "Medium", "Long"])

    
#template
template = load_prompt("template.json")


#prompt = template.invoke({
#   'paper_input': paper_input,
#  'style_input': style_input,
# 'length_input': length_input
#})



if st.button("Summarize"):
    chain = template | model
    #this line creates a chain of the prompt template and the model, allowing for dynamic generation of prompts based on user input.
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
