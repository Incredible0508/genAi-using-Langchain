from langchain_groq import ChatGroq
import streamlit as st
#streamlit is a framework for building web apps in Python. It allows you to create interactive user interfaces for your applications.

from dotenv import load_dotenv

load_dotenv()

st.header("Research Tool")


user_input = st.text_input("Enter your prompt here:")
#st.text_input is a method provided by Streamlit to create a text input field in the web app.     
# It takes a string as input and returns the user's input as a string.


if st.button("Summarize"):
    result = ChatGroq(model="openai/gpt-oss-20b", temperature=0).invoke(user_input)
    st.write(result.content)

#st.write is a method provided by Streamlit to display text or other elements in the web app. In this case, it is used 
# to display the output of the model after invoking it with the user's input.

#Static prompts are pre-defined prompts that are used to guide the model's behavior. 
# They can be used to provide context or instructions to the model, helping it generate more relevant and accurate responses. 
# In this case, the static prompt is "Summarize", which instructs the model to summarize the user's input.

