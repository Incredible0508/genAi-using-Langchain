from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful assistant that translates {language1} to {language2}."),
    ('human', "Translate the following {language1} text to {language2}: '{text}'")
])
prompt = chat_template.invoke({'language1': 'English', 'language2': 'French', 'text': 'India is a really colourful country with a rich history and culture.'})
print(prompt)