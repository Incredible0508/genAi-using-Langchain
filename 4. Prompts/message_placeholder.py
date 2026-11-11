from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=10)
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

chat_history = []
#load chat history from a file or database
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'What is my name?'})

result = model.invoke(prompt)
print(result.content)