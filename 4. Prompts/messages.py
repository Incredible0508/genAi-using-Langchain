from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

#SystemMessage is used to set the context of the conversation, HumanMessage is used to represent the user's input, and AIMessage is used to 
# represent the model's response.
#AIMessage is used to represent the model's response and HumanMessage is used to represent the user's input. 

from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following English text to French: 'Hello, how are you?'")
]
result = model.invoke(messages)
messages.append(result)
print("Bot: " + result.content)