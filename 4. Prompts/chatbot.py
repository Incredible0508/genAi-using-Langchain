from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")   

chatHistory = [
    SystemMessage(content="You are a helpful assistant"),
]

while True:
    userinput = input("You: ")
    chatHistory.append(HumanMessage(content=userinput))
    if userinput == "exit":
        break
    result = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=result.content))
    print("Bot: " + result.content)
print("Chat History: " + str(chatHistory))

