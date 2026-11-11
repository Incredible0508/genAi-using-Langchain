from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

result = model.invoke("Who is the Prime Minister of India?", temperature=0.7, max_completion_tokens=10)
#invoke method takes a string as input and returns a string as output. It also takes optional parameters like temperature and max_completion_tokens which are used to control the output of the model.
print(result)