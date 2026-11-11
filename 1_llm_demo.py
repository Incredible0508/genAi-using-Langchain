from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model = 'gpt-3.5-turbo-instruct')

result = llm.invoke("Write a poem about the ocean.")
# LLMs take string as input and return a string as output

print(result)