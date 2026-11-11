from langchain import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.7, max_completion_tokens=10)
#temperature is a parameter that controls the randomness of the model's output. A higher temperature will result in more random outputs, while a lower temperature will result in more deterministic outputs.
#max_completion_tokens is a parameter that controls the maximum number of tokens that the model can generate in a single response. A token is a unit of text, and the number of tokens in a response can vary depending on the length and complexity of the input prompt.

result = model.invoke("Write a poem about the ocean.")

print(result)
#here the result is the content plus the metadata of the response. You can access the content using result.content and the metadata using result.metadata
