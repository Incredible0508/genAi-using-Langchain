from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimension = 30)
#dimension is the size of the embedding vector. It is 30 for text-embedding-3-large model. It is 1536 for text-embedding-3-small model.
result = embedding.embed_query("What is the meaning of life?")
print(result)