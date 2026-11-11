from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimension = 30)
#dimension is the size of the embedding vector. It is 30 for text-embedding-3-large model. It is 1536 for text-embedding-3-small model.

documents = [
    "What is the meaning of life?",
    "What is the capital of France?",
    "What is the largest mammal on Earth?"
]

result = embedding.embed_documents(documents)
print(result)