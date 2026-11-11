from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = ["How old is the universe?",
             "What is the capital of France?",
             "What is the largest mammal on Earth?"]

vector = embedding.embed_documents(documents)

print(str(vector))