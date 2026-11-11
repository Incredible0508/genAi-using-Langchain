from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#metrics to calculate similarity between two vectors. cosine_similarity is used to calculate the cosine similarity between two vectors. It returns a value between -1 and 1. 1 means the vectors are identical, 0 means the vectors are orthogonal, and -1 means the vectors are opposite.

load_dotenv()
documents = ["Greece has a rich history and culture, with many ancient ruins and landmarks.",
             "The capital of France is Paris, which is known for its art, fashion, and cuisine.",
             "India is a diverse country with a rich cultural heritage, known for its festivals, cuisine, and historical landmarks.", 
             "China is a country with a long history and rich culture, known for its ancient landmarks, cuisine, and festivals.",
             "Lake Baikal is the world's deepest and oldest freshwater lake, located in Siberia, Russia. It is home to a unique ecosystem and many endemic species.",]

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

query = "What is the capital of France?"
document_vectors = embedding.embed_documents(documents)

scores = cosine_similarity([embedding.embed_query(query)], document_vectors)[0]

#max value of similarity score is 1.0 which means the query and the document are identical. The higher the score, the more similar the document is to the query. The output of cosine_similarity is a 2D array with shape (1, n) where n is the number of documents. We use [0] to get the first element of the array which is a 1D array with shape (n,) containing the similarity scores between the query and each document in the documents list.
#[0] is used to get the first element of the array returned by cosine_similarity. The output is a list of similarity scores between the query and each document in the documents list. The higher the score, the more similar the document is to the query.

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

#enumerate is used to get the index of the document along with its similarity score. sorted is used to sort the list of tuples based on the similarity score in descending order. The first element of the sorted list is the document with the highest similarity score.
#reverse=True is used to sort the list in descending order. The output is a list of tuples, where each tuple contains the index of the document and its similarity score with the query. The list is sorted in descending order of similarity scores.
#using list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0], the output of cosine_similarity is converted to a list of tuples using list(enumerate(scores)). Each tuple contains the index of the document and its similarity score with the query. The list is then sorted in descending order of similarity scores using sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True). The first element of the sorted list is the document with the highest similarity score, which is extracted using [0].

print("Query: ", query)
print("Most Similar Document: ", documents[index])
print("Similarity Score: ", score)