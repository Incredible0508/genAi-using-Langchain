from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")
class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""""
    The MacBook Air M3 is an exceptional, whisper-quiet, and ultra-portable laptop boasting incredible battery 
    life and effortless performance for everyday development and productivity.
""")

print(result)


