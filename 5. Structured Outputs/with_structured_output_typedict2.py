from langchain_groq import ChatGroq
from typing import Annotated, TypedDict, Optional, Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review into the list"]
    summary: Annotated[str, "Generate a brief summary"]
    sentiment: Annotated[Literal["P", "N"], "Return sentiment of the review either in positive(P) or negative(N)"]
    pros: Annotated[Optional[list[str]], "Write all the pros into the list"]
    cons: Annotated[Optional[list[str]], "Write all the cons into the list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
    **Product Reviewed:** Samsung 8 kg Front Load Washing Machine

This washing machine has been an absolute lifesaver for managing laundry during hostel life. It runs remarkably quiet, 
handles heavy loads without vibrating excessively, and the steam wash feature gets rid of tough stains effortlessly. 
Water and energy efficiency are massive bonuses, and the digital control panel is super intuitive to use every day.

""")

print(result)