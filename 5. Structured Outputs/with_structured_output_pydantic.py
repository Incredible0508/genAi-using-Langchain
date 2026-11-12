from langchain_groq import ChatGroq
from typing import Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review into the list")
    summary: str = Field(description="Generate a brief summary")
    sentiment : Literal["P", "N"] = Field(description="Return sentiment of the review either in positive(P) or negative(N)")
    pros: Optional[list[str]] = Field(default=None,description="Write all the pros into the list") 
    cons: Optional[list[str]] = Field(default=None,description="Write all the cons into the list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer and not the productß")

structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""
    Product : Samsung 8 kg Front Load Washing Machine

    This washing machine has been an absolute lifesaver for managing laundry during hostel life. It runs remarkably quiet, 
    handles heavy loads without vibrating excessively, and the steam wash feature gets rid of tough stains effortlessly. 
    Water and energy efficiency are massive bonuses, and the digital control panel is super intuitive to use every day.

    by Ayush Mishra

""")

print(result)
print(result.name)