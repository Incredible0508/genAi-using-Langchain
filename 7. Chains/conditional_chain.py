from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
#RunnableBranch is used to implement the if/else statement in the chain
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative: \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Generate an appropriate response to this positive feedback: \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Generate an appropriate response to this negative feedback: \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser1),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser1),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback":"This is a terrible smartphone"})

print(result)