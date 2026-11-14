from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text: /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Rani Padmavati and the Sieze of Chittorgarh"})
print(result)