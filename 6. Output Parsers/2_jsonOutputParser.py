from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give the name, age and city of a fictional character. \n {format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt = template.format();
result = model.invoke(prompt)

final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))
