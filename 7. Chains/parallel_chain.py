from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
#used to make parallel chains

load_dotenv()

model1 = ChatGroq(model = "openai/gpt-oss-20b")

model2 = ChatGroq(model = "qwen/qwen3.8-27b")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text: {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 questions and answers from the following text: {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided {notes} and {quiz} into a single document",
    input_variables=[{"notes", "quiz"}]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2| parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """"
    Linear regression is a foundational statistical technique and supervised machine learning algorithm used to model and predict a continuous numerical outcome based on its linear relationship with one or more input variables. In simple linear regression, the connection between a single independent predictor and a dependent target is modeled as a straight line defined by a baseline intercept, a slope coefficient measuring how much the target shifts for every one-unit increase in the input, and an error component capturing irreducible random noise; multiple linear regression generalizes this framework across several input features by fitting a multidimensional flat plane across all predictors simultaneously. To find the line or plane of best fit, the algorithm estimates optimal parameter weights either analytically via Ordinary Least Squares or iteratively through gradient descent by minimizing a cost function—typically Mean Squared Error, which computes the average of the squared vertical differences, known as residuals, between observed values and model predictions. For these parameter estimates to be statistically valid and unbiased, the data should broadly satisfy key assumptions including a linear relationship between predictors and target, independence of errors, homoscedasticity (constant variance of residuals across all input levels), normally distributed error terms, and low multicollinearity among independent features. While standard linear regression cannot natively capture complex non-linear curves and can be skewed by extreme outliers, pairing it with Lasso or Ridge regularization mitigates overfitting and stabilizes correlated predictors, keeping it one of the fastest, most transparent, and most widely used tools in finance, healthcare, engineering, and business analytics for applications ranging from housing price estimation and demand forecasting to experimental trend analysis.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()