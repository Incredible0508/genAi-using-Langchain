from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
#huggingface pipeline is used to generate text from the model. it is different from HuggingsFaceEndpoint which is used to call the model hosted on huggingface hub. HuggingFacePipeline is a wrapper around the huggingface pipeline which allows us to use it with langchain.
llm = HuggingFacePipeline(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #repo_id is the model name on HuggingFace Hub
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=100,
        temperature=0.7
    )
    #pipeline_kwargs are the parameters for the HuggingFace pipeline which is used to generate text
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the meaning of life?")
print(result)
