from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEndpoint
from Insights_Relevant_Paper_Aggregator import llm_input_aggregator

# 🧠 Get model prediction summary + supporting references
llm_input = llm_input_aggregator()

# 🧾 Prompt Template
template = """
You are a clinical AI assistant trained to interpret fetal health predictions using academic literature.

Given the following context:

{llm_input}

Your task is to write 1–3 concise clinical paragraphs that:

- Summarize the model's predicted class and probability.
- Identify the most influential SHAP features and explain whether each one supports or contradicts the model’s prediction.
- Use academic sources from the context to justify each feature’s role in the prediction. For example, if "prolonged decelerations" is a negative SHAP feature, explain how the literature supports or refutes its impact on fetal health.
- Use APA-style in-text citations when referencing evidence (e.g., Smith et al., 2020 or *Journal Title*, 2019).
- Include a short reference list at the end using APA format.

Use natural, professional clinical language. Do not include bullet points, headings, or excerpts. Only write narrative paragraphs. Do not invent references or cite anything not present in the provided context.

Keep the total explanation under 400 words.
"""

prompt = PromptTemplate.from_template(template)

# ✅ Use a model that supports text-generation
llm = HuggingFaceEndpoint(
    model="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.5,
    repetition_penalty=1.2,
    top_p=0.95,
)

# 🔗 LangChain Runnable Pipeline
chain: Runnable = prompt | llm

# ▶️ Invoke
response = chain.invoke({"llm_input": llm_input})

# 📝 Final Output
print("\n📝 Final Explanation:")
print(response)
