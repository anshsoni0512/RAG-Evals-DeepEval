import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate.from_template(
    '''You are a helpful teaching assistant for a course on LLM evaluations.
Answer the student's question using ONLY the context provided below.

Rules:
- Use only information present in the context. Do not add outside knowledge.
- If the context does not contain enough information to answer, say:
  "I don't have enough information in the course material to answer that."
- Keep the answer clear and concise.

question: {question}

context: {context}'''
)

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    groq_api_key = os.getenv('GROQ_API_KEY'),
    temperature= 0
)

chain = prompt | llm | StrOutputParser()

def generate(query, context):
    return chain.invoke({'question':query, 'context':context})
 
if __name__ == "__main__":
    ctx = [
        "Online eval means evaluating your system on live production traffic "
        "after deployment. It works without an answer key, unlike offline eval."]
    print(generate('what is sonico', ctx))


