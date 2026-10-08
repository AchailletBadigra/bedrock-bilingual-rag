import os
import boto3
from dotenv import load_dotenv
from retrieve import get_chunks

load_dotenv()
llm = boto3.client("bedrock-runtime", region_name=os.environ["REGION"])

PROMPT_TEMPLATE = """You are an assistant answering questions about annual reports.
Answer ONLY from the context below.
Answer in the same language as the question.
If the answer is not in the context, say you don't know.
Cite the source after each fact, like [file p.page].

Context:
{context}

Question: {question}
"""

def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['source'].split('/')[-1]} p.{c['page']}]\n{c['text']}" for c in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)

def generate(prompt: str) -> str:
    response = llm.converse(
        modelId=os.environ["MODEL_ARN"],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )
    return response["output"]["message"]["content"][0]["text"]

def answer(question: str) -> str:
    chunks = get_chunks(question)
    return generate(build_prompt(question, chunks))

if __name__ == "__main__":
    print(answer("What are Euroclear's main risks?"))
    print("\n---\n")
    print(answer("Quels sont les principaux risques d'Euroclear ?"))
    
