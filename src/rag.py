import os
import boto3
from dotenv import load_dotenv
from retrieve import get_chunks

load_dotenv()
llm = boto3.client("bedrock-runtime", region_name=os.environ["REGION"])

PROMPT_TEMPLATE = """You are an assistant answering questions about company and medical documents.
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

def answer(question: str, k: int = 5) -> str:
    chunks = get_chunks(question, k)
    return generate(build_prompt(question, chunks))

def answer_with_context(question: str, k: int = 5) -> dict:
    chunks = get_chunks(question, k)
    return {
        "answer": generate(build_prompt(question, chunks)),
        "contexts": [c["text"] for c in chunks],
    }

if __name__ == "__main__":
       for q in ["What are the very common adverse reactions of Ozempic?",
             "Quels sont les effets indésirables très fréquents d'Ozempic ?"]:
        print(f"\n=== {q}\n")
        print(answer(q))
    
