import os
import boto3
from dotenv import load_dotenv

load_dotenv()
client = boto3.client("bedrock-agent-runtime", region_name=os.environ["REGION"])

def retrieve(question: str, k: int = 5) -> dict:
    return client.retrieve(
        knowledgeBaseId=os.environ["KB_ID"],
        retrievalQuery={"text":question},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": k}
        },
    )

def get_chunks(question: str, k: int = 5) -> list[dict]:
    response = retrieve(question, k)
    chunks = []
    for r in response["retrievalResults"]:
        chunks.append({
            "text": r["content"]["text"],
            "source": r["location"]["s3Location"]["uri"],
            "page": r["metadata"].get("x-amz-bedrock-kb-document-page-number"),
            "score": r["score"],
        })
    return chunks

if __name__ == "__main__":
    for q in ["What are Euroclear's main risks?",
              "Quels sont les principaux risques d'Euroclear ?"]:
        print(f"\n=== {q}")
        for c in get_chunks(q):
            print(f"{c['score']:.3f} | p.{c['page']} | {c['source'].split('/')[-1]} | {c['text'][:100]}")