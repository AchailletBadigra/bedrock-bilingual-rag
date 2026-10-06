import os
import boto3
from dotenv import load_dotenv

load_dotenv()
client = boto3.client("bedrock-agent-runtime", region_name=os.environ["REGION"])

def ask(question: str, max_results: int = 5) -> dict:
    return client.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": os.environ["KB_ID"],
                "modelArn": os.environ["MODEL_ARN"],
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": max_results}
                },
            },
        },
    )

def show(response: dict) -> None:
    print("ANSWER:\n", response["output"]["text"])
    print("\nSOURCES:")
    for citation in response["citations"]:
        for ref in citation["retrievedReferences"]:
            print(" -", ref["location"]["s3Location"]["uri"])
if __name__ == "__main__":
    show(ask("What are the main risks mentioned in the annual report?"))