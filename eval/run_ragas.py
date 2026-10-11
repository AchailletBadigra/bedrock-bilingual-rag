import csv, os, sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (Faithfulness, ResponseRelevancy,
                           LLMContextPrecisionWithReference, LLMContextRecall)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag import answer_with_context

load_dotenv()
REGION = os.environ["REGION"]

judge = LangchainLLMWrapper(ChatBedrockConverse(
    model=os.environ["JUDGE_MODEL_ID"], region_name=REGION, temperature=0))
try:
    judge.langchain_llm.invoke("Say OK")
except Exception as e:
    sys.exit(f"Judge model not usable: {e}")

embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", region_name=REGION))
embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", region_name=REGION))

METRICS = [Faithfulness(), ResponseRelevancy(),
           LLMContextPrecisionWithReference(), LLMContextRecall()]

def main(label: str = "baseline", k: int = 5) -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "eval/questions.csv", encoding="utf-8"))
            if r["expected_keywords"] != "OUT_OF_SCOPE"]

    samples = []
    for r in rows:
        out = answer_with_context(r["question"], k)
        samples.append({
            "user_input": r["question"],
            "response": out["answer"],
            "retrieved_contexts": out["contexts"],
            "reference": r["ground_truth"],
        })
        print(f"done | {r['lang']} | Q{r['id']}")

    result = evaluate(EvaluationDataset.from_list(samples),
                      metrics=METRICS, llm=judge, embeddings=embeddings)

    df = result.to_pandas()
    df["id"] = [r["id"] for r in rows]
    df["lang"] = [r["lang"] for r in rows]
    print(df.groupby("lang").mean(numeric_only=True).round(2))
    df.to_csv(ROOT / f"eval/ragas_{label}_k{k}.csv", index=False)

if __name__ == "__main__":
    main(*sys.argv[1:2])