import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag import answer

REFUSALS = ["don't know", "do not know", "not in the context", "no information",
            "ne dispose pas", "pas d'information", "ne contient pas", "je ne sais pas"]

def score(text: str, expected: str) -> bool:
    text = text.lower()
    if expected == "OUT_OF_SCOPE":
        return any(r in text for r in REFUSALS)
    return all(k.strip().lower() in text for k in expected.split(";"))

def main(label: str) -> None:
    rows = list(csv.DictReader(open(ROOT / "eval/questions.csv", encoding="utf-8")))
    results = []
    for r in rows:
        out = answer(r["question"])
        ok = score(out, r["expected_keywords"])
        results.append({**r, "passed": ok, "answer": out})
        print(f"{'PASS' if ok else 'FAIL'} | {r['lang']} | Q{r['id']} | {r['question']}")

    for lang in ["en", "fr"]:
        sub = [x for x in results if x["lang"] == lang]
        print(f"{lang.upper()}: {sum(x['passed'] for x in sub)}/{len(sub)}")

    out_file = ROOT / f"eval/results_{label}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")