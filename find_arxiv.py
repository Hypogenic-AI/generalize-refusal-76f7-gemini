import arxiv
import json
import os

keywords = [
    "refusal generalization",
    "model over-refusal",
    "safety alignment LLM",
    "fine-tuning refusal",
    "adversarial refusal",
    "benign request refusal"
]

results = []
client = arxiv.Client()

for query in keywords:
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance
    )
    for res in client.results(search):
        paper = {
            "title": res.title,
            "authors": [a.name for a in res.authors],
            "year": res.published.year,
            "url": res.pdf_url,
            "abstract": res.summary,
            "id": res.entry_id.split("/")[-1]
        }
        results.append(paper)

# Remove duplicates
unique_results = {p["id"]: p for p in results}.values()

with open("papers_raw.json", "w") as f:
    json.dump(list(unique_results), f, indent=2)

print(f"Found {len(unique_results)} unique papers.")
