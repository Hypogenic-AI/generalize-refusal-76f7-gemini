import json

with open("papers_raw.json", "r") as f:
    papers = json.load(f)

# Heuristic: Title or abstract contains "refusal" or "generalization" and "model" or "LLM"
relevant_keywords = ["refusal", "over-refusal", "generalization", "safety", "alignment", "finetuning"]

filtered_papers = []
for p in papers:
    score = 0
    text = (p["title"] + " " + p["abstract"]).lower()
    for kw in relevant_keywords:
        if kw in text:
            score += 1
    
    if score >= 2:
        p["score"] = score
        filtered_papers.append(p)

filtered_papers.sort(key=lambda x: x["score"], reverse=True)

with open("papers_filtered.json", "w") as f:
    json.dump(filtered_papers, f, indent=2)

print(f"Filtered to {len(filtered_papers)} relevant papers.")
for p in filtered_papers[:10]:
    print(f"[{p['score']}] {p['title']}")
