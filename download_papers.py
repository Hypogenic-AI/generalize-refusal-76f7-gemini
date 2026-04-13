import json
import requests
import os

with open("papers_filtered.json", "r") as f:
    papers = json.load(f)

for p in papers[:5]:
    pdf_url = p["url"]
    file_path = f"papers/{p['id']}_{p['title'][:50].replace(' ', '_').replace('/', '_')}.pdf"
    print(f"Downloading {p['title']}...")
    try:
        r = requests.get(pdf_url, stream=True)
        with open(file_path, "wb") as f_pdf:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f_pdf.write(chunk)
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Failed to download {p['title']}: {e}")
