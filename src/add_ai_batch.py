import re
import sys
from pathlib import Path
import pandas as pd

# usage: python src/add_ai_batch.py batch.txt model_A bank_alert 1
#        args: file, model name, topic, label_phish (1 = phishing, 0 = legit)
path, model, topic, phish = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

raw = Path(path).read_text(encoding="utf-8-sig")
raw = re.sub(r"<search_triggering>.*?</search_triggering>", "", raw, flags=re.S)
emails = [e.strip() for e in raw.split("###") if len(e.strip().split()) >= 15]

rows = pd.DataFrame({
    "text": emails,
    "label_ai": 1,
    "label_phish": phish,
    "source_model": model,
    "topic": topic,
})

out = Path("data/raw/ai_generated.csv")
rows.to_csv(out, mode="a", header=not out.exists(), index=False)
print(f"Added {len(rows)} emails from {model} ({topic}, phish={phish})")