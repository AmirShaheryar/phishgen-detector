import re
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

HEADER_LINE = re.compile(
    r"^\s*(from|to|cc|bcc|date|subject|message-id|received|return-path|reply-to|"
    r"in-reply-to|references|mime-version|x-[\w-]+|content-[\w-]+)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

def clean_text(t: str) -> str:
    """Shared cleaner: reuse this exact function on AI-generated emails later."""
    t = str(t)
    t = re.sub(r"<[^>]+>", " ", t)                      # HTML tags
    t = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", t)       # common HTML entities
    t = HEADER_LINE.sub(" ", t)                         # leftover header lines
    t = re.sub(r"Message-ID:\s*\S+", " ", t, flags=re.I)
    t = re.sub(r"^\s*>.*$", " ", t, flags=re.MULTILINE) # quoted replies
    t = re.sub(r"https?://\S+|www\.\S+", " URL ", t)    # links -> token
    t = re.sub(r"\S+@\S+\.\S+", " EMAIL ", t)           # addresses -> token
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load(name, source, has_label=True):
    df = pd.read_csv(RAW / name)
    out = pd.DataFrame({
        "text": df["subject"].fillna("").astype(str) + ". " + df["body"].fillna("").astype(str),
        "label_phish": pd.to_numeric(df["label"], errors="coerce"),
        "source": source,
    })
    return out

frames = [
    load("CEAS_08.csv", "ceas"),
    load("SpamAssasin.csv", "spamassassin"),
    load("Nazario.csv", "nazario"),
    load("Nigerian_Fraud.csv", "nigerian"),
]
df = pd.concat(frames, ignore_index=True)
df = df.dropna(subset=["label_phish"])
df["label_phish"] = df["label_phish"].astype(int)

# remove non-email system messages
df = df[~df["text"].str.contains("FOLDER INTERNAL DATA", case=False)]

df["text"] = df["text"].map(clean_text)
df["n_words"] = df["text"].str.split().str.len()
df = df[(df["n_words"] >= 15) & (df["n_words"] <= 400)]
df = df.drop_duplicates(subset="text")

df["label_ai"] = 0   # all of these predate modern AI writing tools
df = df[["text", "label_ai", "label_phish", "source", "n_words"]]
df.to_csv(OUT / "human_clean.csv", index=False)

print(df.shape)
print(df.groupby(["source", "label_phish"]).size())
print(df.groupby("label_phish")["n_words"].describe())
for lab in [0, 1]:
    print(f"===== label_phish={lab} =====")
    for t in df[df.label_phish == lab]["text"].sample(3, random_state=2):
        print(t[:300], "\n---")