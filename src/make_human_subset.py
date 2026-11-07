import re
import pandas as pd

df = pd.read_csv("data/processed/human_clean.csv")

def extra_clean(t):
    t = re.sub(r"\bOn [^.]{0,80}?wrote:", " ", t)
    t = re.sub(r"--\]|\]\]|&gt;|\|\s", " ", t)
    t = re.sub(r"^\[[\w\- #]+\]\s*", "", t)
    t = re.sub(r"^(re|fw|fwd)\s*:\s*", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

df["text"] = df["text"].map(extra_clean)

MAILING = re.compile(
    r"wrote:|>|\bHTH\b|\bpatch\b|\bkernel\b|\bpython\b|\bperl\b|\blinux\b|"
    r"\bdebian\b|\bgentoo\b|\bcvs\b|\bgcc\b|\brpm\b|\bsourceforge\b|\bbug\b|"
    r"\bmailing list\b|\bunsubscribe\b|\bexmh\b|\bspamassassin\b|\bgetpass\b|"
    r"opensuse|\bILUG\b|\bSA\b|\brules?\b|\bplugins?\b",
    re.IGNORECASE,
)
PHARMA = re.compile(
    r"viagra|cialis|pharmacy|medication|prescription|inches|penis|erection|"
    r"replica|rolex|watches|pills|casino|enlarge|\bsoftware\b.*\bcheap\b",
    re.IGNORECASE,
)
TOPICS = (
    r"\b(?:account|bank|password|verify|verification|login|log in|payment|"
    r"invoice|order|delivery|package|shipping|parcel|job|offer|security|"
    r"transaction|card|refund|confirm|statement|billing|paypal|credit)\b"
)

df = df[~df["text"].str.contains(MAILING)]
df = df[~df["text"].str.contains(PHARMA)]
df = df[df["text"].str.count(TOPICS, flags=re.IGNORECASE) >= 2]

df["n_words"] = df["text"].str.split().str.len()
df = df[(df["n_words"] >= 30) & (df["n_words"] <= 200)]
df = df.drop_duplicates(subset="text")

df.to_csv("data/processed/human_topic.csv", index=False)

print(df.shape)
print(df.groupby(["source", "label_phish"]).size())
print(df.groupby("label_phish")["n_words"].describe())
for lab in [0, 1]:
    print(f"===== label_phish={lab} =====")
    n = min(5, (df.label_phish == lab).sum())
    for t in df[df.label_phish == lab]["text"].sample(n, random_state=3):
        print(t[:300], "\n---")
