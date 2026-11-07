import pandas as pd
from sklearn.model_selection import train_test_split
from cleaning import clean_text

HELD_OUT = "model_C"   # no third model yet, so this is skipped
SEED = 42

human = pd.read_csv("data/processed/human_topic.csv")
human["label_ai"] = 0
human["source_model"] = "human"
human["topic"] = "human"

ai = pd.read_csv("data/raw/ai_generated.csv")
ai["text"] = ai["text"].map(clean_text)
ai["source"] = "ai"
ai["n_words"] = ai["text"].str.split().str.len()
ai = ai[(ai.n_words >= 30) & (ai.n_words <= 200)]
ai = ai.drop_duplicates(subset="text")

held = ai[ai.source_model == HELD_OUT]
ai_train = ai[ai.source_model != HELD_OUT]

cols = ["text", "label_ai", "label_phish", "source", "source_model", "topic", "n_words"]
main_parts, used_idx = [], []

for p in [0, 1]:
    a = ai_train[ai_train.label_phish == p]
    h_pool = human[human.label_phish == p]
    h = h_pool.sample(n=min(len(h_pool), len(a)), random_state=SEED)
    used_idx += list(h.index)
    main_parts += [a, h]

main = pd.concat(main_parts)[cols].reset_index(drop=True)
strat = main["label_ai"].astype(str) + main["label_phish"].astype(str)
use = strat if strat.value_counts().min() >= 10 else None

train, rest = train_test_split(main, test_size=0.30, random_state=SEED, stratify=use)
rest_strat = (rest["label_ai"].astype(str) + rest["label_phish"].astype(str)) if use is not None else None
val, test = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=rest_strat)

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

if len(held):
    leftover = human.drop(index=used_idx)
    parts = [held]
    for p in [0, 1]:
        n = (held.label_phish == p).sum()
        pool = leftover[leftover.label_phish == p]
        parts.append(pool.sample(n=min(n, len(pool)), random_state=SEED))
    pd.concat(parts)[cols].to_csv("data/processed/unseen_test.csv", index=False)
    print("Unseen-model test saved:", len(held), "AI emails from", HELD_OUT)

for name, d in [("train", train), ("val", val), ("test", test)]:
    print(name, d.shape)
    print(d.groupby(["label_ai", "label_phish"]).size().to_dict())
print(main.groupby("label_ai")["n_words"].describe())