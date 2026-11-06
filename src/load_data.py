import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

for f in RAW.rglob("*.csv"):
    df = pd.read_csv(f, nrows=3)
    print(f.name, list(df.columns))