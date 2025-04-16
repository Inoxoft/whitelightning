import pandas as pd

ALLOWED_LABELS = {
    "Politics", "Sports", "Business", "World", "Technology",
    "Entertainment", "Science", "Health", "Education", "Environment"
}

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines='skip')
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.replace('"', '').str.replace("'", '')
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(ALLOWED_LABELS)]
    return df

