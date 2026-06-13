import pandas as pd
import numpy as np
import pickle
import faiss

from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/latest_news.csv")

texts = (
    df["title"].fillna("")
    + "\n"
    + df["description"].fillna("")
    + "\n"
    + df["content"].fillna("")
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = model.encode(
    texts.tolist(),
    show_progress_bar=True
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

faiss.write_index(
    index,
    "rag/vectorstore/news.index"
)

pickle.dump(
    texts.tolist(),
    open(
        "rag/vectorstore/texts.pkl",
        "wb"
    )
)

pickle.dump(
    df.to_dict("records"),
    open(
        "rag/vectorstore/meta.pkl",
        "wb"
    )
)

print("Documents:", len(texts))
print("Embedding Shape:", embeddings.shape)
print("Vector store built successfully!")