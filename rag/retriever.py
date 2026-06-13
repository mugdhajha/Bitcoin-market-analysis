import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "rag/vectorstore/news.index"
)

texts = pickle.load(
    open(
        "rag/vectorstore/texts.pkl",
        "rb"
    )
)

meta = pickle.load(
    open(
        "rag/vectorstore/meta.pkl",
        "rb"
    )
)

def retrieve(query, k=5):

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    D, I = index.search(
        query_embedding,
        k
    )

    results = []

    for distance, idx in zip(D[0], I[0]):

        results.append({
            "score": float(distance),
            "text": texts[idx],
            "meta": meta[idx]
        })

    return results