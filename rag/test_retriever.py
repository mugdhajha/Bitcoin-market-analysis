from retriever import retrieve

results = retrieve(
    "bitcoin ETF approval"
)

for i, doc in enumerate(results):

    print("\n")
    print("="*50)

    print(
        f"Result {i+1}"
    )

    print(
        doc["meta"]["title"]
    )

    print(
        doc["meta"]["source"]
    )

    print("Score:", doc["score"])