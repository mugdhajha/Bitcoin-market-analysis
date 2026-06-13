from rag_pipeline import ask_rag

answer, docs = ask_rag(
    "What is driving Bitcoin prices today?"
)

print(answer)