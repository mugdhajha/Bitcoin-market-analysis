from dotenv import load_dotenv
import os
import google.generativeai as genai

from rag.retriever import retrieve

load_dotenv()


genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def ask_rag(query):
    print("STEP 1: retrieve start")

    docs = retrieve(query)

    print("STEP 2: retrieve complete")

    context = "\n\n".join(
        f"""
    Title: {doc['meta']['title']}

    Source: {doc['meta']['source']}

    Content:
    {doc['text']}
    """
        for doc in docs
    )

    prompt = f"""
You are an expert Bitcoin market analyst.

Answer ONLY using the retrieved news articles.

Retrieved Context:
{context}

Question:
{query}

Provide:

1. Summary
2. Key Drivers
3. Market Impact

If the information is not present in the retrieved articles,
say so clearly.
"""
    print("calling gemini")

    try:
        response = model.generate_content(
            prompt
        )
        print("SUCESS")

        print("gemini returned")

        return response.text, docs
    except Exception as e:
        print("GEMINI ERROR:",e)
        raise e