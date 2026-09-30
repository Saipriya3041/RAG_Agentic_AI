import requests
import json

BASE_URL = "http://127.0.0.1:8000/chat"
REFUSAL = "I cannot answer based on the provided document."

SAMPLE_QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]


def run_sample_queries():
    expected_fields = {
        "query",
        "final_answer",
        "retrieved_context_chunks",
        "confidence_score",
    }
    for query in SAMPLE_QUERIES:
        response = requests.post(BASE_URL, json={"query": query}, timeout=120)
        response.raise_for_status()
        payload = response.json()

        assert set(payload) == expected_fields
        assert payload["query"] == query
        assert isinstance(payload["retrieved_context_chunks"], list)
        assert 0.0 <= payload["confidence_score"] <= 1.0

        if query == "What is the capital of France?":
            assert payload["final_answer"] == REFUSAL
            assert payload["confidence_score"] == 0.0
            assert payload["retrieved_context_chunks"] == []
        else:
            assert payload["final_answer"] != REFUSAL
            assert payload["retrieved_context_chunks"]
            assert payload["confidence_score"] > 0.0

        print(json.dumps({
            "query": query,
            "final_answer": payload["final_answer"],
            "context_chunk_count": len(payload["retrieved_context_chunks"]),
            "confidence_score": payload["confidence_score"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    run_sample_queries()
