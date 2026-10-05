import json
import math
from pathlib import Path

from app.embeddings import embed_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "index.json"
)


def load_index() -> list[dict]:
    with open(
        INDEX_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    # Conceptually,
    # similarity = (A.B)/(length(A) * length(B))
    # where A.B is the dot product of vector A and vector B,
    # and length(A) is same as magnitude_a

    # In other words, 
    # we only care about the direction of vectors and not length of vector.
    # So we neutralize the length or magnitude before comparing 
    # the vectors for semantic similarity.

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(
            value * value
            for value in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            value * value
            for value in vector_b
        )
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return (
        dot_product
        / (magnitude_a * magnitude_b)
    )


def search_product_docs(
    query: str,
    top_k: int = 3,
) -> list[dict]:

    index = load_index()

    query_embedding = embed_text(
        query
    )

    scored_chunks = []

    for chunk in index:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk[
                    "document_id"
                ],
                "title": chunk["title"],
                "content": chunk["content"],
                "score": score,
            }
        )

    scored_chunks.sort(
        key=lambda chunk: chunk["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]


if __name__ == "__main__":

    results = search_product_docs(
        "Why can't this user upload files?"
    )

    for result in results:

        print()
        print(
            "Score:",
            round(result["score"], 4),
        )

        print(
            "Chunk:",
            result["chunk_id"],
        )

        print(
            "Content:",
            result["content"],
        )