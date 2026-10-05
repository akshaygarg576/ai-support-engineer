import json
from pathlib import Path

from app.embeddings import embed_texts


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "product_docs.json"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "index.json"
)


def load_documents() -> list[dict]:
    with open(
        DOCUMENTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def chunk_documents(
    documents: list[dict],
) -> list[dict]:

    chunks = []

    for document in documents:

        paragraphs = document[
            "content"
        ].split("\n\n")

        for index, paragraph in enumerate(
            paragraphs,
            start=1,
        ):
            chunks.append(
                {
                    "chunk_id": (
                        f"{document['document_id']}"
                        f"-chunk-{index:03d}"
                    ),
                    "document_id": document[
                        "document_id"
                    ],
                    "title": document["title"],
                    "content": paragraph.strip(),
                }
            )

    return chunks


def build_index() -> None:

    documents = load_documents()

    chunks = chunk_documents(
        documents
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = embed_texts(texts)

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        chunk["embedding"] = embedding

    with open(
        INDEX_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            indent=2,
        )

    print(
        f"Saved index to {INDEX_PATH}"
    )


if __name__ == "__main__":
    build_index()