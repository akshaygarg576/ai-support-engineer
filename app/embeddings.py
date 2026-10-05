import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
embedding_deployment = os.environ[
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
]


client = OpenAI(
    api_key=api_key,
    base_url=f"{endpoint.rstrip('/')}/openai/v1/",
)


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=embedding_deployment,
        input=text,
    )

    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=embedding_deployment,
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]