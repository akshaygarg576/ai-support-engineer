import os

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env into the process environment.
load_dotenv()


endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]


client = OpenAI(
    api_key=api_key,
    base_url=f"{endpoint.rstrip('/')}/openai/v1/",
)


response = client.responses.create(
    model=deployment,
    input="Say hello from our AI Support Engineer project in one sentence.",
)


print(response.output_text)