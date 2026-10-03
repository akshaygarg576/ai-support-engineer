import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

client = OpenAI(
    api_key=api_key,
    base_url=f"{endpoint.rstrip('/')}/openai/v1/",
)

tools = [
    {
        "type": "function",
        "name": "get_customer",
        "description": (
            "Get current account information for a customer, "
            "including their plan and storage usage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "The name of the customer.",
                }
            },
            "required": ["customer_name"],
            "additionalProperties": False,
        },
    }
]

response = client.responses.create(
    model=deployment,
    instructions=(
        "You are a support investigation assistant. "
        "Use the available tools when you need information "
        "from company systems. Do not invent customer data."
    ),
    tools=tools,
    input="What plan is Acme on?",
)


print("\n--- RAW MODEL OUTPUT ---")

for item in response.output:
    print(item)

print("\n--- FUNCTION CALLS ---")

for item in response.output:
    if item.type == "function_call":
        arguments = json.loads(item.arguments)

        print("Tool requested:", item.name)
        print("Arguments:", arguments)
        print("Call ID:", item.call_id)