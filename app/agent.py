import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tools import get_customer

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
        "strict": True,
    }
]


# ---------------------------------------------------------
# Tool execution - aka "Dispatcher"

# @todo: later add a "Tool Registry" when there are multiple tools, instead of if..else if
# TOOL_REGISTRY = {
#     "get_customer": get_customer,
#     "search_logs": search_logs,
#     "search_docs": search_docs,
#     "get_incidents": get_incidents,
# }
# ---------------------------------------------------------

def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool requested by the model."""

    if tool_name == "get_customer":
        return get_customer(
            customer_name=arguments["customer_name"]
        )

    raise ValueError(f"Unknown tool: {tool_name}")


response = client.responses.create(
    model=deployment,
    instructions=(
        "You are a support investigation assistant. "
        "Use available tools when you need information "
        "from company systems. "
        "Do not invent customer data."
    ),
    tools=tools,
    input="What plan is Acme on?",
)


print("\n--- MODEL DECISION ---")


tool_outputs = []

for item in response.output:

    if item.type != "function_call":
        continue

    arguments = json.loads(item.arguments)

    print("Tool requested:", item.name)
    print("Arguments:", arguments)

    # -----------------------------------------------------
    # Our application executes the requested tool
    # -----------------------------------------------------

    result = execute_tool(
        tool_name=item.name,
        arguments=arguments,
    )

    print("Tool result:", result)

    # -----------------------------------------------------
    # Prepare the observation for the model
    # -----------------------------------------------------

    tool_outputs.append(
        {
            "type": "function_call_output",
            # identifies which tool request does this tool result belong to?
            "call_id": item.call_id,
            "output": json.dumps(result),
        }
    )


if not tool_outputs:
    raise RuntimeError(
        "Expected the model to request a tool, but it did not."
    )


# ---------------------------------------------------------
# Send the tool result back to the model
# ---------------------------------------------------------

final_response = client.responses.create(
    model=deployment,
    # Continue from that previous model response.
    # In other words, we are "chaining responses"
    previous_response_id=response.id,
    input=tool_outputs,
)


# ---------------------------------------------------------
# Model now has enough information to answer
# ---------------------------------------------------------

print("\n--- FINAL ANSWER ---")
print(final_response.output_text)