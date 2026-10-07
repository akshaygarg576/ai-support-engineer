import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tools import (
    get_customer,
    get_incidents,
    search_docs,
    search_logs,
)


# =========================================================
# Configuration
# =========================================================

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

client = OpenAI(
    api_key=api_key,
    base_url=f"{endpoint.rstrip('/')}/openai/v1/",
)


# =========================================================
# Agent instructions
# =========================================================

AGENT_INSTRUCTIONS = """
You are a support investigation assistant.

Investigate customer issues using the available tools and provide
an evidence-based answer.

Available information sources:
- get_customer: customer/account information
- search_logs: recent runtime errors for a customer
- search_docs: product documentation and expected product behavior
- get_incidents: currently active service incidents

Investigation rules:

1. Decide what information you need before choosing a tool.

2. Use the source that best matches the information needed.

3. Treat tool results as evidence.

4. Never invent customer data, logs, incidents, or product behavior.

5. When documentation is needed:
   - create a focused search query
   - inspect the retrieved evidence
   - if the evidence is insufficient, ambiguous, or irrelevant,
     search again with a better query
   - avoid repeating essentially the same search

6. Correlate evidence from different sources when useful.

7. Do not treat a hypothesis as a confirmed root cause unless
   the available evidence supports it.

8. Stop calling tools when enough evidence exists to answer.

9. If the available evidence is insufficient, explain what
   additional information would be needed.

Keep investigations concise and avoid unnecessary tool calls.
"""


# =========================================================
# Tool definitions visible to the model
# =========================================================

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
    },
    {
        "type": "function",
        "name": "search_logs",
        "description": (
            "Search recent application errors for a customer. "
            "Use this when investigating failures or unexpected behavior."
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
    },
    {
        "type": "function",
        "name": "search_docs",
        "description": (
            "Search product documentation for product rules, "
            "expected behavior, limits, and troubleshooting information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The product behavior or issue to search for."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_incidents",
        "description": (
            "Get currently active service-wide incidents. "
            "Use this to determine whether a customer problem "
            "may be caused by a broader outage."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# =========================================================
# Tool registry
# =========================================================

TOOL_REGISTRY = {
    "get_customer": get_customer,
    "search_logs": search_logs,
    "search_docs": search_docs,
    "get_incidents": get_incidents,
}


# =========================================================
# Deterministic tool execution
# =========================================================

def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute an explicitly allowed tool."""

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        raise ValueError(f"Unknown tool: {tool_name}")

    return tool(**arguments)


# =========================================================
# Agent loop
# =========================================================

def run_agent(user_input: str) -> dict:

    max_tool_rounds = 5

    trace = []

    response = client.responses.create(
        model=deployment,
        instructions=AGENT_INSTRUCTIONS,
        tools=tools,
        parallel_tool_calls=False,
        input=user_input,
    )

    for tool_round in range(1, max_tool_rounds + 1):

        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        # Model has stopped requesting tools.
        if not function_calls:
            return {
                "answer": response.output_text,
                "trace": trace,
            }

        tool_outputs = []

        for function_call in function_calls:

            arguments = json.loads(function_call.arguments)

            result = execute_tool(
                tool_name=function_call.name,
                arguments=arguments,
            )

            # ---------------------------------------------
            # Record what actually happened.
            # ---------------------------------------------

            trace.append(
                {
                    "round": tool_round,
                    "tool": function_call.name,
                    "arguments": arguments,
                    "result": result,
                }
            )

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": json.dumps(result),
                }
            )

        response = client.responses.create(
            model=deployment,
            instructions=AGENT_INSTRUCTIONS,
            tools=tools,
            parallel_tool_calls=False,
            previous_response_id=response.id,
            input=tool_outputs,
        )

    function_calls = [
        item
        for item in response.output
        if item.type == "function_call"
    ]

    if not function_calls:
        return {
            "answer": response.output_text,
            "trace": trace,
        }

    raise RuntimeError(
        f"Agent exceeded maximum tool rounds: {max_tool_rounds}"
    )


if __name__ == "__main__":

    result = run_agent(
        "What are the possible reasons file uploads can fail?"
    )

    print("\n--- TRACE ---")

    for step in result["trace"]:
        print(
            f"Round {step['round']}: "
            f"{step['tool']}({step['arguments']})"
        )
        print("Result:", step["result"])

    print("\n--- FINAL ANSWER ---")
    print(result["answer"])