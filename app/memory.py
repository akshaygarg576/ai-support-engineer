MEMORIES = []


def save_memory(
    customer_name: str,
    summary: str,
) -> None:
    MEMORIES.append(
        {
            "customer_name": customer_name,
            "summary": summary,
        }
    )


def get_memories(
    customer_name: str,
) -> list[dict]:
    return [
        memory
        for memory in MEMORIES
        if memory["customer_name"] == customer_name
    ]