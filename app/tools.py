def get_customer(customer_name: str) -> dict:
    """Return account information for a customer."""

    customers = {
        "Acme": {
            "customer": "Acme",
            "plan": "enterprise",
            "storage_used_gb": 99.8,
            "storage_limit_gb": 100,
        }
    }

    return customers.get(
        customer_name,
        {"error": f"Customer '{customer_name}' not found"},
    )


def search_logs(customer_name: str) -> dict:
    """Return recent application errors for a customer."""

    logs = {
        "Acme": [
            "403 quota_limit",
            "403 quota_limit",
            "403 quota_limit",
        ]
    }

    return {
        "customer": customer_name,
        "errors": logs.get(customer_name, []),
    }


def search_docs(query: str) -> dict:
    """Search product documentation."""

    return {
        "query": query,
        "result": (
            "Uploads are rejected when current storage plus "
            "the incoming file size exceeds the account quota."
        ),
    }


def get_incidents() -> dict:
    """Return currently active service incidents."""

    return {
        "active_incidents": []
    }