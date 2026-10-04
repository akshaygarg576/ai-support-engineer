from app.agent import run_agent

def evaluate_customer_plan_lookup():
    result = run_agent(
        "What plan is Acme on?"
    )

    answer = result["answer"].lower()
    trace = result["trace"]

    tools_used = [
        step["tool"]
        for step in trace
    ]

    print("\n--- EVAL TRACE ---")

    for step in trace:
        print(
            f"Round {step['round']}: "
            f"{step['tool']}({step['arguments']})"
        )

    print("\n--- EVAL RESULT ---")

    # -----------------------------------------------------
    # Check 1:
    # Did the agent use the correct company data source?
    # -----------------------------------------------------

    used_customer_tool = "get_customer" in tools_used

    # -----------------------------------------------------
    # Check 2:
    # Did the final answer contain the expected fact?
    # -----------------------------------------------------

    answer_is_correct = "enterprise" in answer

    # -----------------------------------------------------
    # Check 3:
    # Did it avoid excessive tool usage?
    # -----------------------------------------------------

    efficient_enough = len(trace) <= 2

    print("Used get_customer:", used_customer_tool)
    print("Correct answer:", answer_is_correct)
    print("Efficient enough:", efficient_enough)

    passed = (
        used_customer_tool
        and answer_is_correct
        and efficient_enough
    )

    print("\nPASSED:", passed)

    return passed


if __name__ == "__main__":
    passed = evaluate_customer_plan_lookup()

    if not passed:
        raise SystemExit(1)