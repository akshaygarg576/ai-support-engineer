import json
from pathlib import Path

from app.agent import run_agent

CASES_PATH = Path(__file__).parent / "cases.json"

def load_cases() -> list[dict]:
    with open(CASES_PATH, "r") as file:
        return json.load(file)

def evaluate_case(case: dict) -> dict:
    result = run_agent(case["question"])

    answer = result["answer"].lower()
    trace = result["trace"]

    tools_used = [
        step["tool"]
        for step in trace
    ]

    # Did the agent use all expected tools?
    tools_correct = all(
        tool in tools_used
        for tool in case["expected_tools"]
    )

    # Does the answer contain expected facts?
    answer_correct = all(
        expected.lower() in answer
        for expected in case["expected_answer_contains"]
    )

    # Did the agent avoid excessive work?
    efficient = (
        len(trace) <= case["max_tool_calls"]
    )

    passed = (
        tools_correct
        and answer_correct
        and efficient
    )

    return {
        "id": case["id"],
        "passed": passed,
        "tools_used": tools_used,
        "tool_calls": len(trace),
        "tools_correct": tools_correct,
        "answer_correct": answer_correct,
        "efficient": efficient,
    }

def main():
    cases = load_cases()

    results = [
        evaluate_case(case)
        for case in cases
    ]

    print("\n--- EVALUATION RESULTS ---")

    for result in results:
        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"{status} | "
            f"{result['id']} | "
            f"tools={result['tools_used']} | "
            f"calls={result['tool_calls']}"
        )

    passed_count = sum(
        result["passed"]
        for result in results
    )

    total_count = len(results)

    pass_rate = (
        passed_count / total_count
        if total_count
        else 0
    )

    print("\n--- SUMMARY ---")

    print(
        f"Passed: {passed_count}/{total_count}"
    )

    print(
        f"Pass rate: {pass_rate:.1%}"
    )

    if passed_count != total_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()