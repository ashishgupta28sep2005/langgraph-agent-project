"""A small LangGraph agent with deterministic and OpenAI modes."""

from __future__ import annotations

import logging
import os
import re
from typing import Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    answer: str
    expression: str
    tool_result: str
    steps: int
    mode: str


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression such as 12 * 3."""
    return calculate(expression)


def calculate(expression: str) -> str:
    """Evaluate arithmetic with a deliberately small safe grammar."""
    if not re.fullmatch(r"\s*\d+(?:\s*[+*/%\-]\s*\d+)+\s*", expression):
        raise ValueError("Only basic arithmetic expressions are supported")
    tokens = re.findall(r"\d+|[+*/%\-]", expression.replace(" ", ""))
    total = float(tokens[0])
    for operator, operand in zip(tokens[1::2], tokens[2::2]):
        value = float(operand)
        if operator == "+":
            total += value
        elif operator == "-":
            total -= value
        elif operator == "*":
            total *= value
        elif operator == "/":
            if value == 0:
                raise ValueError("Cannot divide by zero")
            total /= value
        elif operator == "%":
            if value == 0:
                raise ValueError("Cannot take modulo by zero")
            total %= value
    return str(int(total) if total.is_integer() else total)


def plan(state: AgentState) -> AgentState:
    """Use a deterministic plan or ask the configured LLM which tool to call."""
    question = state["question"]
    if state.get("mode") == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")).bind_tools([calculator])
        message = model.invoke(question)
        if message.tool_calls:
            call = message.tool_calls[0]
            if call["name"] != "calculator":
                raise ValueError(f"Unsupported tool requested: {call['name']}")
            return {"expression": call["args"]["expression"], "steps": 1}
        return {"answer": str(message.content), "steps": 1}

    match = re.search(r"(?:what is|calculate|compute)\s+(.+?)[?.!]*$", question, re.I)
    if match:
        return {"expression": match.group(1), "steps": 1}
    return {"answer": f"Mock response: {question}", "steps": 1}


def calculator_tool(state: AgentState) -> AgentState:
    return {"tool_result": calculate(state["expression"])}


def respond(state: AgentState) -> AgentState:
    return {"answer": f"The answer is {state['tool_result']}.", "steps": state["steps"] + 1}


def route(state: AgentState) -> Literal["respond", "finish"]:
    return "respond" if "expression" in state else "finish"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", plan)
    builder.add_node("calculator", calculator_tool)
    builder.add_node("respond", respond)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route, {"respond": "calculator", "finish": END})
    builder.add_edge("calculator", "respond")
    builder.add_edge("respond", END)
    return builder.compile()


graph = build_graph()


def run(question: str, mode: str | None = None) -> str:
    selected_mode = mode or os.getenv("AGENT_MODE", "mock").lower()
    if selected_mode == "mock":
        result = graph.invoke({"question": question, "mode": "mock"})
        logger.info("Mock graph completed in %s step(s)", result["steps"])
        return result["answer"]
    if selected_mode != "openai":
        raise ValueError("AGENT_MODE must be 'mock' or 'openai'")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or use AGENT_MODE=mock")
    try:
        result = graph.invoke({"question": question, "mode": "openai"})
        logger.info("OpenAI graph completed in %s step(s)", result["steps"])
        return result["answer"]
    except Exception as exc:
        logger.exception("OpenAI request failed")
        raise RuntimeError("OpenAI request failed; check credentials and connectivity") from exc


if __name__ == "__main__":
    try:
        prompt = input("Ask a question (or 'quit'): ").strip()
        while prompt.lower() not in {"quit", "exit"}:
            if prompt:
                try:
                    print(run(prompt))
                except (RuntimeError, ValueError) as exc:
                    print(f"Could not process that request: {exc}")
            prompt = input("Ask a question (or 'quit'): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye")
