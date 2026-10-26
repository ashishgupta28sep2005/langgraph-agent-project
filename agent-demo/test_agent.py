import pytest

from agent import calculate, graph, run


def test_graph_compiles_and_routes_through_tool():
    result = graph.invoke({"question": "What is 12 * 3?"})
    assert result["answer"] == "The answer is 36."
    assert result["steps"] == 2


def test_mock_mode_needs_no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_MODE", "mock")
    assert run("Hello") == "Mock response: Hello"


def test_mock_mode_supports_modulo():
    assert run("What is 12 % 2?") == "The answer is 0."


def test_calculator_rejects_unsafe_input():
    with pytest.raises(ValueError):
        calculate("__import__('os').system('whoami')")
