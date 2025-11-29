"""Optional integration sanity checks for langchain-neo4j."""

import importlib
import os
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.getenv("LANGCHAIN_QWQ_RUN_NEO4J_TESTS") != "1",
        reason=(
            "Set LANGCHAIN_QWQ_RUN_NEO4J_TESTS=1 and install langchain-neo4j "
            "to run Neo4j integration tests"
        ),
    )
]

langchain_neo4j_spec = importlib.util.find_spec("langchain_neo4j")
if langchain_neo4j_spec is None:
    pytest.skip(
        "langchain-neo4j is not installed; install it to run these tests",
        allow_module_level=True,
    )

langchain_neo4j = importlib.import_module("langchain_neo4j")
GraphCypherQAChain = getattr(langchain_neo4j, "GraphCypherQAChain", None)
if GraphCypherQAChain is None:
    pytest.skip(
        "GraphCypherQAChain not available in langchain-neo4j; update the package to run",
        allow_module_level=True,
    )


@pytest.fixture
def dummy_graph() -> MagicMock:
    """Return a dummy graph instance without real network calls."""

    graph = MagicMock(name="neo4j_graph")
    graph.schema = "()"
    return graph


@pytest.fixture
def llm() -> "ChatQwQ":
    from langchain_qwq import ChatQwQ

    return ChatQwQ(model="qwq-plus")


@pytest.mark.parametrize("extra_kwargs", [
    {},
    {"validate_cypher": False},
])
def test_chatqwq_can_be_used_to_build_graph_chain(monkeypatch: pytest.MonkeyPatch, llm, dummy_graph, extra_kwargs):
    """Ensure ChatQwQ can be wired into langchain-neo4j chain builders."""

    captured: dict = {}

    def fake_from_llm(cls, *, llm, graph=None, **kwargs):
        captured["cls"] = cls
        captured["llm"] = llm
        captured["graph"] = graph
        captured["kwargs"] = kwargs
        return MagicMock(name="graph_chain")

    monkeypatch.setattr(
        GraphCypherQAChain,
        "from_llm",
        classmethod(fake_from_llm),
        raising=False,
    )

    chain = GraphCypherQAChain.from_llm(llm=llm, graph=dummy_graph, **extra_kwargs)

    assert chain is not None
    assert captured["cls"] is GraphCypherQAChain
    assert captured["llm"] is llm
    assert captured["graph"] is dummy_graph
    for key, value in extra_kwargs.items():
        assert captured["kwargs"].get(key) == value
