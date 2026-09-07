"""ReAct agent factory (LangChain tool-calling agent).

Only imported when the user actually launches the agent, so the rest of the
platform works without an API key. Set OPENAI_API_KEY in .env (see .env.example).
"""

from __future__ import annotations

from typing import Any

from atml.agent.session import AgentSession
from atml.agent.tools import build_tools
from atml.config import load_config

SYSTEM_PROMPT = """You are a careful data scientist operating a tabular ML platform.
You can only act through the provided tools; you never see raw data.
Workflow: inspect first, fix data quality issues with the smallest reasonable change,
explain each step briefly, then train a model only if the user asked for one.
If a tool refuses (MLGuard, formula guard), explain why and adapt instead of retrying blindly.
Never invent numbers; quote only what tools returned."""


def build_agent(session: AgentSession) -> Any:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    cfg = load_config()["agent"]
    llm = ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    tools = build_tools(session)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=cfg["max_iterations"],
        return_intermediate_steps=True,
        verbose=False,
    )


def run_agent(session: AgentSession, task: str) -> dict[str, Any]:
    """Run one task and return {'output': str, 'steps': [(tool, input, observation), ...]}."""
    executor = build_agent(session)
    result = executor.invoke({"input": task})
    steps = [
        {"tool": action.tool, "input": action.tool_input, "observation": str(obs)[:2000]}
        for action, obs in result.get("intermediate_steps", [])
    ]
    return {"output": result["output"], "steps": steps}
