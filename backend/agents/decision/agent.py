# backend/agents/decision/agent.py
"""
Decision Agent - Makes final fraud prevention decisions
"""

from google.adk.agents import LlmAgent
from . import prompt

MODEL = "gemini-2.5-flash"

decision_agent = LlmAgent(
    name="decision_agent",
    model=MODEL,
    description=(
        "Makes final fraud prevention decisions based on "
        "multi-agent analysis, risk assessment, and "
        "policy guidance for immediate action"
    ),
    instruction=prompt.DECISION_AGENT_PROMPT,
    output_key="decision_result",
    tools=[],
)

root_agent = decision_agent
