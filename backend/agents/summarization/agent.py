# backend/agents/summarization/agent.py
"""
Summarization Agent - Creates professional incident summaries
"""

from google.adk.agents import LlmAgent
from . import prompt

MODEL = "gemini-2.5-flash"

summarization_agent = LlmAgent(
    name="summarization_agent",
    model=MODEL,
    description=(
        "Creates professional incident summaries for "
        "ServiceNow case documentation and compliance "
        "reporting in banking fraud prevention"
    ),
    instruction=prompt.SUMMARIZATION_PROMPT,
    output_key="incident_summary",
    tools=[],
)

root_agent = summarization_agent
