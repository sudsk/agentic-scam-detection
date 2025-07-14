# backend/agents/policy_guidance/agent.py
"""
Policy Guidance Agent - Provides procedural guidance for fraud scenarios
"""

from google.adk.agents import LlmAgent
from . import prompt

MODEL = "gemini-2.5-flash"

policy_guidance_agent = LlmAgent(
    name="policy_guidance_agent",
    model=MODEL,
    description=(
        "Provides procedural guidance and escalation "
        "recommendations for detected fraud scenarios "
        "based on HSBC policies and regulations"
    ),
    instruction=prompt.POLICY_GUIDANCE_PROMPT,
    output_key="policy_guidance",
    tools=[],
)

root_agent = policy_guidance_agent
