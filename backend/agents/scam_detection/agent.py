# backend/agents/scam_detection/agent.py
"""
Scam Detection Agent - Analyzes customer speech for fraud patterns
"""

from google.adk.agents import LlmAgent
from . import prompt

MODEL = "gemini-2.5-flash"

scam_detection_agent = LlmAgent(
    name="scam_detection_agent",
    model=MODEL,
    description=(
        "Analyzes customer speech for fraud patterns, "
        "calculates risk scores, and identifies scam types "
        "in real-time banking conversations"
    ),
    instruction=prompt.SCAM_DETECTION_PROMPT,
    output_key="fraud_analysis",
    tools=[],
)

root_agent = scam_detection_agent
