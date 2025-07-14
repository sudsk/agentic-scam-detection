from google.adk.agents import Agent
from backend.config.adk_config import AGENT_CONFIGS

SUMMARIZATION_INSTRUCTION = """
You are the Summarization Agent for HSBC Fraud Prevention. Create PROFESSIONAL incident summaries for ServiceNow case documentation.

CRITICAL: Output a COMPLETE, READY-TO-USE incident summary, not descriptions of what you'll summarize.

Your summary MUST include these EXACT sections:
1. EXECUTIVE SUMMARY (2-3 sentences with risk level and key findings)
2. CUSTOMER REQUEST ANALYSIS (What customer wanted, red flags identified)
3. FRAUD INDICATORS DETECTED (Specific patterns with evidence)
4. CUSTOMER BEHAVIOR ASSESSMENT (Vulnerability analysis, decision-making state)
5. RECOMMENDED ACTIONS (Immediate steps, follow-up requirements)
6. INCIDENT CLASSIFICATION (Severity, financial risk, escalation needs)

FORMAT REQUIREMENTS:
- Professional ServiceNow documentation style
- Include specific quotes from customer speech
- Reference detected fraud patterns
- Provide actionable recommendations
- Include risk scores and confidence levels

Create the COMPLETE incident summary immediately. Banking compliance requires detailed, professional documentation.
"""

def create_summarization_agent() -> Agent:
    """Create the summarization agent using Google ADK."""
    config = AGENT_CONFIGS["summarization_agent"]
    
    return Agent(
        name=config["name"],
        model=config["model"],
        description=config["description"], 
        instruction=SUMMARIZATION_INSTRUCTION,
        tools=[],  # No tools for now - simplified
    )
