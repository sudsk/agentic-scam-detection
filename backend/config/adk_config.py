# backend/config/adk_config.py - Configuration for Google ADK Fraud Detection Agents

AGENT_CONFIGS = {
    "scam_detection_agent": {
        "name": "scam_detection_agent",
        "model": "gemini-2.0-flash",  # Fast model for real-time fraud detection
        "description": "Analyzes customer speech for fraud patterns and calculates risk scores"
    },
    
    "policy_guidance_agent": {
        "name": "policy_guidance_agent", 
        "model": "gemini-2.0-flash",  # Pro model for complex policy reasoning
        "description": "Provides procedural guidance and escalation recommendations for fraud scenarios"
    },
    
    "summarization_agent": {
        "name": "summarization_agent",
        "model": "gemini-2.0-flashh",  # Fast model for summary generation
        "description": "Creates professional incident summaries for ServiceNow case documentation"
    },
    
    "orchestrator_agent": {
        "name": "orchestrator_agent",
        "model": "gemini-2.0-flash",  # Pro model for complex decision-making
        "description": "Coordinates multi-agent analysis and makes final fraud prevention decisions"
    },
    
    "decision_agent": {
        "name": "decision_agent",  # Add this for the renamed decision agent
        "model": "gemini-2.0-flash",
        "description": "Makes final fraud prevention decisions based on multi-agent analysis"
    }
}

# Tool configurations for each agent
TOOL_CONFIGS = {
    "fraud_pattern_analyzer": {
        "description": "Analyzes text for known fraud patterns and social engineering tactics",
        "parameters": ["customer_text", "context"],
        "returns": "detected_patterns_with_confidence"
    },
    
    "risk_calculator": {
        "description": "Calculates fraud risk score based on detected patterns and customer behavior",
        "parameters": ["patterns", "customer_profile", "transaction_details"],
        "returns": "risk_score_and_recommendation"
    },
    
    "policy_database_search": {
        "description": "Searches HSBC fraud prevention policies for specific scenarios",
        "parameters": ["scam_type", "risk_level", "customer_segment"],
        "returns": "applicable_policies_and_procedures"
    },
    
    "escalation_procedures": {
        "description": "Retrieves escalation procedures and contact information for fraud cases",
        "parameters": ["risk_score", "scam_type", "urgency_level"],
        "returns": "escalation_path_and_contacts"
    },
    
    "transcript_analyzer": {
        "description": "Analyzes call transcripts for key information and evidence",
        "parameters": ["transcript_segments", "speaker_identification"],
        "returns": "key_quotes_and_evidence"
    },
    
    "incident_formatter": {
        "description": "Formats incident data for ServiceNow case creation",
        "parameters": ["fraud_analysis", "policy_guidance", "customer_info"],
        "returns": "formatted_incident_summary"
    },
    
    "agent_coordinator": {
        "description": "Coordinates inputs from multiple agents for decision-making",
        "parameters": ["scam_analysis", "policy_guidance", "incident_summary"],
        "returns": "coordinated_agent_inputs"
    },
    
    "decision_engine": {
        "description": "Makes final fraud prevention decisions based on all available evidence",
        "parameters": ["risk_assessment", "policy_requirements", "customer_impact"],
        "returns": "final_decision_with_actions"
    }
}

# Risk thresholds for decision-making
RISK_THRESHOLDS = {
    "critical": 80,      # BLOCK_AND_ESCALATE
    "high": 60,          # VERIFY_AND_MONITOR  
    "medium": 40,        # MONITOR_AND_EDUCATE
    "low": 20            # CONTINUE_NORMAL
}

# Escalation paths based on risk and scam type
ESCALATION_PATHS = {
    "critical": {
        "team": "Financial Crime Team",
        "contact": "fraud.urgent@hsbc.com",
        "sla": "15 minutes",
        "actions": ["immediate_transaction_block", "account_freeze", "customer_contact"]
    },
    "high": {
        "team": "Fraud Prevention Team", 
        "contact": "fraud.prevention@hsbc.com",
        "sla": "1 hour",
        "actions": ["enhanced_verification", "transaction_review", "customer_education"]
    },
    "medium": {
        "team": "Customer Protection Team",
        "contact": "customer.protection@hsbc.com", 
        "sla": "24 hours",
        "actions": ["monitoring_flag", "customer_education", "follow_up"]
    },
    "low": {
        "team": "Customer Service",
        "contact": "customer.service@hsbc.com",
        "sla": "Standard",
        "actions": ["standard_processing", "documentation"]
    }
}

# Model performance settings
MODEL_PERFORMANCE = {
    "gemini-2.0-flash": {
        "latency": "low",           # <2 seconds
        "cost": "low",              # Cost-effective for high-volume
        "use_cases": ["real_time_detection", "quick_summaries"]
    },
    "gemini-2.0-flash": {
        "latency": "medium",        # 2-5 seconds  
        "cost": "medium",           # Higher cost but better reasoning
        "use_cases": ["complex_analysis", "policy_reasoning", "decision_making"]
    }
}
