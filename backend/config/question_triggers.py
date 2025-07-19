# backend/config/question_triggers.py
"""
Question triggers for pattern-based progressive question display
Maps fraud patterns to contextual questions for agents
"""

from typing import Dict, List, Any

# Pattern-triggered questions with realistic context
QUESTION_TRIGGERS = {
    'romance_exploitation': [
        {
            'trigger_phrases': ['online', 'internet', 'dating', 'met online'],
            'question': 'How long have you known this person?',
            'context': 'Online relationship detected',
            'urgency': 'medium',
            'risk_threshold': 40
        },
        {
            'trigger_phrases': ['never met', 'not met', "haven't met"],
            'question': 'Have you met this person in person?',
            'context': 'Customer has not met recipient',
            'urgency': 'high',
            'risk_threshold': 60
        },
        {
            'trigger_phrases': ['emergency', 'stuck', 'hospital', 'urgent'],
            'question': 'Why is this request urgent?',
            'context': 'Emergency request detected',
            'urgency': 'high',
            'risk_threshold': 70
        },
        {
            'trigger_phrases': ['secret', 'private', "don't tell"],
            'question': 'Are they asking you to keep this secret?',
            'context': 'Secrecy request detected',
            'urgency': 'high',
            'risk_threshold': 75
        }
    ],
    
    'investment_scam': [
        {
            'trigger_phrases': ['guaranteed', 'guaranteed returns', 'no risk'],
            'question': 'What regulatory body oversees this investment?',
            'context': 'Guaranteed returns claimed',
            'urgency': 'high',
            'risk_threshold': 60
        },
        {
            'trigger_phrases': ['today only', 'limited time', 'expires'],
            'question': 'Why is this opportunity time-sensitive?',
            'context': 'Time pressure detected',
            'urgency': 'medium',
            'risk_threshold': 50
        },
        {
            'trigger_phrases': ['crypto', 'bitcoin', 'trading platform'],
            'question': 'Have you verified this platform is legitimate?',
            'context': 'Cryptocurrency investment detected',
            'urgency': 'medium',
            'risk_threshold': 55
        }
    ],
    
    'impersonation_scam': [
        {
            'trigger_phrases': ['security team', 'fraud department', 'bank security'],
            'question': 'What department do they claim to be from?',
            'context': 'Impersonation attempt detected',
            'urgency': 'high',
            'risk_threshold': 70
        },
        {
            'trigger_phrases': ['PIN', 'password', 'security code'],
            'question': 'Did they ask for your PIN or passwords?',
            'context': 'Credential request detected',
            'urgency': 'high',
            'risk_threshold': 80
        }
    ],
    
    'authority_scam': [
        {
            'trigger_phrases': ['police', 'court', 'legal action', 'arrest'],
            'question': 'What is their badge number or court reference?',
            'context': 'Authority impersonation detected',
            'urgency': 'high',
            'risk_threshold': 80
        },
        {
            'trigger_phrases': ['fine', 'penalty', 'immediate payment'],
            'question': 'Why must payment be made immediately?',
            'context': 'Immediate payment demand',
            'urgency': 'high',
            'risk_threshold': 85
        }
    ]
}

def get_questions_for_pattern(pattern_name: str, risk_score: float, customer_text: str) -> List[Dict[str, Any]]:
    """
    Get relevant questions for a detected pattern
    """
    pattern_questions = QUESTION_TRIGGERS.get(pattern_name, [])
    relevant_questions = []
    
    customer_text_lower = customer_text.lower()
    
    for question_config in pattern_questions:
        # Check if risk threshold is met
        if risk_score >= question_config['risk_threshold']:
            # Check if trigger phrases are present
            phrase_matches = []
            for phrase in question_config['trigger_phrases']:
                if phrase.lower() in customer_text_lower:
                    phrase_matches.append(phrase)
            
            if phrase_matches:
                relevant_questions.append({
                    **question_config,
                    'matched_phrases': phrase_matches,
                    'pattern': pattern_name
                })
    
    return relevant_questions

def select_best_question(detected_patterns: Dict, risk_score: float, customer_text: str) -> Dict[str, Any]:
    """
    Select the best question to show based on patterns and risk
    """
    all_questions = []
    
    for pattern_name in detected_patterns.keys():
        questions = get_questions_for_pattern(pattern_name, risk_score, customer_text)
        all_questions.extend(questions)
    
    if not all_questions:
        return None
    
    # Sort by urgency and risk threshold
    urgency_priority = {'high': 3, 'medium': 2, 'low': 1}
    all_questions.sort(
        key=lambda q: (urgency_priority.get(q['urgency'], 0), q['risk_threshold']),
        reverse=True
    )
    
    return all_questions[0]
