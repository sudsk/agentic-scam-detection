# backend/config/question_triggers.py
"""
Question triggers for pattern-based progressive question display
Maps fraud patterns to contextual questions for agents
"""

from typing import Dict, List, Any

# Pattern-triggered questions with realistic context
QUESTION_TRIGGERS = {
    # Handle both formats: backend sends "Romance Exploitation", we also check "romance_exploitation"
    'Romance Exploitation': [
        {
            'trigger_phrases': ['online', 'internet', 'dating', 'met online'],
            'question': 'How long have you known this person?',
            'context': 'Online relationship detected',
            'urgency': 'medium',
            'risk_threshold': 30  # Lower threshold
        },
        {
            'trigger_phrases': ['never met', 'not met', "haven't met"],
            'question': 'Have you met this person in person?',
            'context': 'Customer has not met recipient',
            'urgency': 'high',
            'risk_threshold': 40
        },
        {
            'trigger_phrases': ['emergency', 'stuck', 'hospital', 'urgent'],
            'question': 'Why is this request urgent?',
            'context': 'Emergency request detected',
            'urgency': 'high',
            'risk_threshold': 35  # Lower threshold
        }
    ],
    
    'romance_exploitation': [  # Keep old format too
        {
            'trigger_phrases': ['online', 'internet', 'dating', 'met online'],
            'question': 'How long have you known this person?',
            'context': 'Online relationship detected',
            'urgency': 'medium',
            'risk_threshold': 30
        }
    ],
    
    'Urgency Pressure': [  # Add this - matches your backend pattern!
        {
            'trigger_phrases': ['urgent', 'emergency', 'immediately', 'right away', 'asap'],
            'question': 'Why is this transaction urgent?',
            'context': 'Urgency pressure detected',
            'urgency': 'medium',
            'risk_threshold': 25  # Match your current risk score
        },
        {
            'trigger_phrases': ['today only', 'limited time', 'expires'],
            'question': 'Why must this be done today?',
            'context': 'Time pressure detected',
            'urgency': 'high',
            'risk_threshold': 30
        }
    ],
    
    'urgency_pressure': [  # Keep old format
        {
            'trigger_phrases': ['urgent', 'emergency'],
            'question': 'Why is this transaction urgent?',
            'context': 'Urgency pressure detected',
            'urgency': 'medium',
            'risk_threshold': 25
        }
    ],
    
    'Investment Scam': [
        {
            'trigger_phrases': ['guaranteed', 'guaranteed returns', 'no risk'],
            'question': 'What regulatory body oversees this investment?',
            'context': 'Guaranteed returns claimed',
            'urgency': 'high',
            'risk_threshold': 40
        }
    ],
    
    'investment_scam': [  # Keep old format
        {
            'trigger_phrases': ['guaranteed', 'returns'],
            'question': 'What regulatory body oversees this investment?',
            'context': 'Investment scam detected',
            'urgency': 'high',
            'risk_threshold': 40
        }
    ]
}

def get_questions_for_pattern(pattern_name: str, risk_score: float, customer_text: str) -> List[Dict[str, Any]]:
    """Get relevant questions for a detected pattern with enhanced debugging"""
    #print(f"🔍 PATTERN DEBUG - get_questions_for_pattern called:")
    #print(f"  - pattern_name: '{pattern_name}'")
    #print(f"  - risk_score: {risk_score}")
    #print(f"  - customer_text: '{customer_text[:100]}...'")
    
    pattern_questions = QUESTION_TRIGGERS.get(pattern_name, [])
    #print(f"🔍 PATTERN DEBUG - Found {len(pattern_questions)} potential questions for '{pattern_name}'")
    
    if not pattern_questions:
        #print(f"🔍 PATTERN DEBUG - No questions found for pattern '{pattern_name}'")
        #print(f"🔍 PATTERN DEBUG - Available patterns: {list(QUESTION_TRIGGERS.keys())}")
        return []
    
    relevant_questions = []
    customer_text_lower = customer_text.lower()
    
    for i, question_config in enumerate(pattern_questions):
        #print(f"🔍 PATTERN DEBUG - Checking question {i+1}:")
        #print(f"  - question: {question_config['question']}")
        #print(f"  - risk_threshold: {question_config['risk_threshold']}")
        #print(f"  - trigger_phrases: {question_config['trigger_phrases']}")
        
        # Check if risk threshold is met
        if risk_score >= question_config['risk_threshold']:
            #print(f"  ✅ Risk threshold met: {risk_score} >= {question_config['risk_threshold']}")
            
            # Check if trigger phrases are present
            phrase_matches = []
            for phrase in question_config['trigger_phrases']:
                if phrase.lower() in customer_text_lower:
                    phrase_matches.append(phrase)
                    #print(f"  ✅ Found trigger phrase: '{phrase}'")
            
            if phrase_matches:
                #print(f"  ✅ Question qualifies! Matched phrases: {phrase_matches}")
                relevant_questions.append({
                    **question_config,
                    'matched_phrases': phrase_matches,
                    'pattern': pattern_name
                })
            else:
                print(f"  ❌ No trigger phrases found in text")
        else:
            print(f"  ❌ Risk threshold not met: {risk_score} < {question_config['risk_threshold']}")
    
    #print(f"🔍 PATTERN DEBUG - Returning {len(relevant_questions)} relevant questions")
    return relevant_questions

def select_best_question(detected_patterns: Dict, risk_score: float, customer_text: str) -> Dict[str, Any]:
    """Select the best question to show based on patterns and risk with enhanced debugging"""
    #print(f"🔍 QUESTION SELECT DEBUG - select_best_question called:")
    #print(f"  - detected_patterns: {detected_patterns}")
    #print(f"  - detected_patterns keys: {list(detected_patterns.keys()) if detected_patterns else 'None'}")
    #print(f"  - risk_score: {risk_score}")
    #print(f"  - customer_text: '{customer_text[:100]}...'")
    
    all_questions = []
    
    for pattern_name in detected_patterns.keys():
        #print(f"🔍 QUESTION SELECT DEBUG - Processing pattern: '{pattern_name}'")
        questions = get_questions_for_pattern(pattern_name, risk_score, customer_text)
        #print(f"🔍 QUESTION SELECT DEBUG - Got {len(questions)} questions for '{pattern_name}'")
        all_questions.extend(questions)
    
    #print(f"🔍 QUESTION SELECT DEBUG - Total questions collected: {len(all_questions)}")
    
    if not all_questions:
        #print(f"🔍 QUESTION SELECT DEBUG - No questions found, returning None")
        return None
    
    # Sort by urgency and risk threshold
    urgency_priority = {'high': 3, 'medium': 2, 'low': 1}
    all_questions.sort(
        key=lambda q: (urgency_priority.get(q['urgency'], 0), q['risk_threshold']),
        reverse=True
    )
    
    selected_question = all_questions[0]
    #print(f"🔍 QUESTION SELECT DEBUG - Selected question:")
    #print(f"  - question: {selected_question['question']}")
    #print(f"  - urgency: {selected_question['urgency']}")
    #print(f"  - pattern: {selected_question['pattern']}")
    #print(f"  - matched_phrases: {selected_question.get('matched_phrases', [])}")
    
    return selected_question
