"""Reward functions for the Shielded RecRL trainer.

This module provides reward functions for the PPO trainer. The current implementation
includes a simple toy reward function that rewards explanations based on length.
In Section 9, this will be replaced with live CTR from RecSim-NG.
"""

def click_reward(explanation:str) -> float:
    """
    Toy reward: +1 if explanation length > 20 tokens, else 0.
    Replace with live CTR in Section 9.
    
    Args:
        explanation: The generated explanation text
        
    Returns:
        float: 1.0 if explanation has more than 20 tokens, 0.0 otherwise
    """
    return float(len(explanation.split()) > 20)

def improved_click_reward(explanation: str) -> float:
    """
    Improved reward function that provides more granular feedback.
    
    Args:
        explanation: The generated explanation text
        
    Returns:
        float: Reward between 0.0 and 1.0 based on multiple criteria
    """
    words = explanation.split()
    word_count = len(words)
    
    # Base reward for length (0.0 to 0.5)
    length_reward = min(word_count / 40.0, 0.5)  # Max 0.5 for 40+ words
    
    # Content reward for relevant keywords (0.0 to 0.3)
    relevant_keywords = ['mistborn', 'sanderson', 'fantasy', 'magic', 'book', 'series', 'story', 'character', 'world', 'recommend', 'like', 'similar']
    found_keywords = sum(1 for word in words if word.lower() in relevant_keywords)
    content_reward = min(found_keywords * 0.1, 0.3)  # 0.1 per keyword, max 0.3
    
    # Coherence reward for having complete sentences (0.0 to 0.2)
    sentences = explanation.split('.')
    complete_sentences = sum(1 for s in sentences if len(s.strip().split()) >= 3)
    coherence_reward = min(complete_sentences * 0.1, 0.2)  # 0.1 per complete sentence, max 0.2
    
    total_reward = length_reward + content_reward + coherence_reward
    return min(total_reward, 1.0)  # Cap at 1.0

def keyword_reward(text: str, keywords=("Mistborn","Sanderson","magic","fantasy")) -> float:
    """
    Simple keyword-based reward for smoke testing.
    Returns 1.0 if any of the keywords appear in the text, 0.0 otherwise.
    
    Args:
        text: The generated text to check
        keywords: Tuple of keywords to look for
        
    Returns:
        float: 1.0 if any keyword found, 0.0 otherwise
    """
    t = text.lower()
    return float(any(k.lower() in t for k in keywords))