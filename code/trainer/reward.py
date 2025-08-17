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