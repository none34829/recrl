#!/usr/bin/env python
"""
Simple validation script to test if the trained Shielded RecRL model works.
"""
import torch
import json
import argparse
from pathlib import Path
import sys
sys.path.append('..')

from trainer.shielded_ppo_trainer import ShieldedPPO

def validate_model(model_path, dataset="books"):
    """Test if the trained model generates reasonable explanations."""
    
    print(f"Loading model from: {model_path}")
    
    # Initialize the agent
    agent = ShieldedPPO(dataset=dataset, proj=Path("."))
    
    # Load the trained checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    print("Model loaded successfully!")
    
    # Test prompts
    test_prompts = [
        "User likes fantasy novels like LOTR and Harry Potter. Explain why we recommend Mistborn.",
        "User enjoys epic fantasy with magic systems. Explain why we recommend The Way of Kings.",
        "User loves character-driven fantasy stories. Explain why we recommend The Name of the Wind.",
        "User prefers dark fantasy with complex plots. Explain why we recommend The First Law trilogy."
    ]
    
    print("\n" + "="*60)
    print("TESTING EXPLANATION GENERATION")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt}")
        
        # Generate explanation
        with torch.no_grad():
            generated = agent.generate([prompt], max_new_tokens=50, do_sample=True, temperature=0.7)
        
        explanation = generated[0]
        print(f"Explanation: {explanation}")
        
        # Simple quality check
        words = explanation.split()
        print(f"Length: {len(words)} words")
        
        # Check for relevant keywords
        relevant_words = ['fantasy', 'magic', 'story', 'character', 'book', 'series', 'author']
        found_words = [w for w in relevant_words if w.lower() in explanation.lower()]
        print(f"Relevant keywords found: {found_words}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--dataset", default="books", help="Dataset name")
    
    args = parser.parse_args()
    validate_model(args.model_path, args.dataset)
