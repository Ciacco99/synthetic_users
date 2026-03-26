#!/usr/bin/env python3
"""
03_llm_synthesis.py
===================
Generate synthetic survey responses using an LLM with few-shot prompting.

This script takes personas from the holdout set and generates synthetic
coffee preference predictions using an LLM (gpt-4.1-mini).

Usage:
    python scripts/03_llm_synthesis.py --holdout data/splits/holdout_n700.csv

Output:
    - results/synthetic/synthetic_predictions.csv

Requirements:
    - OPENAI_API_KEY environment variable set
    - openai package installed (pip install openai)
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "synthetic"

# LLM settings
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 50

# Coffee choices
COFFEE_CHOICES = ["Coffee A", "Coffee B", "Coffee C", "Coffee D"]

# Few-shot examples (fixed for reproducibility)
FEW_SHOT_EXAMPLES = [
    {
        "persona": "Age: 25-34\nGender: Male\nCoffee expertise: 8/10\nFlavor preference: Fruity\nRoast preference: Light",
        "choice": "Coffee D"
    },
    {
        "persona": "Age: 35-44\nGender: Female\nCoffee expertise: 3/10\nFlavor preference: Chocolatey\nRoast preference: Medium",
        "choice": "Coffee B"
    },
    {
        "persona": "Age: 45-54\nGender: Male\nCoffee expertise: 5/10\nFlavor preference: Bold\nRoast preference: Dark",
        "choice": "Coffee C"
    },
    {
        "persona": "Age: 25-34\nGender: Female\nCoffee expertise: 7/10\nFlavor preference: Bright, citrusy\nRoast preference: Light",
        "choice": "Coffee A"
    },
    {
        "persona": "Age: 55-64\nGender: Male\nCoffee expertise: 9/10\nFlavor preference: Complex, unique\nRoast preference: Light",
        "choice": "Coffee D"
    }
]

# -----------------------------------------------------------------------------
# Prompt Construction
# -----------------------------------------------------------------------------

def generate_persona_description(row):
    """Generate a text description of a survey respondent."""
    parts = []
    
    if pd.notna(row.get('age')):
        parts.append(f"Age: {row['age']}")
    if pd.notna(row.get('gender')):
        parts.append(f"Gender: {row['gender']}")
    if pd.notna(row.get('expertise')):
        parts.append(f"Coffee expertise: {int(row['expertise'])}/10")
    if pd.notna(row.get('flavor_preference')):
        parts.append(f"Flavor preference: {row['flavor_preference']}")
    if pd.notna(row.get('roast_preference')):
        parts.append(f"Roast preference: {row['roast_preference']}")
    
    return "\n".join(parts) if parts else "General coffee drinker"


def build_prompt(row):
    """Build the complete prompt with few-shot examples."""
    # Build few-shot examples section
    examples_text = "Here are examples of how different coffee drinkers chose their favorite in a blind tasting:\n\n"
    
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"Example {i}:\n{ex['persona']}\nFavorite coffee: {ex['choice']}\n\n"
    
    # Build persona description for target
    persona_desc = generate_persona_description(row)
    
    # Complete prompt
    prompt = f"""{examples_text}
Now, consider this person participating in the same blind coffee tasting:

{persona_desc}

Based on this person's profile and preferences, which of the four coffees (A, B, C, or D) would they most likely choose as their favorite?

Important: Different people have genuinely different tastes. Some love bold, experimental flavors (like fermented/natural process coffees) while others prefer classic, familiar tastes. Answer authentically for THIS specific person.

Respond with only the coffee name (e.g., "Coffee A", "Coffee B", "Coffee C", or "Coffee D").
"""
    return prompt


def parse_response(response):
    """Parse the LLM response to extract the coffee choice."""
    if not response:
        return None
    
    response_upper = response.upper()
    for coffee in COFFEE_CHOICES:
        if coffee.upper() in response_upper:
            return coffee
    
    return None


# -----------------------------------------------------------------------------
# LLM Calling
# -----------------------------------------------------------------------------

def call_llm(client, prompt):
    """Call the LLM API and return the response."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API Error: {e}")
        return None


def generate_predictions(df, client, progress_interval=50):
    """Generate predictions for all rows in the dataframe."""
    predictions = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % progress_interval == 0:
            print(f"  Progress: {i+1}/{len(df)}")
        
        prompt = build_prompt(row)
        response = call_llm(client, prompt)
        choice = parse_response(response)
        
        predictions.append({
            'index': idx,
            'ground_truth': row['overall_favorite'],
            'llm_response': response,
            'llm_choice': choice
        })
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return pd.DataFrame(predictions)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic predictions with LLM')
    parser.add_argument('--holdout', type=str, required=True, help='Path to holdout CSV')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of predictions (for testing)')
    args = parser.parse_args()
    
    print("="*60)
    print("LLM Synthesis: Generating Synthetic Predictions")
    print("="*60)
    
    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize client
    client = OpenAI()
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    
    # Load holdout data
    holdout_path = Path(args.holdout)
    if not holdout_path.exists():
        print(f"Error: Holdout file not found at {holdout_path}")
        return
    
    df = pd.read_csv(holdout_path)
    print(f"Loaded holdout data: {len(df)} rows")
    
    # Limit if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} rows for testing")
    
    # Generate predictions
    print("\nGenerating predictions...")
    results = generate_predictions(df, client)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "synthetic_predictions.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total predictions: {len(results)}")
    print(f"Valid predictions: {results['llm_choice'].notna().sum()}")
    print(f"\nGround Truth Distribution:")
    print(results['ground_truth'].value_counts().to_string())
    print(f"\nLLM Prediction Distribution:")
    print(results['llm_choice'].value_counts().to_string())


if __name__ == "__main__":
    main()
