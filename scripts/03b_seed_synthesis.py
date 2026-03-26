#!/usr/bin/env python3
"""
03b_seed_synthesis.py
=====================
Generate LLM predictions for the seed data.
This is needed to build the confusion matrix P(true | llm).

For each seed sample, we have the ground truth (y) and we generate
the LLM prediction (z), giving us the (y, z) pairs needed for correction.

Usage:
    python scripts/03b_seed_synthesis.py --seed data/splits/seed_m100.csv
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "synthetic"

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 50

COFFEE_CHOICES = ["Coffee A", "Coffee B", "Coffee C", "Coffee D"]

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


def generate_persona_description(row):
    """Generate a text description of a survey respondent from their row data."""
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
    """Build the complete few-shot prompt for a given respondent."""
    examples_text = "Here are examples of how different coffee drinkers chose their favorite in a blind tasting:\n\n"
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"Example {i}:\n{ex['persona']}\nFavorite coffee: {ex['choice']}\n\n"
    
    persona_desc = generate_persona_description(row)
    
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


def main():
    parser = argparse.ArgumentParser(description='Generate LLM predictions for seed data')
    parser.add_argument('--seed', type=str, required=True, help='Path to seed CSV')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Seed Data LLM Synthesis")
    print("=" * 60)
    
    client = OpenAI()
    print(f"Model: {MODEL}, Temperature: {TEMPERATURE}")
    
    df = pd.read_csv(args.seed)
    seed_size = len(df)
    print(f"Loaded seed data: {seed_size} rows")
    
    predictions = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{seed_size}")
        
        prompt = build_prompt(row)
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            llm_response = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API Error: {e}")
            llm_response = None
        
        llm_choice = parse_response(llm_response)
        predictions.append({
            'index': idx,
            'ground_truth': row['overall_favorite'],
            'llm_response': llm_response,
            'llm_choice': llm_choice
        })
        time.sleep(0.1)
    
    results = pd.DataFrame(predictions)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"seed_predictions_m{seed_size}.csv"
    results.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"\nGround Truth Distribution:")
    print(results['ground_truth'].value_counts().to_string())
    print(f"\nLLM Prediction Distribution:")
    print(results['llm_choice'].value_counts().to_string())


if __name__ == "__main__":
    main()
