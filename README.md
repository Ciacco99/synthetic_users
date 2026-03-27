# Synthetic Users for Market Research: Correcting LLM Distributional Bias with Minimal Human Data

**Jacopo Ferro** | EPFL IC Master Thesis (PDM), 2026

Supervised by Prof. Davide Bavato, EPFL and Laurent Rochat, [Innovation Atelier](https://www.innovationatelier.com/ )

---

## Abstract

Traditional market research surveys are expensive, with typical concept tests requiring 150--300 respondents per product variant. This thesis investigates whether Large Language Models can generate synthetic survey responses that recover the true preference distribution of a population, using only a small seed sample of real human data to correct for systematic LLM bias. We apply a confusion-matrix correction method to a real consumer preference dataset (the Great American Coffee Taste Test, n=4,042) and show that with as few as 100 real responses, the corrected synthetic distribution achieves over 91% reduction in Total Variation Distance compared to the uncorrected LLM output.

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   ├── 01_data_analysis.py        # EDA on GACTT dataset, produces cleaned data
│   ├── 02_data_split.py           # Stratified seed/holdout splits
│   ├── 03_llm_synthesis.py        # LLM synthesis on holdout personas (n=700)
│   ├── 03b_seed_synthesis.py      # LLM synthesis on seed personas (m=50,100,150)
│   ├── 04_correction.py           # Confusion matrix learning + correction
│   ├── 05_bootstrap_ci.py         # Bootstrap confidence intervals (B=1000)
│   ├── 06_final_figures.py        # Generate all thesis figures
│   └── extra_exploration.py       # Extended dataset analysis (standalone)
├── data/
│   ├── GACTT_RESULTS_ANONYMIZED_v2.csv   # Original GACTT dataset
│   ├── coffee_data_cleaned.csv           # Cleaned version (output of script 01)
│   └── splits/
│       ├── seed_m50.csv
│       ├── seed_m100.csv
│       ├── seed_m150.csv
│       └── holdout_n700.csv
└── results/
    ├── synthetic/                  # LLM predictions
    ├── correction/                 # Correction matrix, distributions, metrics
    ├── bootstrap/                  # Bootstrap results, confidence intervals
    └── figures/
        ├── eda/                    # 5 exploratory data analysis figures (PNG)
        ├── exploration/            # 10 extended analysis figures (PNG)
        └── final/                  # 5 thesis figures (PNG)
```

## Reproducing the Results

### Prerequisites

- Python 3.11+
- An OpenAI API key (for scripts 03 and 03b)

### Setup

```bash
git clone https://github.com/Ciacco99/synthetic_users.git
cd synthetic_users
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

### Pipeline

Run scripts in order from the project root:

```bash
python scripts/01_data_analysis.py          # Clean data, produce EDA figures (results/figures/eda/)
python scripts/02_data_split.py             # Create seed and holdout splits
python scripts/03b_seed_synthesis.py --seed data/splits/seed_m100.csv
python scripts/03_llm_synthesis.py --holdout data/splits/holdout_n700.csv
python scripts/04_correction.py --seed results/synthetic/seed_predictions_m100.csv \
                                --synthetic results/synthetic/synthetic_predictions.csv
python scripts/05_bootstrap_ci.py --seed results/synthetic/seed_predictions_m100.csv \
                                  --synthetic results/synthetic/synthetic_predictions.csv \
                                  --n_bootstrap 1000
python scripts/06_final_figures.py          # Generate thesis figures
```

The repository ships with all results pre-computed. Re-running the LLM synthesis scripts (03, 03b) requires API access and will produce slightly different outputs due to LLM non-determinism.

## Data Source

The dataset used is the **Great American Coffee Taste Test (GACTT)** by [James Hoffmann](https://www.youtube.com/watch?v=bMOOQfeloH0), collected via a large-scale public blind tasting event with approximately 4,000 US participants evaluating four coffees. The anonymized survey results are publicly available.

## Key Results

| Seed Size (m) | TVD (Uncorrected) | TVD (Corrected) | Improvement |
|:-:|:-:|:-:|:-:|
| 50 | 0.154 | 0.063 | 59.4% |
| 100 | 0.154 | 0.013 | 91.4% |
| 150 | 0.154 | 0.005 | 97.0% |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
