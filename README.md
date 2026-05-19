# 🎬 Sentiment Analysis — Movie Reviews

PyTorch bidirectional LSTM classifying IMDB movie reviews as positive/negative, demonstrating word embeddings and recurrent neural networks.

## Overview
| Detail | Value |
|--------|-------|
| Type | Binary Text Classification |
| Dataset | IMDB (50,000 reviews, auto-downloaded) |
| Framework | PyTorch |
| Architecture | Embedding → BiLSTM (2 layers) → FC |

## Getting Started
```bash
git clone https://github.com/Dnshitobu/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
python sentiment_analysis.py
```

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | ~86% |
| ROC-AUC | ~0.94 |

## Concepts Covered
Word embeddings · Bidirectional LSTM · BCEWithLogitsLoss · Gradient clipping · ReduceLROnPlateau
