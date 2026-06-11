# Airline Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-4C72B0.svg)](https://seaborn.pydata.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/bhadra0401/airline-sentiment-analysis)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Airline Sentiment Analysis is an end-to-end Natural Language Processing (NLP) application that classifies airline customer reviews into Positive, Neutral, and Negative sentiments. The project leverages TF-IDF feature extraction and Machine Learning techniques to provide real-time sentiment predictions and analytical insights through an interactive Streamlit dashboard.

## Live Application

**Live Demo:** https://huggingface.co/spaces/bhadra0401/airline-sentiment-analysis

## Features

* Real-time sentiment prediction
* NLP-based text preprocessing pipeline
* TF-IDF feature engineering
* Multi-model evaluation and comparison
* Interactive analytics dashboard
* Airline-wise sentiment analysis
* Bulk CSV sentiment prediction
* Downloadable prediction results
* Word cloud visualization
* Responsive Streamlit interface

## Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 79.85%   |
| Random Forest       | 77.29%   |
| Naive Bayes         | 73.77%   |

**Best Performing Model:** Logistic Regression

## Technology Stack

### Machine Learning & NLP

* Scikit-Learn
* TF-IDF Vectorization
* Logistic Regression
* Random Forest
* Naive Bayes

### Data Processing

* Pandas
* NumPy
* Regular Expressions

### Visualization

* Matplotlib
* Seaborn
* WordCloud

### Deployment

* Streamlit


## Project Structure

```text
Airline_Sentiment_Analysis/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── sentiment_model.pkl
├── vectorizer.pkl
└── Tweets.csv
```

## Workflow

1. Data Collection and Loading
2. Text Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training and Evaluation
5. Real-Time Prediction
6. Dashboard Visualization
7. Deployment using Streamlit and Hugging Face Spaces

## Installation

Clone the repository:

```bash
git clone https://github.com/bhadra0401/airline-sentiment-analysis.git
cd airline-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Results

* Developed a complete NLP sentiment analysis pipeline.
* Achieved 79.85% classification accuracy using Logistic Regression.
* Implemented an interactive web application for real-time inference.
* Deployed the solution on Hugging Face Spaces using Docker.

## Future Improvements

* Transformer-based sentiment analysis using BERT
* Multi-language support
* Advanced sentiment explainability
* Real-time streaming review analysis
* Cloud-native deployment architecture

## Author

**Naga Veera Bhadra Kumar Akkala**

LinkedIn: Add Your LinkedIn Profile
GitHub: Add Your GitHub Profile

## License

This project is licensed under the MIT License.
