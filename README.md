# Foodie Finder

Foodie Finder is a machine learning project for restaurant review sentiment analysis and restaurant recommendation. The system collects review data from Foody.vn, cleans noisy reviews, trains multiple classical ML sentiment models, then uses the best model to score and recommend restaurants in a web demo.

## Data Scraping

The scraping pipeline collects restaurant metadata and user reviews from Foody.vn. It supports browser-based crawling, dynamic page loading, retry handling, checkpointing, and JSON/JSONL output.

Main scraping outputs:
- Restaurant data
- Review data
- Restaurant-review mapping
- Cleaned review dataset for ML

The raw review data is evaluated and cleaned by removing reviews detected as duplicate, too short, spam/ad-like, or rating-content mismatch.

## Main ML Project

The main ML task is sentiment analysis on restaurant reviews. Each review is classified into sentiment labels such as negative, neutral, or positive based on the review rating and content.

Before training, reviews with `rating == 10` are removed to reduce bias from overly perfect ratings.

### Models

The project compares these machine learning models:

1. TF-IDF + Naive Bayes
2. TF-IDF + Logistic Regression
3. TF-IDF + Linear SVM
4. TF-IDF + Random Forest
5. TF-IDF + XGBoost

### Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision by class
- Recall by class
- F1-score by class
- Macro F1
- Weighted F1
- Confusion Matrix

Metric formulas:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1-score = 2 * Precision * Recall / (Precision + Recall)

Macro F1 = average(F1-score of all classes)

Weighted F1 = sum(F1_class * support_class) / total_samples
```

Where:
- `TP`: correctly predicted samples of a class
- `FP`: samples incorrectly predicted as that class
- `FN`: samples of that class predicted as another class
- `TN`: samples correctly predicted as not belonging to that class
- `support_class`: number of true samples in that class

After all models are implemented and evaluated, the best model will be selected based mainly on Macro F1 and Weighted F1, then integrated into the web demo.

### Algorithm Overview

All models use TF-IDF features. TF-IDF converts review text into numerical vectors by increasing the weight of words that are frequent in one review but less common across the full dataset. This helps the model focus on more informative words instead of very common words.

- **Naive Bayes:** a probabilistic classifier based on Bayes' theorem. It assumes words/features are conditionally independent given the sentiment class. This assumption is simple but often works well for text classification.
- **Logistic Regression:** a linear classifier that learns feature weights and estimates the probability of each sentiment class. It is strong for sparse TF-IDF text features and easy to interpret.
- **Linear SVM:** a margin-based linear classifier that tries to separate classes with the largest possible margin. It is commonly effective for high-dimensional text classification.
- **Random Forest:** an ensemble of decision trees trained on different data/feature samples. It can model non-linear patterns, but may be less efficient on very sparse TF-IDF vectors.
- **XGBoost:** a gradient boosting model that builds trees sequentially, where each new tree corrects errors from previous trees. It is powerful for structured features and will be tested against the linear text baselines.

## Current Naive Bayes Pipeline

The Naive Bayes baseline is implemented in:

```text
algorithm/naive_bayes/
```

Main files:
- `dataset.py`: load clean review data, remove `rating == 10`, split train/test
- `model.py`: define TF-IDF + Naive Bayes pipeline
- `train.py`: train and save the model
- `evaluate.py`: evaluate the saved model and export metrics/confusion matrix

Run:

```bash
python algorithm/naive_bayes/train.py
python algorithm/naive_bayes/evaluate.py
```

## Web Demo: Foodie Finder

The final selected model will be integrated into a web demo named `foodie-finder`.

The demo will recommend the top 10 restaurants near the user's location. Restaurant ranking will be based on a combined score using:

- Total number of reviews
- Number of high-quality reviews
- Predicted sentiment distribution
- Review quality signals
- User preference fields such as:
  - Food quality
  - Service
  - Price
  - Space / atmosphere

The scoring formula will adapt to the user's needs. For example, if the user cares more about price and service, those fields will receive higher weights in the final restaurant score.

## New Restaurant Handling

If a user searches for a restaurant that is not available in the local database but exists on Foody.vn, the system will:

1. Crawl the restaurant's reviews from Foody.vn
2. Clean and preprocess the reviews
3. Run the selected sentiment model
4. Calculate the restaurant score
5. Summarize the restaurant's strengths and weaknesses
6. Show the result in the recommendation/demo interface

## Project Goal

The goal is to build an end-to-end Vietnamese restaurant recommendation system that combines review crawling, sentiment analysis, restaurant scoring, and location-based recommendation into a usable web demo.
