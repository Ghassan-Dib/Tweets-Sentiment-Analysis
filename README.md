# Tweet Sentiment Classification Using Support Vector Machine (SVM)

This repository implements a Support Vector Machine (SVM) classifier to classify tweets as having positive or negative sentiment. The process involves data preprocessing, feature extraction, and model training, followed by performance evaluation using cross-validation and error analysis.

## Features

### 1. Data Preprocessing
Two preprocessing approaches were implemented:
- **Simple Preprocessing**: Tokenizes the text into words.
- **Enhanced Preprocessing**: Applies advanced cleaning techniques, including:
  - Removing punctuation and stopwords
  - Lowercasing
  - Tokenization
  - Stemming and lemmatization

### 2. Feature Extraction
Feature extraction methods used:
- **Bag of Words**: Counts word occurrences to create a feature vector.
- **TF-IDF with N-grams**: Uses unigrams and bigrams for context-aware features, emphasizing important terms while reducing the weight of common ones.

### 3. Model Training
- **Algorithm**: Linear Support Vector Classifier (SVC)
- **Cross-Validation**: k-fold cross-validation was used to evaluate model performance, measuring accuracy, precision, recall, and F1 score.

### 4. Error Analysis
- **Approach 1**: Used simple preprocessing and basic feature extraction.
  - Achieved 82.88% accuracy and an F1 score of 82.74%.
  - Highlighted misclassification issues due to imbalanced data.
- **Approach 2**: Enhanced preprocessing and feature extraction.
  - Best configuration: TF-IDF with unigrams and bigrams, achieving 85.95% accuracy and an F1 score of 86.3%.
  - Balanced false positives and false negatives, showing improved contextual understanding.

## Results
- Best performance achieved using TF-IDF with unigrams and bigrams.
- Accuracy: **85.95%**
- F1 Score: **86.3%**

## Data
The dataset contains approximately 33,540 tweets labeled as positive or negative. Data visualization revealed a class imbalance, with 65% of tweets labeled as positive.

## Dependencies
- Python 3.x
- scikit-learn
- NLTK
- NumPy
- Matplotlib

## Usage
1. Preprocess the dataset using the `data_preprocessing` functions.
2. Extract features using `bag_of_words` or `tfidf_ngrams`.
3. Train and evaluate the SVM model using `model_training`.
4. Perform error analysis to understand model limitations.
