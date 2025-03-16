# sentiment_keyword_analysis.py

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Machine Learning components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Keyword extraction tools
from rake_nltk import Rake
from keybert import KeyBERT

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP explainability tool
import shap

# Download NLTK data if not already downloaded
nltk.download(['punkt', 'wordnet', 'stopwords'])

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans and preprocesses text data."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = word_tokenize(text)  # Tokenize text into words
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(cleaned)

def extract_rake_keywords(texts, top_n=20):
    """Extracts keywords using the RAKE algorithm."""
    r = Rake()
    keywords = []
    for text in texts:
        r.extract_keywords_from_text(text)
        keywords.extend(r.get_ranked_phrases())
    return pd.Series(keywords).value_counts().head(top_n)

def extract_keybert_keywords(texts, top_n=20):
    """Extracts keywords using KeyBERT."""
    combined_text = ' '.join(texts)
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=top_n
    )
    return dict(keywords)

def main():
    # Load dataset (replace 'reviews.csv' with your file path)
    df = pd.read_csv('reviews.csv')

    # Drop missing values and preprocess data
    df = df.dropna(subset=['review_text', 'rating'])
    
    # Create sentiment labels (positive: 4-5, negative: 1-2)
    df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')
    
    # Filter out neutral reviews (optional)
    df = df[df['sentiment'] != 'neutral']
    
    # Clean the review text
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    # Display basic stats about the dataset
    print(f"Dataset shape after preprocessing: {df.shape}")
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train logistic regression model for sentiment classification
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Evaluate model performance on test data
    y_pred = lr_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Extract feature importance from logistic regression coefficients
    feature_names = tfidf.get_feature_names_out()
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr_model.coef_[0]
    })
    
    positive_keywords = coef_df.sort_values('coefficient', ascending=False).head(20)
    negative_keywords = coef_df.sort_values('coefficient').head(20)
    
    print("\nTop Positive Keywords:")
    print(positive_keywords['feature'].tolist())
    
    print("\nTop Negative Keywords:")
    print(negative_keywords['feature'].tolist())
    
    # Extract keywords using RAKE for positive and negative reviews separately
    positive_rake_keywords = extract_rake_keywords(df[df['sentiment'] == 'positive']['cleaned_text'])
    negative_rake_keywords = extract_rake_keywords(df[df['sentiment'] == 'negative']['cleaned_text'])
    
    print("\nRAKE Positive Keywords:")
    print(positive_rake_keywords.index.tolist())
    
    print("\nRAKE Negative Keywords:")
    print(negative_rake_keywords.index.tolist())
    
    # Extract keywords using KeyBERT for positive and negative reviews separately
    positive_keybert_keywords = extract_keybert_keywords(df[df['sentiment'] == 'positive']['cleaned_text'])
    negative_keybert_keywords = extract_keybert_keywords(df[df['sentiment'] == 'negative']['cleaned_text'])
    
    print("\nKeyBERT Positive Keywords:")
    print(list(positive_keybert_keywords.keys()))
    
    print("\nKeyBERT Negative Keywords:")
    print(list(negative_keybert_keywords.keys()))
    
if __name__ == "__main__":
    main()
