#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature building script.
This script handles the extraction and preparation of features for the sentiment analysis models.
"""

import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import json

# Download NLTK resources if not already present
nltk.download(['punkt', 'wordnet', 'stopwords'], quiet=True)

def clear_content(content):
    """
    Clean and process text data for feature extraction.
    
    Args:
        content: Raw text content
        
    Returns:
        Cleaned and processed text
    """
    if not isinstance(content, str) or pd.isna(content):
        return ""
        
    # Step 1: Expand contractions
    content = contractions.fix(content)
    
    # Step 2: Convert text to lowercase
    content = content.lower()
    
    # Step 3: Remove special characters
    content = re.sub(r'[^a-zA-Z\s]', '', content)
    
    # Step 4: Tokenization
    tokens = word_tokenize(content)
    
    # Step 5: Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleared = []
    for word in tokens:
        if (word not in stop_words) and len(word) > 2:  # Exclude stopwords and small words
            cleared.append(lemmatizer.lemmatize(word))
    
    return ' '.join(cleared)

def process_sub_comments(sub_comments_json):
    """
    Process JSON sub-comments to extract additional features.
    
    Args:
        sub_comments_json: JSON string containing sub-comments
        
    Returns:
        Dictionary of extracted features
    """
    if not isinstance(sub_comments_json, str) or pd.isna(sub_comments_json):
        return {}
    
    try:
        sub_comments = json.loads(sub_comments_json)
        
        # Extract features from sub-comments
        features = {
            'num_subcomments': len(sub_comments),
            'positive_count': sum(1 for comment in sub_comments if comment.get('sentiment') == 'positive'),
            'negative_count': sum(1 for comment in sub_comments if comment.get('sentiment') == 'negative'),
            'neutral_count': sum(1 for comment in sub_comments if comment.get('sentiment') == 'neutral')
        }
        
        return features
    except Exception as e:
        print(f"Error processing sub-comments: {e}")
        return {}

def build_features():
    """
    Main function to build features from preprocessed data.
    """
    # Input and output file paths
    input_file = os.path.join('data', 'processed', 'processed_data.csv')
    output_file = os.path.join('data', 'processed', 'processed_reviews.csv')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return
    
    # Read preprocessed data
    print(f"Reading preprocessed data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Clean text content
    print("Cleaning text content...")
    df['cleaned_content'] = df['content'].apply(clear_content)
    
    # Process sub-comments to extract additional features
    print("Processing sub-comments...")
    sub_features = df['sub_comments'].apply(process_sub_comments)
    
    # Convert dict series to DataFrame and join with original data
    sub_features_df = pd.DataFrame(sub_features.tolist())
    if not sub_features_df.empty:
        df = pd.concat([df, sub_features_df], axis=1)
    
    # Prepare sentiment labels
    print("Preparing sentiment labels...")
    df['sentiment'] = df['score'].apply(
        lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral'
    )
    
    # Filter out neutral reviews if needed
    df_filtered = df[df['sentiment'] != 'neutral']
    
    # Select and reorder columns
    columns_to_keep = ['content', 'cleaned_content', 'sentiment', 'score']
    
    # Add any additional columns from sub-comments if they exist
    for col in sub_features_df.columns:
        if col not in columns_to_keep:
            columns_to_keep.append(col)
    
    # Keep only columns that exist in the DataFrame
    columns_to_keep = [col for col in columns_to_keep if col in df_filtered.columns]
    df_final = df_filtered[columns_to_keep]
    
    # Save processed data
    print(f"Saving processed data to {output_file}...")
    df_final.to_csv(output_file, index=False)
    print(f"Feature building complete! Processed data saved to {output_file}")

if __name__ == "__main__":
    build_features() 