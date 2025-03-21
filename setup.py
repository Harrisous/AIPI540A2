#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for the project.
This script sets up the project by getting data, building features, and training models.
"""

import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.make_dataset import get_data
from scripts.preprocess_data import preprocess
from scripts.build_features import build_features
from scripts.model import train_models

def main():
    """
    Main function to set up the project.
    """
    print("Setting up the project...")
    
    # Create necessary directories if they don't exist
    dirs = ['data/raw', 'data/processed', 'data/outputs', 'models']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Get data
    print("Getting data...")
    get_data()
    
    # Preprocess data
    print("Preprocessing data...")
    preprocess()
    
    # Build features
    print("Building features...")
    build_features()
    
    # Train model
    print("Training models...")
    processed_data_path = os.path.join('data', 'processed', 'processed_reviews.csv')
    if os.path.exists(processed_data_path):
        train_models(processed_data_path)
    else:
        print(f"Error: Processed data file not found at {processed_data_path}")
    
    print("Setup complete!")

if __name__ == "__main__":
    main() 